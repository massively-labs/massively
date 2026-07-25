mod common;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::{
    op::BinaryPredicateOp,
    op::PredicateOp,
    op::ReductionOp,
    op::UnaryOp,
    seg::{
        AdjacentDifference, Executable, Filter, ForEachSegment, InclusiveScan, IsSorted, Map,
        Reduce, Reverse, SegmentIterator, Sort, Unique,
    },
    vector::inclusive_scan_by_key,
    vector::reduce_by_key,
};

struct AddOne;
struct Equal;
struct Even;
struct Less;
struct Sum;

#[cubecl::cube]
impl UnaryOp<u32> for AddOne {
    type Output = u32;

    fn apply(input: u32) -> u32 {
        input + 1u32
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Equal {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs == rhs)
    }
}

#[cubecl::cube]
impl PredicateOp<u32> for Even {
    fn apply(input: u32) -> massively::MFlag {
        massively::flag::from_bool(input % 2u32 == 0u32)
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

#[cubecl::cube]
impl ReductionOp<u32> for Sum {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

fn segment_lengths(len: usize) -> Vec<usize> {
    let mut lengths = vec![1, 8, 128, 4_096, len];
    for length in &mut lengths {
        *length = (*length).min(len);
    }
    lengths.sort_unstable();
    lengths.dedup();
    lengths
}

fn segment_name(len: usize, segment_len: usize) -> String {
    if segment_len == len {
        "all".into()
    } else {
        format!("run{segment_len}")
    }
}

fn offsets(len: usize, segment_len: usize) -> Vec<u32> {
    let mut result = Vec::with_capacity(len.div_ceil(segment_len) + 1);
    result.push(0);
    let mut offset = 0;
    while offset < len {
        offset = (offset + segment_len).min(len);
        result.push(offset as u32);
    }
    result
}

fn empty_heavy_offsets(len: usize) -> Vec<u32> {
    let mut result = Vec::with_capacity(len.div_ceil(8) * 2 + 2);
    result.push(0);
    let mut offset = 0;
    while offset < len {
        result.push(offset as u32);
        offset = (offset + 8).min(len);
        result.push(offset as u32);
    }
    result.push(offset as u32);
    result
}

fn alternating_offsets(len: usize) -> Vec<u32> {
    let mut result = Vec::with_capacity(len.div_ceil(2_048) + 1);
    result.push(0);
    let mut offset = 0;
    let mut short = true;
    while offset < len {
        let segment_len = if short { 1 } else { 4_095 };
        offset = (offset + segment_len).min(len);
        result.push(offset as u32);
        short = !short;
    }
    result
}

fn skewed_offsets(len: usize) -> Vec<u32> {
    const CYCLE: &[(usize, usize)] = &[
        (1, 64),
        (2, 32),
        (4, 16),
        (8, 8),
        (16, 4),
        (64, 2),
        (4_096, 1),
    ];

    let mut result = Vec::new();
    result.push(0);
    let mut offset = 0;
    while offset < len {
        for &(segment_len, repeat) in CYCLE {
            for _ in 0..repeat {
                if offset == len {
                    break;
                }
                offset = (offset + segment_len).min(len);
                result.push(offset as u32);
            }
        }
    }
    result
}

fn control_geometries(len: usize) -> Vec<(&'static str, Vec<u32>)> {
    vec![
        ("uniform8", offsets(len, 8)),
        ("uniform255", offsets(len, 255)),
        ("uniform256", offsets(len, 256)),
        ("uniform257", offsets(len, 257)),
        ("empty50", empty_heavy_offsets(len)),
        ("alternating1_4095", alternating_offsets(len)),
        ("skewed", skewed_offsets(len)),
        ("all", offsets(len, len)),
    ]
}

fn dense_u32(len: usize) -> Vec<u32> {
    (0..len).map(|index| (index % 251) as u32).collect()
}

fn repeated_u32(len: usize, run_len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| (index / run_len.max(1)) as u32)
        .collect()
}

fn reverse_within_segments(len: usize, segment_len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| {
            let begin = index / segment_len * segment_len;
            let end = (begin + segment_len).min(len);
            (end - index - 1) as u32
        })
        .collect()
}

fn bench_map(c: &mut Criterion) {
    let exec = common::exec();
    let mut group = c.benchmark_group("seg_map");
    for &len in common::SIZES {
        group.throughput(Throughput::Elements(len as u64));
        let values = exec.to_device(&dense_u32(len));
        let segment_offsets = exec.to_device(&[0u32, len as u32]);
        exec.sync().unwrap();

        group.bench_function(BenchmarkId::from_parameter(len), |b| {
            b.iter(|| {
                let output = ForEachSegment(Map(AddOne))
                    .run(
                        &exec,
                        SegmentIterator::new(
                            black_box(values.slice(..)),
                            black_box(segment_offsets.slice(..)),
                        ),
                    )
                    .unwrap();
                exec.sync().unwrap();
                black_box(output);
            })
        });
    }
    group.finish();
}

fn bench_inclusive_scan(c: &mut Criterion) {
    let exec = common::exec();
    let mut group = c.benchmark_group("seg_inclusive_scan");
    for &len in common::SIZES {
        group.throughput(Throughput::Elements(len as u64));
        let values = exec.to_device(&dense_u32(len));

        for segment_len in segment_lengths(len) {
            let pattern = segment_name(len, segment_len);
            let segment_offsets = exec.to_device(&offsets(len, segment_len));
            let keys = exec.to_device(&common::run_keys(len, segment_len));
            exec.sync().unwrap();

            group.bench_function(BenchmarkId::new(format!("foreach_{pattern}"), len), |b| {
                b.iter(|| {
                    let output = ForEachSegment(InclusiveScan(Sum))
                        .run(
                            &exec,
                            SegmentIterator::new(
                                black_box(values.slice(..)),
                                black_box(segment_offsets.slice(..)),
                            ),
                        )
                        .unwrap();
                    exec.sync().unwrap();
                    black_box(output);
                })
            });

            group.bench_function(BenchmarkId::new(format!("by_key_{pattern}"), len), |b| {
                b.iter(|| {
                    let output = inclusive_scan_by_key(
                        &exec,
                        black_box(keys.slice(..)),
                        black_box(values.slice(..)),
                        Equal,
                        Sum,
                    )
                    .unwrap();
                    exec.sync().unwrap();
                    black_box(output);
                })
            });
        }
    }
    group.finish();
}

fn bench_reduce(c: &mut Criterion) {
    let exec = common::exec();
    let mut group = c.benchmark_group("seg_reduce");
    for &len in common::SIZES {
        group.throughput(Throughput::Elements(len as u64));
        let values = exec.to_device(&dense_u32(len));

        for segment_len in segment_lengths(len) {
            let pattern = segment_name(len, segment_len);
            let host_offsets = offsets(len, segment_len);
            let segment_count = host_offsets.len() - 1;
            let segment_offsets = exec.to_device(&host_offsets);
            let keys = exec.to_device(&common::run_keys(len, segment_len));
            let init = exec.value(0u32).unwrap();
            let _segment_count = segment_count;
            exec.sync().unwrap();

            group.bench_function(BenchmarkId::new(format!("foreach_{pattern}"), len), |b| {
                b.iter(|| {
                    let output = ForEachSegment(Reduce(Sum, init.clone()))
                        .run(
                            &exec,
                            SegmentIterator::new(
                                black_box(values.slice(..)),
                                black_box(segment_offsets.slice(..)),
                            ),
                        )
                        .unwrap();
                    exec.sync().unwrap();
                    black_box(output);
                })
            });

            group.bench_function(BenchmarkId::new(format!("by_key_{pattern}"), len), |b| {
                b.iter(|| {
                    let output = reduce_by_key(
                        &exec,
                        black_box(keys.slice(..)),
                        black_box(values.slice(..)),
                        Equal,
                        init.clone(),
                        Sum,
                    )
                    .unwrap();
                    exec.sync().unwrap();
                    black_box(output);
                })
            });
        }
    }
    group.finish();
}

fn bench_unique(c: &mut Criterion) {
    let exec = common::exec();
    let mut group = c.benchmark_group("seg_unique");
    for &len in common::SIZES {
        group.throughput(Throughput::Elements(len as u64));
        let values = exec.to_device(&repeated_u32(len, 8));

        for segment_len in segment_lengths(len) {
            let pattern = segment_name(len, segment_len);
            let host_offsets = offsets(len, segment_len);
            let segment_offsets = exec.to_device(&host_offsets);
            exec.sync().unwrap();

            group.bench_function(BenchmarkId::new(pattern, len), |b| {
                b.iter(|| {
                    let output = ForEachSegment(Unique(Equal))
                        .run(
                            &exec,
                            SegmentIterator::new(
                                black_box(values.slice(..)),
                                black_box(segment_offsets.slice(..)),
                            ),
                        )
                        .unwrap();
                    exec.sync().unwrap();
                    black_box(output);
                })
            });
        }
    }
    group.finish();
}

fn bench_reverse(c: &mut Criterion) {
    let exec = common::exec();
    let mut group = c.benchmark_group("seg_reverse");
    for &len in common::SIZES {
        group.throughput(Throughput::Elements(len as u64));
        let values = exec.to_device(&dense_u32(len));

        for segment_len in segment_lengths(len) {
            let pattern = segment_name(len, segment_len);
            let segment_offsets = exec.to_device(&offsets(len, segment_len));
            exec.sync().unwrap();

            group.bench_function(BenchmarkId::new(pattern, len), |b| {
                b.iter(|| {
                    let output = ForEachSegment(Reverse)
                        .run(
                            &exec,
                            SegmentIterator::new(
                                black_box(values.slice(..)),
                                black_box(segment_offsets.slice(..)),
                            ),
                        )
                        .unwrap();
                    exec.sync().unwrap();
                    black_box(output);
                })
            });
        }
    }
    group.finish();
}

fn bench_sort(c: &mut Criterion) {
    let exec = common::exec();
    let mut group = c.benchmark_group("seg_sort");
    for &len in common::SORT_SIZES {
        group.throughput(Throughput::Elements(len as u64));
        let values = exec.to_device(&common::shuffled_u32(len));

        for segment_len in segment_lengths(len) {
            let pattern = segment_name(len, segment_len);
            let segment_offsets = exec.to_device(&offsets(len, segment_len));
            exec.sync().unwrap();

            group.bench_function(BenchmarkId::new(pattern, len), |b| {
                b.iter(|| {
                    let output = ForEachSegment(Sort(Less))
                        .run(
                            &exec,
                            SegmentIterator::new(
                                black_box(values.slice(..)),
                                black_box(segment_offsets.slice(..)),
                            ),
                        )
                        .unwrap();
                    exec.sync().unwrap();
                    black_box(output);
                })
            });
        }
    }
    group.finish();
}

fn bench_unique_patterns(c: &mut Criterion) {
    let exec = common::exec();
    let len = common::SORT_PATTERN_SIZE;
    let patterns = [
        ("all_unique", (0..len as u32).collect::<Vec<_>>()),
        ("run8", repeated_u32(len, 8)),
    ];
    let mut group = c.benchmark_group("seg_unique_patterns");
    group.throughput(Throughput::Elements(len as u64));

    for segment_len in [8, 128, 4_096] {
        let host_offsets = offsets(len, segment_len);
        let segment_offsets = exec.to_device(&host_offsets);

        for (pattern, host_values) in &patterns {
            let values = exec.to_device(host_values);
            exec.sync().unwrap();
            group.bench_function(
                BenchmarkId::new(*pattern, format!("run{segment_len}")),
                |b| {
                    b.iter(|| {
                        let output = ForEachSegment(Unique(Equal))
                            .run(
                                &exec,
                                SegmentIterator::new(
                                    black_box(values.slice(..)),
                                    black_box(segment_offsets.slice(..)),
                                ),
                            )
                            .unwrap();
                        exec.sync().unwrap();
                        black_box(output);
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_sort_patterns(c: &mut Criterion) {
    let exec = common::exec();
    let len = common::SORT_PATTERN_SIZE;
    let mut group = c.benchmark_group("seg_sort_patterns");
    group.throughput(Throughput::Elements(len as u64));

    for segment_len in [8, 128, 4_096] {
        let patterns = [
            (
                "sorted",
                (0..len).map(|index| (index % segment_len) as u32).collect(),
            ),
            ("reverse", reverse_within_segments(len, segment_len)),
            ("shuffled", common::shuffled_u32(len)),
        ];
        let segment_offsets = exec.to_device(&offsets(len, segment_len));

        for (pattern, host_values) in patterns {
            let values = exec.to_device(&host_values);
            exec.sync().unwrap();
            group.bench_function(
                BenchmarkId::new(pattern, format!("run{segment_len}")),
                |b| {
                    b.iter(|| {
                        let output = ForEachSegment(Sort(Less))
                            .run(
                                &exec,
                                SegmentIterator::new(
                                    black_box(values.slice(..)),
                                    black_box(segment_offsets.slice(..)),
                                ),
                            )
                            .unwrap();
                        exec.sync().unwrap();
                        black_box(output);
                    })
                },
            );
        }
    }
    group.finish();
}

fn bench_control_geometries(c: &mut Criterion) {
    const LEN: usize = 1_024 * 1_024;

    let exec = common::exec();
    let values = exec.to_device(&dense_u32(LEN));
    exec.sync().unwrap();

    let mut group = c.benchmark_group("seg_control_geometries");
    group.throughput(Throughput::Elements(LEN as u64));

    for (geometry, host_offsets) in control_geometries(LEN) {
        let segment_offsets = exec.to_device(&host_offsets);
        let init = exec.value(0u32).unwrap();
        exec.sync().unwrap();

        group.bench_function(BenchmarkId::new("inclusive_scan", geometry), |b| {
            b.iter(|| {
                let output = ForEachSegment(InclusiveScan(Sum))
                    .run(
                        &exec,
                        SegmentIterator::new(
                            black_box(values.slice(..)),
                            black_box(segment_offsets.slice(..)),
                        ),
                    )
                    .unwrap();
                exec.sync().unwrap();
                black_box(output);
            })
        });

        group.bench_function(BenchmarkId::new("reduce", geometry), |b| {
            b.iter(|| {
                let output = ForEachSegment(Reduce(Sum, init.clone()))
                    .run(
                        &exec,
                        SegmentIterator::new(
                            black_box(values.slice(..)),
                            black_box(segment_offsets.slice(..)),
                        ),
                    )
                    .unwrap();
                exec.sync().unwrap();
                black_box(output);
            })
        });

        group.bench_function(BenchmarkId::new("adjacent_difference", geometry), |b| {
            b.iter(|| {
                let output = ForEachSegment(AdjacentDifference(Sum))
                    .run(
                        &exec,
                        SegmentIterator::new(
                            black_box(values.slice(..)),
                            black_box(segment_offsets.slice(..)),
                        ),
                    )
                    .unwrap();
                exec.sync().unwrap();
                black_box(output);
            })
        });

        group.bench_function(BenchmarkId::new("filter50", geometry), |b| {
            b.iter(|| {
                let output = ForEachSegment(Filter(Even))
                    .run(
                        &exec,
                        SegmentIterator::new(
                            black_box(values.slice(..)),
                            black_box(segment_offsets.slice(..)),
                        ),
                    )
                    .unwrap();
                exec.sync().unwrap();
                black_box(output);
            })
        });

        group.bench_function(BenchmarkId::new("is_sorted", geometry), |b| {
            b.iter(|| {
                let output = ForEachSegment(IsSorted(Less))
                    .run(
                        &exec,
                        SegmentIterator::new(
                            black_box(values.slice(..)),
                            black_box(segment_offsets.slice(..)),
                        ),
                    )
                    .unwrap();
                exec.sync().unwrap();
                black_box(output);
            })
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets =
        bench_map,
        bench_inclusive_scan,
        bench_reduce,
        bench_unique,
        bench_reverse,
        bench_sort,
        bench_unique_patterns,
        bench_sort_patterns,
        bench_control_geometries
}
criterion_main!(benches);
