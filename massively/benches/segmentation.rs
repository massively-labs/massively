mod common;

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::{
    lazy,
    op::{ExpandOp, UnaryOp},
    seg::{Executable, FlatMap, ForEachSegment, SegmentIterator, Segmentation},
    vector::map,
    zip2,
};

struct Fanout012;
struct BalancedFanout02;
struct AddContext;

#[cubecl::cube]
impl ExpandOp<u32> for Fanout012 {
    type Output = u32;

    fn count(input: u32) -> u32 {
        input % 3u32
    }

    fn generate(input: u32, local_index: u32) -> u32 {
        input * 2u32 + local_index
    }
}

#[cubecl::cube]
impl ExpandOp<u32> for BalancedFanout02 {
    type Output = u32;

    fn count(input: u32) -> u32 {
        (input % 2u32) * 2u32
    }

    fn generate(_input: u32, local_index: u32) -> u32 {
        local_index
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for AddContext {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        input.0 + input.1
    }
}

fn uniform_offsets(len: usize, segment_len: usize) -> Vec<u32> {
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

fn skewed_offsets(len: usize) -> Vec<u32> {
    const LENGTHS: &[usize] = &[1, 2, 4, 8, 16, 64, 4_096];

    let mut result = vec![0];
    let mut offset = 0;
    let mut index = 0;
    while offset < len {
        offset = (offset + LENGTHS[index % LENGTHS.len()]).min(len);
        result.push(offset as u32);
        index += 1;
    }
    result
}

fn conversion_geometries(len: usize) -> Vec<(&'static str, Vec<u32>)> {
    vec![
        ("uniform4", uniform_offsets(len, 4)),
        ("uniform8", uniform_offsets(len, 8)),
        ("uniform16", uniform_offsets(len, 16)),
        ("empty50", empty_heavy_offsets(len)),
        ("skewed", skewed_offsets(len)),
        ("all", vec![0, len as u32]),
    ]
}

fn lengths_from_offsets(offsets: &[u32]) -> Vec<u32> {
    offsets
        .windows(2)
        .map(|bounds| bounds[1] - bounds[0])
        .collect()
}

fn ids_from_offsets(offsets: &[u32]) -> Vec<u32> {
    let mut ids = Vec::with_capacity(*offsets.last().unwrap_or(&0) as usize);
    for (segment, bounds) in offsets.windows(2).enumerate() {
        ids.extend(std::iter::repeat_n(
            segment as u32,
            (bounds[1] - bounds[0]) as usize,
        ));
    }
    ids
}

fn dense_u32(len: usize) -> Vec<u32> {
    (0..len).map(|index| (index % 251) as u32).collect()
}

fn balanced_fanout_u32(len: usize) -> Vec<u32> {
    (0..len).map(|index| (index % 2) as u32).collect()
}

fn bench_conversions(c: &mut Criterion) {
    const THROUGHPUT_LEN: usize = 1_024 * 1_024;
    const SMALL_LATENCY_LEN: usize = 1_024;
    const LATENCY_LEN: usize = 4 * 1_024;

    let exec = common::exec();
    let mut group = c.benchmark_group("segmentation_conversions");

    for (len, geometries) in [
        (
            SMALL_LATENCY_LEN,
            vec![("uniform8", uniform_offsets(SMALL_LATENCY_LEN, 8))],
        ),
        (
            LATENCY_LEN,
            vec![("uniform8", uniform_offsets(LATENCY_LEN, 8))],
        ),
        (THROUGHPUT_LEN, conversion_geometries(THROUGHPUT_LEN)),
    ] {
        for (geometry, host_offsets) in geometries {
            let host_lengths = lengths_from_offsets(&host_offsets);
            let host_ids = ids_from_offsets(&host_offsets);
            let offsets = exec.to_device(&host_offsets);
            let lengths = exec.to_device(&host_lengths);
            let ids = exec.to_device(&host_ids);
            let segmentation = Segmentation::from_offsets(&exec, offsets.slice(..)).unwrap();
            let case = format!("{len}/{geometry}");

            group.throughput(Throughput::Elements(host_offsets.len() as u64));
            group.bench_function(BenchmarkId::new("from_offsets", &case), |b| {
                b.iter(|| {
                    let output =
                        Segmentation::from_offsets(&exec, black_box(offsets.slice(..))).unwrap();
                    black_box(output);
                })
            });

            group.throughput(Throughput::Elements(host_lengths.len() as u64));
            group.bench_function(BenchmarkId::new("from_lengths", &case), |b| {
                b.iter(|| {
                    let output =
                        Segmentation::from_lengths(&exec, black_box(lengths.slice(..))).unwrap();
                    black_box(output);
                })
            });

            group.throughput(Throughput::Elements(host_ids.len() as u64));
            group.bench_function(BenchmarkId::new("from_ids", &case), |b| {
                b.iter(|| {
                    let output = Segmentation::from_segment_ids(
                        &exec,
                        black_box(ids.slice(..)),
                        host_lengths.len() as u32,
                    )
                    .unwrap();
                    exec.sync().unwrap();
                    black_box(output);
                })
            });

            group.throughput(Throughput::Elements(host_lengths.len() as u64));
            group.bench_function(BenchmarkId::new("to_lengths", &case), |b| {
                b.iter(|| {
                    let output = segmentation.lengths(&exec).unwrap();
                    exec.sync().unwrap();
                    black_box(output);
                })
            });

            group.throughput(Throughput::Elements(host_ids.len() as u64));
            group.bench_function(BenchmarkId::new("to_ids", &case), |b| {
                b.iter(|| {
                    let output = segmentation.segment_ids(&exec).unwrap();
                    exec.sync().unwrap();
                    black_box(output);
                })
            });

            group.throughput(Throughput::Elements(host_ids.len() as u64));
            group.bench_function(BenchmarkId::new("lengths_to_ids", &case), |b| {
                b.iter(|| {
                    let segmentation =
                        Segmentation::from_lengths(&exec, black_box(lengths.slice(..))).unwrap();
                    let output = segmentation.segment_ids(&exec).unwrap();
                    exec.sync().unwrap();
                    black_box(output);
                })
            });
        }
    }
    group.finish();
}

fn bench_round_transition(c: &mut Criterion) {
    let exec = common::exec();
    let mut group = c.benchmark_group("segmentation_round_transition");

    for len in [1_024, 4 * 1_024, 1_024 * 1_024] {
        group.throughput(Throughput::Elements(len as u64));
        let values = exec.to_device(&dense_u32(len));

        for (geometry, host_offsets) in [
            ("uniform8", uniform_offsets(len, 8)),
            ("empty50", empty_heavy_offsets(len)),
        ] {
            let offsets = exec.to_device(&host_offsets);
            let contexts = exec.to_device(
                &(0..host_offsets.len() - 1)
                    .map(|segment| segment as u32)
                    .collect::<Vec<_>>(),
            );
            exec.sync().unwrap();
            let case = format!("{len}/{geometry}");

            group.bench_function(BenchmarkId::new("flat_map", &case), |b| {
                b.iter(|| {
                    let output = ForEachSegment(FlatMap(Fanout012))
                        .run(
                            &exec,
                            SegmentIterator::new(
                                black_box(values.slice(..)),
                                black_box(offsets.slice(..)),
                            ),
                        )
                        .unwrap();
                    exec.sync().unwrap();
                    black_box(output);
                })
            });

            group.bench_function(BenchmarkId::new("plus_ids", &case), |b| {
                b.iter(|| {
                    let output = ForEachSegment(FlatMap(Fanout012))
                        .run(
                            &exec,
                            SegmentIterator::new(
                                black_box(values.slice(..)),
                                black_box(offsets.slice(..)),
                            ),
                        )
                        .unwrap();
                    let (output_values, segmentation) = output.into_parts();
                    let ids = segmentation.segment_ids(&exec).unwrap();
                    exec.sync().unwrap();
                    black_box((output_values, segmentation, ids));
                })
            });

            group.bench_function(BenchmarkId::new("plus_context_consumer", &case), |b| {
                b.iter(|| {
                    let output = ForEachSegment(FlatMap(Fanout012))
                        .run(
                            &exec,
                            SegmentIterator::new(
                                black_box(values.slice(..)),
                                black_box(offsets.slice(..)),
                            ),
                        )
                        .unwrap();
                    let (output_values, segmentation) = output.into_parts();
                    let ids = segmentation.segment_ids(&exec).unwrap();
                    let consumed = map(
                        &exec,
                        zip2(
                            output_values.slice(..),
                            lazy::permute(contexts.slice(..), ids.slice(..)),
                        ),
                        AddContext,
                    )
                    .unwrap();
                    exec.sync().unwrap();
                    black_box((segmentation, consumed));
                })
            });
        }
    }
    group.finish();
}

fn bench_repeated_rounds(c: &mut Criterion) {
    const SEGMENT_COUNT: usize = 256;

    let exec = common::exec();
    let mut group = c.benchmark_group("segmentation_repeated_rounds");
    group
        .sample_size(20)
        .measurement_time(Duration::from_secs(1));

    for (geometry, value_len, host_offsets) in [
        (
            "all_empty",
            0,
            std::iter::repeat_n(0, SEGMENT_COUNT + 1).collect(),
        ),
        (
            "uniform4",
            SEGMENT_COUNT * 4,
            uniform_offsets(SEGMENT_COUNT * 4, 4),
        ),
        (
            "uniform8",
            SEGMENT_COUNT * 8,
            uniform_offsets(SEGMENT_COUNT * 8, 8),
        ),
        (
            "uniform16",
            SEGMENT_COUNT * 16,
            uniform_offsets(SEGMENT_COUNT * 16, 16),
        ),
    ] {
        let values = exec.to_device(&balanced_fanout_u32(value_len));
        let offsets = exec.to_device(&host_offsets);
        let initial_segmentation = Segmentation::from_offsets(&exec, offsets.slice(..)).unwrap();

        for rounds in [2, 8] {
            let case = format!("{rounds}r/{geometry}");

            // Library-produced offsets stay wrapped in Segmentation across
            // rounds and require no validation at the round boundary.
            group.bench_function(BenchmarkId::new("trusted_segmentation", &case), |b| {
                b.iter(|| {
                    let mut round_values = values.clone();
                    let mut segmentation = initial_segmentation.clone();
                    for _ in 0..rounds {
                        let output = ForEachSegment(FlatMap(BalancedFanout02))
                            .run(
                                &exec,
                                segmentation.segments(round_values.slice(..)).unwrap(),
                            )
                            .unwrap();
                        (round_values, segmentation) = output.into_parts();
                    }
                    exec.sync().unwrap();
                    black_box((round_values, segmentation));
                })
            });

            // Measures the validation that callers previously had to repeat
            // before using each generated result as a Segmentation.
            group.bench_function(BenchmarkId::new("revalidated_segmentation", &case), |b| {
                b.iter(|| {
                    let mut round_values = values.clone();
                    let mut segmentation = initial_segmentation.clone();
                    for _ in 0..rounds {
                        let output = ForEachSegment(FlatMap(BalancedFanout02))
                            .run(
                                &exec,
                                segmentation.segments(round_values.slice(..)).unwrap(),
                            )
                            .unwrap();
                        let (output_values, generated) = output.into_parts();
                        segmentation =
                            Segmentation::from_offsets(&exec, generated.offsets()).unwrap();
                        round_values = output_values;
                    }
                    exec.sync().unwrap();
                    black_box((round_values, segmentation));
                })
            });
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = bench_conversions, bench_round_transition, bench_repeated_rounds
}
criterion_main!(benches);
