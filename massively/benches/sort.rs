mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::{
    op::BinaryPredicateOp,
    vector::{radix_sort_by_key, sort_by_key},
};

struct Less;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

fn sort_patterns(len: usize) -> [(&'static str, Vec<u32>); 5] {
    [
        ("shuffled", common::shuffled_u32(len)),
        ("sorted", (0..len as u32).collect()),
        ("reverse", common::reverse_u32(len)),
        ("equal", vec![7_u32; len]),
        (
            "low_cardinality",
            (0..len).map(|index| (index % 32) as u32).collect(),
        ),
    ]
}

fn bench_sort_by_key(c: &mut Criterion) {
    let exec = common::exec();
    let mut group = c.benchmark_group("sort_by_key");

    for &len in common::SORT_SIZES {
        let keys = exec.to_device(&common::shuffled_u32(len));
        let values = exec.to_device(&common::dense_f32(len));
        exec.sync().unwrap();

        group.bench_function(BenchmarkId::new("gpu", len), |b| {
            b.iter(|| {
                let output = sort_by_key(
                    &exec,
                    black_box(keys.slice(..)),
                    black_box(values.slice(..)),
                    Less,
                )
                .unwrap();
                exec.sync().unwrap();
                black_box(output);
            })
        });
    }
    group.finish();

    let len = common::SORT_PATTERN_SIZE;
    let mut pattern_group = c.benchmark_group("sort_by_key_patterns");
    for (name, input) in sort_patterns(len) {
        let keys = exec.to_device(&input);
        let values = exec.to_device(&common::dense_f32(len));
        pattern_group.bench_function(name, |b| {
            b.iter(|| {
                let output = sort_by_key(
                    &exec,
                    black_box(keys.slice(..)),
                    black_box(values.slice(..)),
                    Less,
                )
                .unwrap();
                exec.sync().unwrap();
                black_box(output);
            })
        });
    }
    pattern_group.finish();
}

fn bench_radix_sort_by_key(c: &mut Criterion) {
    let exec = common::exec();
    let mut group = c.benchmark_group("radix_sort_by_key");

    for &len in common::SORT_SIZES {
        let keys = exec.to_device(&common::shuffled_u32(len));
        let values = exec.to_device(&common::dense_f32(len));
        exec.sync().unwrap();

        group.bench_function(BenchmarkId::new("gpu", len), |b| {
            b.iter(|| {
                let output = radix_sort_by_key(
                    &exec,
                    black_box(keys.slice(..)),
                    black_box(values.slice(..)),
                )
                .unwrap();
                exec.sync().unwrap();
                black_box(output);
            })
        });
    }
    group.finish();

    let len = common::SORT_PATTERN_SIZE;
    let mut pattern_group = c.benchmark_group("radix_sort_by_key_patterns");
    for (name, input) in sort_patterns(len) {
        let keys = exec.to_device(&input);
        let values = exec.to_device(&common::dense_f32(len));
        pattern_group.bench_function(name, |b| {
            b.iter(|| {
                let output = radix_sort_by_key(
                    &exec,
                    black_box(keys.slice(..)),
                    black_box(values.slice(..)),
                )
                .unwrap();
                exec.sync().unwrap();
                black_box(output);
            })
        });
    }
    pattern_group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = bench_sort_by_key, bench_radix_sort_by_key
}
criterion_main!(benches);
