mod common;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use massively::{op::BinaryPredicateOp, vector::lower_bound, vector::upper_bound};

struct Less;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

fn queries(len: usize) -> Vec<u32> {
    (0..len)
        .map(|index| ((index * 1_103_515_245 + 12_345) % (len.max(1) * 2)) as u32)
        .collect()
}

fn bench_search(c: &mut Criterion) {
    let exec = common::exec();
    let mut lower = c.benchmark_group("lower_bound");
    for &len in common::SIZES {
        let source = exec.to_device(&(0..len).map(|index| index as u32).collect::<Vec<_>>());
        let values = exec.to_device(&queries(len));
        lower.bench_function(BenchmarkId::new("gpu", len), |b| {
            b.iter(|| {
                let output = lower_bound(
                    &exec,
                    black_box(source.slice(..)),
                    black_box(values.slice(..)),
                    Less,
                )
                .unwrap();
                exec.sync().unwrap();
                black_box(output);
            })
        });
    }
    lower.finish();

    let mut upper = c.benchmark_group("upper_bound");
    for &len in common::SIZES {
        let source = exec.to_device(&(0..len).map(|index| index as u32).collect::<Vec<_>>());
        let values = exec.to_device(&queries(len));
        upper.bench_function(BenchmarkId::new("gpu", len), |b| {
            b.iter(|| {
                let output = upper_bound(
                    &exec,
                    black_box(source.slice(..)),
                    black_box(values.slice(..)),
                    Less,
                )
                .unwrap();
                exec.sync().unwrap();
                black_box(output);
            })
        });
    }
    upper.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion();
    targets = bench_search
}
criterion_main!(benches);
