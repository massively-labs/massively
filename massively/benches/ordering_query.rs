use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{
    Executor, lazy, op::BinaryPredicateOp, vector::max_element, vector::min_element,
    vector::minmax_element, zip7,
};

const SIZES: &[usize] = &[1 << 20, 1 << 24];

type Seven = (u32, u32, u32, u32, u32, u32, u32);

struct LessU32;
struct FirstLeafLess;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<Seven> for FirstLeafLess {
    fn apply(lhs: Seven, rhs: Seven) -> massively::MFlag {
        massively::flag::from_bool(lhs.0 < rhs.0)
    }
}

fn bench_ordering_query(c: &mut Criterion) {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let mut group = c.benchmark_group("ordering_query");

    for &len in SIZES {
        group.bench_function(BenchmarkId::new("min_element_lazy", len), |b| {
            b.iter(|| {
                black_box(min_element(&exec, lazy::counting(0).take(len as u32), LessU32).unwrap())
            })
        });
        group.bench_function(BenchmarkId::new("max_element_lazy", len), |b| {
            b.iter(|| {
                black_box(max_element(&exec, lazy::counting(0).take(len as u32), LessU32).unwrap())
            })
        });
        group.bench_function(BenchmarkId::new("minmax_element_lazy", len), |b| {
            b.iter(|| {
                black_box(
                    minmax_element(&exec, lazy::counting(0).take(len as u32), LessU32).unwrap(),
                )
            })
        });

        let streams = [
            lazy::counting(0).take(len as u32),
            lazy::counting(1).take(len as u32),
            lazy::counting(2).take(len as u32),
            lazy::counting(3).take(len as u32),
            lazy::counting(4).take(len as u32),
            lazy::counting(5).take(len as u32),
            lazy::counting(6).take(len as u32),
        ];
        group.bench_function(BenchmarkId::new("min_element_lazy_zip7", len), |b| {
            b.iter(|| {
                black_box(
                    min_element(
                        &exec,
                        zip7(
                            streams[0], streams[1], streams[2], streams[3], streams[4], streams[5],
                            streams[6],
                        ),
                        FirstLeafLess,
                    )
                    .unwrap(),
                )
            })
        });
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(500));
    targets = bench_ordering_query
}
criterion_main!(benches);
