use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{
    Executor, lazy, op::PredicateOp, vector::all_of, vector::any_of, vector::count_if,
    vector::find_if, vector::is_partitioned, vector::none_of, zip7,
};

const SIZES: &[usize] = &[1 << 20, 1 << 24];

struct Even;
struct FirstLeafEven;

type Seven = (u32, u32, u32, u32, u32, u32, u32);

#[cubecl::cube]
impl PredicateOp<u32> for Even {
    fn apply(value: u32) -> massively::MFlag {
        massively::flag::from_bool(value % 2u32 == 0u32)
    }
}

#[cubecl::cube]
impl PredicateOp<Seven> for FirstLeafEven {
    fn apply(value: Seven) -> massively::MFlag {
        massively::flag::from_bool(value.0 % 2u32 == 0u32)
    }
}

fn bench_predicate_query(c: &mut Criterion) {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let mut group = c.benchmark_group("predicate_query");

    for &len in SIZES {
        group.bench_function(BenchmarkId::new("count_if_lazy", len), |b| {
            b.iter(|| black_box(count_if(&exec, lazy::counting(0).take(len as u32), Even).unwrap()))
        });
        group.bench_function(BenchmarkId::new("all_of_lazy", len), |b| {
            b.iter(|| black_box(all_of(&exec, lazy::counting(0).take(len as u32), Even).unwrap()))
        });
        group.bench_function(BenchmarkId::new("any_of_lazy", len), |b| {
            b.iter(|| black_box(any_of(&exec, lazy::counting(0).take(len as u32), Even).unwrap()))
        });
        group.bench_function(BenchmarkId::new("none_of_lazy", len), |b| {
            b.iter(|| black_box(none_of(&exec, lazy::counting(0).take(len as u32), Even).unwrap()))
        });
        group.bench_function(BenchmarkId::new("find_if_lazy", len), |b| {
            b.iter(|| black_box(find_if(&exec, lazy::counting(0).take(len as u32), Even).unwrap()))
        });
        group.bench_function(BenchmarkId::new("is_partitioned_lazy", len), |b| {
            b.iter(|| {
                black_box(is_partitioned(&exec, lazy::counting(0).take(len as u32), Even).unwrap())
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
        group.bench_function(BenchmarkId::new("count_if_lazy_zip7", len), |b| {
            b.iter(|| {
                black_box(
                    count_if(
                        &exec,
                        zip7(
                            streams[0], streams[1], streams[2], streams[3], streams[4], streams[5],
                            streams[6],
                        ),
                        FirstLeafEven,
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
    targets = bench_predicate_query
}
criterion_main!(benches);
