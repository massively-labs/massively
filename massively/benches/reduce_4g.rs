use std::time::Duration;

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{
    Executor, lazy, op::ReductionOp, op::UnaryOp, vector::reduce, zip2, zip3, zip4, zip5,
};

const LEN: massively::MIndex = 4_000_000_000;
const X_KEY: u32 = seed_key(0);
const Y_KEY: u32 = seed_key(1);

type FourArgs = (u32, u32, u32, u32);
type RandomArgs = (u32, f32, f32, u32);
type RuntimeKeyArgs = (u32, u32, u32);
type RuntimeRangeArgs = (u32, u32, u32, f32, f32);

struct Sum;
struct DetectHit;
struct FourMix;
struct PairXor;
struct PcgPairHash;
struct PcgPairUnitCompare;
struct PcgPairCircle;
struct PcgPairCircleRuntime;
struct PcgPairCircleRuntimeRange;
struct UniformFromArgs;

const fn pcg_hash32_host(input: u32) -> u32 {
    let state = input.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

const fn seed_key(seed: u64) -> u32 {
    let low = seed as u32;
    let high = (seed >> 32) as u32;
    pcg_hash32_host(low ^ pcg_hash32_host(high))
}

#[cubecl::cube]
fn pcg_hash32(input: u32) -> u32 {
    let state = input * 747_796_405u32 + 2_891_336_453u32;
    let word = ((state >> ((state >> 28u32) + 4u32)) ^ state) * 277_803_737u32;
    (word >> 22u32) ^ word
}

#[cubecl::cube]
fn random_u32_at(key: u32, index: u32) -> u32 {
    pcg_hash32(index ^ key)
}

#[cubecl::cube]
fn unit_f32(key: u32, index: u32) -> f32 {
    ((random_u32_at(key, index) >> 8u32) as f32) * 0.00000005960464832810486063f32
}

#[cubecl::cube]
fn uniform_f32(min: f32, max: f32, key: u32, index: u32) -> f32 {
    min + unit_f32(key, index) * (max - min)
}

#[cubecl::cube]
fn circle_hit(x: f32, y: f32) -> u32 {
    let d2 = x * x + y * y;
    if d2 <= 1.0 { 1u32 } else { 0u32 }
}

#[cubecl::cube]
impl ReductionOp<u32> for Sum {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

#[cubecl::cube]
impl UnaryOp<(f32, f32)> for DetectHit {
    type Output = u32;

    fn apply(input: (f32, f32)) -> u32 {
        let d2 = input.0 * input.0 + input.1 * input.1;
        if d2 <= 1.0 { 1_u32 } else { 0_u32 }
    }
}

#[cubecl::cube]
impl UnaryOp<FourArgs> for FourMix {
    type Output = u32;

    fn apply(input: FourArgs) -> u32 {
        input.0 as u32 ^ input.1 ^ input.2 ^ input.3
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, u32)> for PairXor {
    type Output = u32;

    fn apply(input: (u32, u32)) -> u32 {
        input.0 ^ input.1
    }
}

#[cubecl::cube]
impl UnaryOp<u32> for PcgPairHash {
    type Output = u32;

    fn apply(index: u32) -> u32 {
        random_u32_at(X_KEY, index) ^ random_u32_at(Y_KEY, index)
    }
}

#[cubecl::cube]
impl UnaryOp<u32> for PcgPairUnitCompare {
    type Output = u32;

    fn apply(index: u32) -> u32 {
        let x = unit_f32(X_KEY, index);
        let y = unit_f32(Y_KEY, index);
        if x <= y { 1u32 } else { 0u32 }
    }
}

#[cubecl::cube]
impl UnaryOp<u32> for PcgPairCircle {
    type Output = u32;

    fn apply(index: u32) -> u32 {
        circle_hit(unit_f32(X_KEY, index), unit_f32(Y_KEY, index))
    }
}

#[cubecl::cube]
impl UnaryOp<RuntimeKeyArgs> for PcgPairCircleRuntime {
    type Output = u32;

    fn apply(input: RuntimeKeyArgs) -> u32 {
        let index = input.0 as u32;
        circle_hit(unit_f32(input.1, index), unit_f32(input.2, index))
    }
}

#[cubecl::cube]
impl UnaryOp<RuntimeRangeArgs> for PcgPairCircleRuntimeRange {
    type Output = u32;

    fn apply(input: RuntimeRangeArgs) -> u32 {
        let index = input.0 as u32;
        let x_key = input.1;
        let y_key = input.2;
        let min = input.3;
        let max = input.4;
        circle_hit(
            uniform_f32(min, max, x_key, index),
            uniform_f32(min, max, y_key, index),
        )
    }
}

#[cubecl::cube]
impl UnaryOp<RandomArgs> for UniformFromArgs {
    type Output = f32;

    fn apply(input: RandomArgs) -> f32 {
        uniform_f32(input.1, input.2, input.3, input.0 as u32)
    }
}

fn assert_pi(count: u32) {
    let pi = count as f64 * 4.0 / LEN as f64;
    assert!((3.10..3.18).contains(&black_box(pi)), "pi={pi}");
}

fn bench_reduce_4g(c: &mut Criterion) {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let init = 0_u32;

    let mut group = c.benchmark_group("reduce_4g");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(6));
    group.throughput(Throughput::Elements(LEN as u64));
    group.bench_function("constant_u32", |b| {
        b.iter(|| {
            let input = lazy::constant(black_box(1_u32)).take(LEN);
            let result = reduce(&exec, input, init.clone(), Sum).unwrap();
            assert_eq!(black_box(result), LEN as u32);
        })
    });
    group.bench_function("diag_a8_parameter_mix", |b| {
        b.iter(|| {
            let left = zip4(
                lazy::counting(black_box(0)).take(LEN),
                lazy::constant(black_box(0x1357_9bdf_u32)).take(LEN),
                lazy::constant(black_box(0x2468_ace0_u32)).take(LEN),
                lazy::constant(black_box(0xfdb9_7531_u32)).take(LEN),
            );
            let right = zip4(
                lazy::counting(black_box(1)).take(LEN),
                lazy::constant(black_box(0x0f0f_0f0f_u32)).take(LEN),
                lazy::constant(black_box(0xf0f0_f0f0_u32)).take(LEN),
                lazy::constant(black_box(0x55aa_aa55_u32)).take(LEN),
            );
            let left = lazy::map(left, FourMix);
            let right = lazy::map(right, FourMix);
            let result = reduce(
                &exec,
                lazy::map(zip2(left, right), PairXor),
                init.clone(),
                Sum,
            )
            .unwrap();
            black_box(result);
        })
    });
    group.bench_function("diag_a1_pcg2_hash", |b| {
        b.iter(|| {
            let values = lazy::map(lazy::counting(black_box(0)).take(LEN), PcgPairHash);
            black_box(reduce(&exec, values, init.clone(), Sum).unwrap());
        })
    });
    group.bench_function("diag_a1_pcg2_unit_compare", |b| {
        b.iter(|| {
            let values = lazy::map(lazy::counting(black_box(0)).take(LEN), PcgPairUnitCompare);
            let count = reduce(&exec, values, init.clone(), Sum).unwrap();
            assert!((1_900_000_000..2_100_000_000).contains(&black_box(count)));
        })
    });
    group.bench_function("diag_a1_pcg2_circle", |b| {
        b.iter(|| {
            let values = lazy::map(lazy::counting(black_box(0)).take(LEN), PcgPairCircle);
            assert_pi(reduce(&exec, values, init.clone(), Sum).unwrap());
        })
    });
    group.bench_function("diag_a3_pcg2_circle_runtime_keys", |b| {
        b.iter(|| {
            let input = zip3(
                lazy::counting(black_box(0)).take(LEN),
                lazy::constant(black_box(X_KEY)).take(LEN),
                lazy::constant(black_box(Y_KEY)).take(LEN),
            );
            let values = lazy::map(input, PcgPairCircleRuntime);
            assert_pi(reduce(&exec, values, init.clone(), Sum).unwrap());
        })
    });
    group.bench_function("diag_a5_pcg2_circle_runtime_range", |b| {
        b.iter(|| {
            let input = zip5(
                lazy::counting(black_box(0)).take(LEN),
                lazy::constant(black_box(X_KEY)).take(LEN),
                lazy::constant(black_box(Y_KEY)).take(LEN),
                lazy::constant(black_box(0.0f32)).take(LEN),
                lazy::constant(black_box(1.0f32)).take(LEN),
            );
            let values = lazy::map(input, PcgPairCircleRuntimeRange);
            assert_pi(reduce(&exec, values, init.clone(), Sum).unwrap());
        })
    });
    group.bench_function("diag_a8_pcg2_circle_direct", |b| {
        b.iter(|| {
            let x = zip4(
                lazy::counting(black_box(0)).take(LEN),
                lazy::constant(black_box(0.0f32)).take(LEN),
                lazy::constant(black_box(1.0f32)).take(LEN),
                lazy::constant(black_box(X_KEY)).take(LEN),
            );
            let y = zip4(
                lazy::counting(black_box(0)).take(LEN),
                lazy::constant(black_box(0.0f32)).take(LEN),
                lazy::constant(black_box(1.0f32)).take(LEN),
                lazy::constant(black_box(Y_KEY)).take(LEN),
            );
            let x = lazy::map(x, UniformFromArgs);
            let y = lazy::map(y, UniformFromArgs);
            let values = lazy::map(zip2(x, y), DetectHit);
            assert_pi(reduce(&exec, values, init.clone(), Sum).unwrap());
        })
    });
    group.bench_function("monte_carlo_pi", |b| {
        b.iter(|| {
            let x = massively::util::random::uniform_f32(0.0, 1.0, black_box(0))
                .unwrap()
                .take(LEN);
            let y = massively::util::random::uniform_f32(0.0, 1.0, black_box(1))
                .unwrap()
                .take(LEN);
            let hits = lazy::map(zip2(x, y), DetectHit);
            let count = reduce(&exec, hits, init.clone(), Sum).unwrap();
            assert_pi(count);
        })
    });
    group.finish();
}

criterion_group!(benches, bench_reduce_4g);
criterion_main!(benches);
