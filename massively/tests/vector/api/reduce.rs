use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, lazy, op::ReductionOp, op::UnaryOp, vector::reduce, zip2};

struct DetectHit;

#[cubecl::cube]
impl UnaryOp<(f32, f32)> for DetectHit {
    type Output = u32;

    fn apply(input: (f32, f32)) -> u32 {
        let d2 = input.0 * input.0 + input.1 * input.1;
        if d2 <= 1.0 { 1_u32 } else { 0_u32 }
    }
}

struct CountHit;

#[cubecl::cube]
impl ReductionOp<u32> for CountHit {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

#[test]
fn reduce_estimates_pi_from_lazy_random_map() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let samples = 100_000_usize;
    let x = massively::util::random::uniform_f32(0.0, 1.0, 0)
        .unwrap()
        .take(samples as massively::MIndex);
    let y = massively::util::random::uniform_f32(0.0, 1.0, 1)
        .unwrap()
        .take(samples as massively::MIndex);
    let hits = lazy::map(zip2(x, y), DetectHit);

    let count = reduce(&exec, hits, exec.value(0_u32).unwrap(), CountHit)
        .unwrap()
        .read(&exec)
        .unwrap();
    let pi = (count as f64 / samples as f64) * 4.0;

    assert!((3.0..3.3).contains(&pi), "pi={pi}, count={count}");
}

#[test]
#[ignore = "4G-scale regression test; run explicitly on a GPU-capable machine"]
fn reduce_estimates_pi_from_lazy_random_map_4g() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let samples = 4_000_000_000_usize;
    let x = massively::util::random::uniform_f32(0.0, 1.0, 0)
        .unwrap()
        .take(samples as massively::MIndex);
    let y = massively::util::random::uniform_f32(0.0, 1.0, 1)
        .unwrap()
        .take(samples as massively::MIndex);
    let hits = lazy::map(zip2(x, y), DetectHit);

    let count = reduce(&exec, hits, exec.value(0_u32).unwrap(), CountHit)
        .unwrap()
        .read(&exec)
        .unwrap();
    let pi = (count as f64 / samples as f64) * 4.0;

    assert!((3.10..3.18).contains(&pi), "pi={pi}, count={count}");
}

#[test]
#[ignore = "4G-scale regression test; run explicitly on a GPU-capable machine"]
fn reduce_counts_four_billion_lazy_constants() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let len = 4_000_000_000_usize;

    let count = reduce(
        &exec,
        lazy::constant(1_u32).take(len as massively::MIndex),
        exec.value(0_u32).unwrap(),
        CountHit,
    )
    .unwrap()
    .read(&exec)
    .unwrap();

    assert_eq!(count, len as u32);
}

#[test]
fn reduce_result_can_feed_the_next_reduce_without_a_host_read() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let first = exec.to_device(&[1_u32, 2, 3]);
    let second = exec.to_device(&[4_u32, 5]);

    let subtotal = reduce(&exec, first.slice(..), exec.value(0_u32).unwrap(), CountHit).unwrap();
    let total = reduce(&exec, second.slice(..), subtotal, CountHit).unwrap();

    assert_eq!(total.read(&exec).unwrap(), 15);
}
