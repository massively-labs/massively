use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{Executor, MStorage, op::ExpandOp, vector::flat_map, zip2};

struct RepeatValue;

#[cubecl::cube]
impl ExpandOp<u32> for RepeatValue {
    type Output = u32;

    fn count(input: u32) -> u32 {
        input
    }

    fn generate(input: u32, local_index: u32) -> u32 {
        input * 10 + local_index
    }
}

struct ExpandPair;

#[cubecl::cube]
impl ExpandOp<(u32, u32)> for ExpandPair {
    type Output = (u32, u64);

    fn count(input: (u32, u32)) -> u32 {
        input.0
    }

    fn generate(input: (u32, u32), local_index: u32) -> Self::Output {
        (
            input.0 + local_index,
            u64::cast_from(input.1) + u64::cast_from(local_index),
        )
    }
}

#[test]
fn flat_map_expands_in_stable_input_order() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[2_u32, 0, 3]);

    let output = flat_map(&exec, input.slice(..), RepeatValue).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![20, 21, 30, 31, 32]);
}

#[test]
fn flat_map_handles_empty_input() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.alloc::<u32>(0);

    let output = flat_map(&exec, input.slice(..), RepeatValue).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), Vec::<u32>::new());
}

#[test]
fn flat_map_handles_nonempty_input_with_no_outputs() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[0_u32, 0, 0]);

    let output = flat_map(&exec, input.slice(..), RepeatValue).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), Vec::<u32>::new());
}

#[test]
fn flat_map_supports_multi_column_inputs_and_outputs() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let counts = exec.to_device(&[2_u32, 0, 1]);
    let values = exec.to_device(&[10_u32, 20, 30]);
    let input = zip2(counts.slice(..), values.slice(..));

    let output = flat_map(&exec, input, ExpandPair).unwrap();
    let (left, right) = MStorage::into_columns(output);

    assert_eq!(exec.to_host(&left).unwrap(), vec![2, 3, 1]);
    assert_eq!(exec.to_host(&right).unwrap(), vec![10_u64, 11, 30]);
}
