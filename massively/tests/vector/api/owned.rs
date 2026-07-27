use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::{
    Executor, MFlag, MStorage,
    op::{BinaryPredicateOp, PredicateOp, ReductionOp, UnaryOp},
    vector,
};

struct AddOne;
struct Triple;
struct Add;
struct Less;
struct Equal;
struct Even;

#[cubecl::cube]
impl UnaryOp<u32> for AddOne {
    type Output = u32;

    fn apply(value: u32) -> u32 {
        value + 1
    }
}

#[cubecl::cube]
impl UnaryOp<u32> for Triple {
    type Output = (u32, u32, u32);

    fn apply(value: u32) -> Self::Output {
        (value, value + 10, value + 20)
    }
}

#[cubecl::cube]
impl ReductionOp<u32> for Add {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Less {
    fn apply(lhs: u32, rhs: u32) -> MFlag {
        massively::flag::from_bool(lhs < rhs)
    }
}

#[cubecl::cube]
impl BinaryPredicateOp<u32> for Equal {
    fn apply(lhs: u32, rhs: u32) -> MFlag {
        massively::flag::from_bool(lhs == rhs)
    }
}

#[cubecl::cube]
impl PredicateOp<u32> for Even {
    fn apply(value: u32) -> MFlag {
        massively::flag::from_bool(value % 2 == 0)
    }
}

#[test]
fn owned_vector_apis_return_device_storage() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[3_u32, 1, 2, 2]);
    let stencil = exec.to_device(&[1_u32, 0, 1, 0]);
    let indices = exec.to_device(&[1_u32, 3, 0]);

    let mapped = vector::map(&exec, input.slice(..), AddOne).unwrap();
    assert_eq!(exec.to_host(&mapped).unwrap(), vec![4, 2, 3, 3]);

    let inclusive = vector::inclusive_scan(&exec, input.slice(..), Add).unwrap();
    let exclusive = vector::exclusive_scan(&exec, input.slice(..), 0, Add).unwrap();
    let adjacent = vector::adjacent_difference(&exec, input.slice(..), Add).unwrap();
    assert_eq!(exec.to_host(&inclusive).unwrap(), vec![3, 4, 6, 8]);
    assert_eq!(exec.to_host(&exclusive).unwrap(), vec![0, 3, 4, 6]);
    assert_eq!(exec.to_host(&adjacent).unwrap(), vec![3, 4, 3, 4]);

    let gathered = vector::gather(&exec, input.slice(..), indices.slice(..)).unwrap();
    let reversed = vector::reverse(&exec, input.slice(..)).unwrap();
    assert_eq!(exec.to_host(&gathered).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(&reversed).unwrap(), vec![2, 2, 1, 3]);

    let sorted = vector::sort(&exec, input.slice(..), Less).unwrap();
    let unique = vector::unique(&exec, sorted.slice(..), Equal).unwrap();
    assert_eq!(exec.to_host(&sorted).unwrap(), vec![1, 2, 2, 3]);
    assert_eq!(exec.to_host(&unique).unwrap(), vec![1, 2, 3]);

    let copied = vector::copy_where(&exec, input.slice(..), stencil.slice(..)).unwrap();
    let removed = vector::remove_where(&exec, input.slice(..), stencil.slice(..)).unwrap();
    let (partitioned, boundary) = vector::partition(&exec, input.slice(..), Even).unwrap();
    let filled = exec.alloc::<u32>(3);
    vector::fill(&exec, 7_u32, filled.slice_mut(..)).unwrap();
    assert_eq!(exec.to_host(&copied).unwrap(), vec![3, 2]);
    assert_eq!(exec.to_host(&removed).unwrap(), vec![1, 2]);
    assert_eq!(boundary, 2);
    assert_eq!(exec.to_host(&partitioned).unwrap(), vec![2, 2, 3, 1]);
    assert_eq!(exec.to_host(&filled).unwrap(), vec![7, 7, 7]);

    let left = exec.to_device(&[1_u32, 2, 2]);
    let right = exec.to_device(&[2_u32, 3]);
    let merged = vector::merge(&exec, left.slice(..), right.slice(..), Less).unwrap();
    let union = vector::set_union(&exec, left.slice(..), right.slice(..), Less).unwrap();
    let intersection =
        vector::set_intersection(&exec, left.slice(..), right.slice(..), Less).unwrap();
    let difference = vector::set_difference(&exec, left.slice(..), right.slice(..), Less).unwrap();
    assert_eq!(exec.to_host(&merged).unwrap(), vec![1, 2, 2, 2, 3]);
    assert_eq!(exec.to_host(&union).unwrap(), vec![1, 2, 2, 3]);
    assert_eq!(exec.to_host(&intersection).unwrap(), vec![2]);
    assert_eq!(exec.to_host(&difference).unwrap(), vec![1, 2]);

    let queries = exec.to_device(&[0_u32, 2, 4]);
    let lower = vector::lower_bound(&exec, sorted.slice(..), queries.slice(..), Less).unwrap();
    let upper = vector::upper_bound(&exec, sorted.slice(..), queries.slice(..), Less).unwrap();
    assert_eq!(exec.to_host(&lower).unwrap(), vec![0, 1, 4]);
    assert_eq!(exec.to_host(&upper).unwrap(), vec![0, 3, 4]);
}

#[test]
fn u32_stencil_transform_treats_every_nonzero_value_as_true() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[10_u32, 20, 30, 40]);
    let stencil = exec.to_device(&[0_u32, 7, u32::MAX, 0]);

    let copied = vector::copy_where(&exec, input.slice(..), stencil.slice(..)).unwrap();

    assert_eq!(exec.to_host(&copied).unwrap(), vec![20, 30]);
}

#[test]
fn synchronized_logical_length_flows_through_an_algorithm_pipeline() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[5_u32, 1, 3, 1, 4, 2]);
    let stencil = exec.to_device(&[1_u32, 1, 1, 1, 0, 0]);

    // copy_where resolves its device-produced length at the public boundary;
    // following fixed-length operations need no further length readback.
    let compacted = vector::copy_where(&exec, input.slice(..), stencil.slice(..)).unwrap();
    let sorted = vector::sort(&exec, compacted.slice(..), Less).unwrap();
    let unique = vector::unique(&exec, sorted.slice(..), Equal).unwrap();
    let incremented = vector::map(&exec, unique.slice(..), AddOne).unwrap();
    let scanned = vector::inclusive_scan(&exec, incremented.slice(..), Add).unwrap();
    let sum = vector::reduce(&exec, scanned.slice(..), 0, Add).unwrap();

    assert_eq!(exec.to_host(&unique).unwrap(), vec![1, 3, 5]);
    assert_eq!(exec.to_host(&scanned).unwrap(), vec![2, 6, 12]);
    assert_eq!(sum, 20);
}

#[test]
fn synchronized_zero_length_remains_composable() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[3_u32, 2, 1]);
    let stencil = exec.to_device(&[0_u32, 0, 0]);

    let compacted = vector::copy_where(&exec, input.slice(..), stencil.slice(..)).unwrap();
    let sorted = vector::sort(&exec, compacted.slice(..), Less).unwrap();
    let unique = vector::unique(&exec, sorted.slice(..), Equal).unwrap();
    let sum = vector::reduce(&exec, unique.slice(..), 7, Add).unwrap();

    assert_eq!(exec.to_host(&unique).unwrap(), Vec::<u32>::new());
    assert_eq!(sum, 7);
}

#[test]
fn device_logical_length_stays_private_and_flows_through_slices() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[10_u32, 20, 30, 40, 50]);
    let stencil = exec.to_device(&[1_u32, 0, 1, 1, 0]);
    let selected = vector::copy_where(&exec, input.slice(..), stencil.slice(..)).unwrap();

    let middle = selected.slice(1..4);
    assert_eq!(exec.to_host(&middle).unwrap(), vec![30, 40]);
}

#[test]
fn owned_by_key_and_flat_tuple_results() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = exec.to_device(&[1_u32, 2]);
    let triples = vector::map(&exec, input.slice(..), Triple).unwrap();
    let (first, second, third) = MStorage::into_columns(triples);
    assert_eq!(exec.to_host(&first).unwrap(), vec![1, 2]);
    assert_eq!(exec.to_host(&second).unwrap(), vec![11, 12]);
    assert_eq!(exec.to_host(&third).unwrap(), vec![21, 22]);

    let keys = exec.to_device(&[2_u32, 1, 1, 3]);
    let values = exec.to_device(&[20_u32, 10, 11, 30]);
    let sorted_keys = vector::sort(&exec, keys.slice(..), Less).unwrap();
    let sorted_values = vector::sort_by_key(&exec, keys.slice(..), values.slice(..), Less).unwrap();
    assert_eq!(exec.to_host(&sorted_keys).unwrap(), vec![1, 1, 2, 3]);
    assert_eq!(exec.to_host(&sorted_values).unwrap(), vec![10, 11, 20, 30]);

    let scanned = vector::inclusive_scan_by_key(
        &exec,
        sorted_keys.slice(..),
        sorted_values.slice(..),
        Equal,
        Add,
    )
    .unwrap();
    let init = 0_u32;
    let exclusive = vector::exclusive_scan_by_key(
        &exec,
        sorted_keys.slice(..),
        sorted_values.slice(..),
        Equal,
        init.clone(),
        Add,
    )
    .unwrap();
    assert_eq!(exec.to_host(&scanned).unwrap(), vec![10, 21, 20, 30]);
    assert_eq!(exec.to_host(&exclusive).unwrap(), vec![0, 10, 0, 0]);

    let (reduced_keys, reduced_values) = vector::reduce_by_key(
        &exec,
        sorted_keys.slice(..),
        sorted_values.slice(..),
        Equal,
        init,
        Add,
    )
    .unwrap();
    let unique_values =
        vector::unique_by_key(&exec, sorted_keys.slice(..), sorted_values.slice(..), Equal)
            .unwrap();
    assert_eq!(exec.to_host(&reduced_keys).unwrap(), vec![1, 2, 3]);
    assert_eq!(exec.to_host(&reduced_values).unwrap(), vec![21, 20, 30]);
    assert_eq!(exec.to_host(&unique_values).unwrap(), vec![10, 20, 30]);

    let left_keys = exec.to_device(&[1_u32, 3]);
    let left_values = exec.to_device(&[10_u32, 30]);
    let right_keys = exec.to_device(&[2_u32, 4]);
    let right_values = exec.to_device(&[20_u32, 40]);
    let merged_values = vector::merge_by_key(
        &exec,
        left_keys.slice(..),
        left_values.slice(..),
        right_keys.slice(..),
        right_values.slice(..),
        Less,
    )
    .unwrap();
    assert_eq!(exec.to_host(&merged_values).unwrap(), vec![10, 20, 30, 40]);
}

#[test]
fn host_values_compose_across_single_value_input_apis() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let empty = exec.alloc::<u32>(0);
    let zero = vector::reduce(&exec, empty.slice(..), 0_u32, Add).unwrap();
    let source = exec.to_device(&[1_u32, 2]);
    let total = vector::reduce(&exec, source.slice(..), zero.clone(), Add).unwrap();

    let scan_input = exec.to_device(&[1_u32, 2, 4]);
    let scanned = vector::exclusive_scan(&exec, scan_input.slice(..), total.clone(), Add).unwrap();
    assert_eq!(exec.to_host(&scanned).unwrap(), vec![3, 4, 6]);

    let keys = exec.to_device(&[0_u32, 0, 1]);
    let values = exec.to_device(&[1_u32, 2, 4]);
    let by_key_scan = vector::exclusive_scan_by_key(
        &exec,
        keys.slice(..),
        values.slice(..),
        Equal,
        zero.clone(),
        Add,
    )
    .unwrap();
    assert_eq!(exec.to_host(&by_key_scan).unwrap(), vec![0, 1, 0]);

    let (reduced_keys, reduced_values) = vector::reduce_by_key(
        &exec,
        keys.slice(..),
        values.slice(..),
        Equal,
        zero.clone(),
        Add,
    )
    .unwrap();
    assert_eq!(exec.to_host(&reduced_keys).unwrap(), vec![0, 1]);
    assert_eq!(exec.to_host(&reduced_values).unwrap(), vec![3, 4]);

    let scatter_values = exec.to_device(&[1_u32, 2]);
    let scatter_indices = exec.to_device(&[0_u32, 0]);
    let scatter_output = exec.to_device(&[10_u32, 20]);
    vector::scatter_reduce(
        &exec,
        scatter_values.slice(..),
        scatter_indices.slice(..),
        zero,
        Add,
        scatter_output.slice_mut(..),
    )
    .unwrap();
    assert_eq!(exec.to_host(&scatter_output).unwrap(), vec![13, 20]);

    let filled = exec.alloc::<u32>(3);
    vector::fill(&exec, total.clone(), filled.slice_mut(..)).unwrap();
    assert_eq!(exec.to_host(&filled).unwrap(), vec![3, 3, 3]);

    let stencil = exec.to_device(&[0_u32, 1, 0]);
    let replaced = exec.to_device(&[8_u32, 8, 8]);
    vector::replace_where(
        &exec,
        total.clone(),
        stencil.slice(..),
        replaced.slice_mut(..),
    )
    .unwrap();
    assert_eq!(exec.to_host(&replaced).unwrap(), vec![8, 3, 8]);

    let filled = exec.alloc::<u32>(2);
    vector::fill(&exec, total, filled.slice_mut(..)).unwrap();
    assert_eq!(exec.to_host(&filled).unwrap(), vec![3, 3]);
}
