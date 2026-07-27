use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::*;

struct IsTwo;

#[cubecl::cube]
impl op::PredicateOp<u32> for IsTwo {
    fn apply(value: u32) -> MFlag {
        flag::from_bool(value == 2u32)
    }
}

fn assert_flat_triple<I: MIterMut<WgpuRuntime, Item = (u32, u32, u32)>>(_value: &I) {}

#[test]
fn zip_tree_type_is_opaque_but_its_flat_row_contract_is_public() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let a = exec.alloc::<u32>(1);
    let b = exec.alloc::<u32>(1);
    let c = exec.alloc::<u32>(1);

    let left = zip2(zip2(a.slice_mut(..), b.slice_mut(..)), c.slice_mut(..));
    let right = zip2(a.slice_mut(..), zip2(b.slice_mut(..), c.slice_mut(..)));
    assert_flat_triple(&left);
    assert_flat_triple(&right);
}

#[test]
fn public_device_slice_methods_return_public_view_types() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let values: DeviceVec<WgpuRuntime, u32> = exec.to_device(&[1_u32, 2, 3, 4, 5]);

    let read: DeviceSlice<WgpuRuntime, u32> = values.slice(1..5);
    let nested_read: DeviceSlice<WgpuRuntime, u32> = read.slice(1..3);
    assert_eq!(exec.to_host(&nested_read).unwrap(), vec![3, 4]);

    let write: DeviceSliceMut<WgpuRuntime, u32> = values.slice_mut(1..5);
    let read_from_write: DeviceSlice<WgpuRuntime, u32> = write.slice(1..3);
    let nested_write: DeviceSliceMut<WgpuRuntime, u32> = write.slice_mut(1..3);
    assert_eq!(exec.to_host(&read_from_write).unwrap(), vec![3, 4]);
    assert_eq!(exec.to_host(&nested_write).unwrap(), vec![3, 4]);
}

#[test]
fn slice_bounds_are_mindex_values() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let values = exec.to_device(&[10_u32, 20, 30, 40, 50]);
    let start: MIndex = 1;
    let end: MIndex = 4;

    let read = values.slice(start..end);
    let _write = values.slice_mut(start..end);
    let lazy = lazy::counting(10).take(5).slice(start..end);

    assert_eq!(exec.to_host(&read).unwrap(), vec![20, 30, 40]);
    assert!(exec.to_host(&values.slice(end..end)).unwrap().is_empty());
    assert_eq!(
        vector::map(&exec, lazy, op::Identity)
            .and_then(|values| exec.to_host(&values))
            .unwrap(),
        vec![11, 12, 13],
    );
}

#[test]
fn allocation_lengths_are_mindex_values() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let len: MIndex = 3;

    let filled = exec.alloc::<u32>(len);
    vector::fill(&exec, 7_u32, filled.slice_mut(..)).unwrap();

    assert_eq!(exec.to_host(&filled).unwrap(), vec![7, 7, 7]);
}

#[test]
fn host_value_exposes_a_one_item_iterator() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let indices = lazy::constant(0_u32).take(3);

    let gathered = vector::gather(&exec, lazy::constant(7_u32).take(1), indices).unwrap();
    let materialized = vector::map(&exec, lazy::constant(7_u32).take(1), op::Identity).unwrap();

    assert_eq!(exec.to_host(&gathered).unwrap(), vec![7, 7, 7]);
    assert_eq!(exec.to_host(&materialized).unwrap(), vec![7]);
}

#[test]
fn optional_indices_are_resolved_on_the_host() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let values = exec.to_device(&[1_u32, 2, 3]);

    let found = vector::find_if(&exec, values.slice(..), IsTwo).unwrap();
    assert_eq!(found, Some(1));

    let empty = exec.to_device(&[] as &[u32]);
    let missing = vector::find_if(&exec, empty.slice(..), IsTwo).unwrap();
    assert_eq!(missing, None);
}
