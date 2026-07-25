use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::*;

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

    let read: DeviceSlice<u32> = values.slice(1..5);
    let nested_read: DeviceSlice<u32> = read.slice(1..3);
    assert_eq!(exec.to_host(&nested_read).unwrap(), vec![3, 4]);

    let write: DeviceSliceMut<u32> = values.slice_mut(1..5);
    let read_from_write: DeviceSlice<u32> = write.slice(1..3);
    let nested_write: DeviceSliceMut<u32> = write.slice_mut(1..3);
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
    let write = values.slice_mut(start..end);
    let lazy = lazy::counting(10).take(5).slice(start..end);

    assert_eq!(read.len(), 3);
    assert_eq!(write.len(), 3);
    let read_is_empty: bool = read.is_empty();
    let empty_is_empty: bool = values.slice(end..end).is_empty();
    assert!(!read_is_empty);
    assert!(empty_is_empty);
    assert_eq!(MIter::<WgpuRuntime>::len(&lazy).unwrap(), 3);
    assert_eq!(exec.to_host(&read).unwrap(), vec![20, 30, 40]);
}

#[test]
fn allocation_lengths_are_mindex_values() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let len: MIndex = 3;

    let allocated = exec.alloc::<u32>(len);
    let filled = exec.full(len, 7_u32).unwrap();

    assert_eq!(allocated.len(), len);
    assert_eq!(exec.to_host(&filled).unwrap(), vec![7, 7, 7]);
}

#[test]
fn executor_values_round_trip_through_mval() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);

    assert_eq!(exec.value(7_u32).unwrap().read(&exec).unwrap(), 7);
    let pair = exec.value((3_u32, 2.5_f32)).unwrap();
    assert_eq!(pair.clone().read(&exec).unwrap(), (3, 2.5));
    assert_eq!(pair.read(&exec).unwrap(), (3, 2.5));
}

#[test]
fn mval_resolves_host_and_device_representations_in_both_directions() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);

    let from_plain_host: Scalar<WgpuRuntime, u32> = MVal::into_device(7_u32, &exec).unwrap();
    assert_eq!(MVal::into_host(from_plain_host, &exec).unwrap(), 7);

    let host = 11_u32;
    let device: Scalar<WgpuRuntime, u32> = MVal::into_device(host, &exec).unwrap();
    assert_eq!(MVal::into_host(&device, &exec).unwrap(), 11);
    assert_eq!(device.read(&exec).unwrap(), 11);
}

#[test]
fn scalar_exposes_a_one_item_iterator_for_gather_and_permute() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let scalar = exec.value(7_u32).unwrap();
    let indices = lazy::constant(0_u32).take(3);

    let gathered = vector::gather(&exec, scalar.as_iter(), indices.clone()).unwrap();
    let permuted = lazy::permute(scalar.as_iter(), indices);
    let materialized = vector::map(&exec, permuted, op::Identity).unwrap();

    assert_eq!(exec.to_host(&gathered).unwrap(), vec![7, 7, 7]);
    assert_eq!(exec.to_host(&materialized).unwrap(), vec![7, 7, 7]);
}
