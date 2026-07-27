use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use massively::seg::{Segment, SegmentIterator};
use massively::{
    Executor, MIter, MStorage, lazy, op::ReductionOp, op::UnaryOp, vector::gather, vector::map,
    vector::reduce, zip2, zip7,
};

struct Double;
struct Sum;
struct Difference;
struct PairDifference;
struct LookupTable;
struct LookupPairTable;
struct LookupTwoTables;
struct TableLen;

#[cubecl::cube]
impl UnaryOp<massively::MIndex> for Double {
    type Output = u32;

    fn apply(input: massively::MIndex) -> u32 {
        input * 2u32
    }
}

#[cubecl::cube]
impl ReductionOp<u32> for Sum {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

#[cubecl::cube]
impl ReductionOp<u32> for Difference {
    fn apply(previous: u32, current: u32) -> u32 {
        current - previous
    }
}

#[cubecl::cube]
impl ReductionOp<(u32, u32)> for PairDifference {
    fn apply(previous: (u32, u32), current: (u32, u32)) -> (u32, u32) {
        (current.0 - previous.0, current.1 - previous.1)
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, Segment<u32>)> for LookupTable {
    type Output = u32;

    fn apply(input: (u32, Segment<u32>)) -> u32 {
        input.1.at(input.0)
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, Segment<(u32, u32)>)> for LookupPairTable {
    type Output = u32;

    fn apply(input: (u32, Segment<(u32, u32)>)) -> u32 {
        let row = input.1.at(input.0);
        row.0 + row.1
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, Segment<u32>, Segment<u32>)> for LookupTwoTables {
    type Output = u32;

    fn apply(input: (u32, Segment<u32>, Segment<u32>)) -> u32 {
        input.1.at(input.0) + input.2.at(input.0)
    }
}

#[cubecl::cube]
impl UnaryOp<(u32, Segment<u32>)> for TableLen {
    type Output = u32;

    fn apply(input: (u32, Segment<u32>)) -> u32 {
        input.1.len()
    }
}

#[test]
fn public_lazy_constructors_compose_as_miter() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);

    let constant: lazy::Taken<lazy::Constant<u32>> = lazy::constant(3_u32).take(4);
    assert_eq!(reduce(&exec, constant, 0, Sum).unwrap(), 12);

    let counting: lazy::Taken<lazy::Counting> = lazy::counting(1).take(4);
    let output = map(
        &exec,
        lazy::identity(lazy::map(counting, Double)),
        massively::op::Identity,
    )
    .unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![2, 4, 6, 8]);

    let values = exec.to_device(&[10_u32, 20, 30, 40]);
    let permuted = lazy::permute(values.slice(..), lazy::counting(0).take(4));
    assert_eq!(reduce(&exec, permuted, 0, Sum).unwrap(), 100);
}

#[test]
fn stride_is_a_sliceable_arithmetic_progression() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let input = lazy::stride(2, 3).take(6).slice(1..5).slice(1..3);
    let output = map(&exec, input, massively::op::Identity).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![8, 11]);
}

#[test]
fn with_table_shares_an_entire_lazy_iterator_with_every_context() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let indices = exec.to_device(&[3_u32, 0, 2, 1]);
    let values = exec.to_device(&[5_u32, 10, 15, 20]);
    let table = lazy::map(values.slice(..), Double);
    let input = lazy::with_table(indices.slice(..), table);

    let output = map(&exec, input, LookupTable).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![40, 10, 30, 20]);
}

#[test]
fn with_table_broadcasts_an_empty_segment() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let contexts = exec.to_device(&[1_u32, 2, 3]);
    let table = exec.to_device::<u32>(&[]);
    let output = map(
        &exec,
        lazy::with_table(contexts.slice(..), table.slice(..)),
        TableLen,
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![0, 0, 0]);
}

#[test]
fn with_table_supports_multi_column_tables() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let indices = exec.to_device(&[2_u32, 0, 1]);
    let left = exec.to_device(&[1_u32, 2, 3]);
    let right = exec.to_device(&[10_u32, 20, 30]);
    let table = zip2(left.slice(..), right.slice(..));

    let output = map(
        &exec,
        lazy::with_table(indices.slice(..), table),
        LookupPairTable,
    )
    .unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![33, 11, 22]);
}

#[test]
fn slicing_with_table_slices_only_the_contexts() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let indices = exec.to_device(&[4_u32, 3, 2, 1, 0]);
    let table = exec.to_device(&[999_u32, 10, 20, 30, 40, 50, 999]);
    let input = lazy::with_table(indices.slice(..), table.slice(1..6))
        .slice(1..5)
        .slice(1..3);

    let output = map(&exec, input, LookupTable).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![30, 20]);
}

#[test]
fn with_table_can_be_nested_without_materializing_intermediates() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let indices = exec.to_device(&[2_u32, 0, 1]);
    let first = exec.to_device(&[1_u32, 2, 3]);
    let second = exec.to_device(&[10_u32, 20, 30]);
    let input = lazy::with_table(
        lazy::with_table(indices.slice(..), first.slice(..)),
        second.slice(..),
    );

    fn assert_flat_item<R: Runtime, Input: MIter<R, Item = (u32, Segment<u32>, Segment<u32>)>>(
        _input: &Input,
    ) {
    }
    assert_flat_item::<WgpuRuntime, _>(&input);

    let output = map(&exec, input, LookupTwoTables).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![33, 11, 22]);
}

#[test]
fn with_table_matches_permuted_single_segment_iterators() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let indices = exec.to_device(&[2_u32, 0, 1]);
    let first = exec.to_device(&[1_u32, 2, 3]);
    let second = exec.to_device(&[10_u32, 20, 30]);

    let first_table = lazy::permute(
        SegmentIterator::new(first.slice(..), lazy::stride(0, 3).take(2)),
        lazy::constant(0).take(3),
    );
    let second_table = lazy::permute(
        SegmentIterator::new(second.slice(..), lazy::stride(0, 3).take(2)),
        lazy::constant(0).take(3),
    );
    let composed = massively::zip3(indices.slice(..), first_table, second_table);
    let nested = lazy::with_table(
        lazy::with_table(indices.slice(..), first.slice(..)),
        second.slice(..),
    );

    let composed_output = map(&exec, composed, LookupTwoTables).unwrap();
    let nested_output = map(&exec, nested, LookupTwoTables).unwrap();

    assert_eq!(
        exec.to_host(&composed_output).unwrap(),
        exec.to_host(&nested_output).unwrap()
    );
}

#[test]
fn taken_tracks_nested_slice_offsets() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let taken: lazy::Taken<lazy::Counting> = lazy::counting(10).take(8);
    let sliced = taken.slice(2..6).slice(1..3);

    let values = exec.to_device(&(0_u32..20).collect::<Vec<_>>());
    let output = gather(&exec, values.slice(..), sliced).unwrap();

    assert_eq!(exec.to_host(&output).unwrap(), vec![13, 14]);
}

#[test]
fn slicing_a_lazy_permutation_slices_its_logical_rows() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let values = exec.to_device(&[10_u32, 20, 30, 40, 50, 60]);
    let indices = exec.to_device(&[4_u32, 1, 5, 0, 3, 2]);

    let sliced = lazy::permute(values.slice(..), indices.slice(..))
        .slice(1..5)
        .slice(1..3);

    let output = map(&exec, sliced, massively::op::Identity).unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![60, 10]);
}

#[test]
fn slicing_does_not_increase_read_arity_eight() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let inputs: Vec<_> = (0_u32..7)
        .map(|column| {
            let base = column * 10;
            exec.to_device(&[base, base + 1, base + 2, base + 3])
        })
        .collect();

    let sliced = lazy::permute(
        zip7(
            inputs[0].slice(..),
            inputs[1].slice(..),
            inputs[2].slice(..),
            inputs[3].slice(..),
            inputs[4].slice(..),
            inputs[5].slice(..),
            inputs[6].slice(..),
        ),
        lazy::counting(0).take(4),
    )
    .slice(1..3);

    let outputs = map(&exec, sliced, massively::op::Identity).unwrap();

    let (a, b, c, d, e, f, g) = MStorage::into_columns(outputs);
    let outputs = [&a, &b, &c, &d, &e, &f, &g];
    for (column, output) in outputs.into_iter().enumerate() {
        let base = column as u32 * 10;
        assert_eq!(exec.to_host(output).unwrap(), vec![base + 1, base + 2]);
    }
}

#[test]
fn reverse_composes_with_slicing_and_multi_column_inputs() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let empty = exec.alloc::<u32>(0);
    let reversed_empty = lazy::reverse(empty.slice(..));
    let empty_output = map(&exec, reversed_empty, massively::op::Identity).unwrap();
    assert!(exec.to_host(&empty_output).unwrap().is_empty());

    let values = exec.to_device(&[10_u32, 20, 30, 40, 50]);
    let middle = lazy::reverse(values.slice(..)).slice(1..4).slice(1..2);

    let output = map(&exec, middle, massively::op::Identity).unwrap();
    assert_eq!(exec.to_host(&output).unwrap(), vec![30]);

    let first = exec.to_device(&[1_u32, 2, 3]);
    let second = exec.to_device(&[10_u32, 20, 30]);
    let reversed = lazy::reverse(zip2(first.slice(..), second.slice(..)));

    let output = map(&exec, reversed, massively::op::Identity).unwrap();
    let (output_first, output_second) = MStorage::into_columns(output);
    assert_eq!(exec.to_host(&output_first).unwrap(), vec![3, 2, 1]);
    assert_eq!(exec.to_host(&output_second).unwrap(), vec![30, 20, 10]);
}

#[test]
fn repeat_each_is_lazy_sliceable_and_supports_multi_column_rows() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let first = exec.to_device(&[10_u32, 20, 30]);
    let second = exec.to_device(&[1_u32, 2, 3]);
    let repeated = lazy::repeat_each(zip2(first.slice(..), second.slice(..)), 3)
        .slice(2..8)
        .slice(1..5);

    let output = map(&exec, repeated, massively::op::Identity).unwrap();
    let (first, second) = MStorage::into_columns(output);

    assert_eq!(exec.to_host(&first).unwrap(), vec![20, 20, 20, 30]);
    assert_eq!(exec.to_host(&second).unwrap(), vec![2, 2, 2, 3]);
}

#[test]
fn tile_is_lazy_sliceable_and_supports_multi_column_rows() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let first = exec.to_device(&[10_u32, 20, 30]);
    let second = exec.to_device(&[1_u32, 2, 3]);
    let tiled = lazy::tile(zip2(first.slice(..), second.slice(..)), 3)
        .slice(2..8)
        .slice(1..5);

    let output = map(&exec, tiled, massively::op::Identity).unwrap();
    let (first, second) = MStorage::into_columns(output);

    assert_eq!(exec.to_host(&first).unwrap(), vec![10, 20, 30, 10]);
    assert_eq!(exec.to_host(&second).unwrap(), vec![1, 2, 3, 1]);
}

#[test]
fn repeat_each_and_tile_handle_empty_and_zero_repetitions() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let empty = exec.to_device::<u32>(&[]);
    let values = exec.to_device(&[1_u32, 2, 3]);

    let repeated_empty = lazy::repeat_each(empty.slice(..), 5);
    let tiled_empty = lazy::tile(empty.slice(..), 5);
    let repeated_zero = lazy::repeat_each(values.slice(..), 0);
    let tiled_zero = lazy::tile(values.slice(..), 0);

    let repeated_empty = map(&exec, repeated_empty, massively::op::Identity).unwrap();
    let tiled_empty = map(&exec, tiled_empty, massively::op::Identity).unwrap();
    let repeated_zero = map(&exec, repeated_zero, massively::op::Identity).unwrap();
    let tiled_zero = map(&exec, tiled_zero, massively::op::Identity).unwrap();
    assert!(exec.to_host(&repeated_empty).unwrap().is_empty());
    assert!(exec.to_host(&tiled_empty).unwrap().is_empty());
    assert!(exec.to_host(&repeated_zero).unwrap().is_empty());
    assert!(exec.to_host(&tiled_zero).unwrap().is_empty());
}

#[test]
fn repeated_lazy_views_reject_lengths_larger_than_mindex() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let repeated = lazy::repeat_each(lazy::counting(0).take(u32::MAX), 2);
    let tiled = lazy::tile(lazy::counting(0).take(u32::MAX), 2);

    assert!(matches!(
        map(&exec, repeated, massively::op::Identity),
        Err(massively::Error::LengthTooLarge { .. })
    ));
    assert!(matches!(
        map(&exec, tiled, massively::op::Identity),
        Err(massively::Error::LengthTooLarge { .. })
    ));
}

#[test]
fn lazy_adjacent_difference_preserves_global_neighbors_across_slices() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let first = exec.to_device(&[1_u32, 4, 10, 20]);
    let second = exec.to_device(&[2_u32, 8, 18, 32]);
    let differences =
        lazy::adjacent_difference(zip2(first.slice(..), second.slice(..)), PairDifference)
            .slice(1..4)
            .slice(1..3);

    let output = map(&exec, differences, massively::op::Identity).unwrap();
    let (first, second) = MStorage::into_columns(output);

    assert_eq!(exec.to_host(&first).unwrap(), vec![6, 10]);
    assert_eq!(exec.to_host(&second).unwrap(), vec![10, 14]);
}

#[test]
fn lazy_adjacent_difference_handles_empty_and_singleton_inputs() {
    let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    let empty = exec.to_device::<u32>(&[]);
    let singleton = exec.to_device(&[7_u32]);

    let empty_output = map(
        &exec,
        lazy::adjacent_difference(empty.slice(..), Difference),
        massively::op::Identity,
    )
    .unwrap();
    let singleton_output = map(
        &exec,
        lazy::adjacent_difference(singleton.slice(..), Difference),
        massively::op::Identity,
    )
    .unwrap();

    assert!(exec.to_host(&empty_output).unwrap().is_empty());
    assert_eq!(exec.to_host(&singleton_output).unwrap(), vec![7]);
}
