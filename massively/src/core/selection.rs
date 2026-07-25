//! Stable selection composed from scan, permutation, and materialization.

use cubecl::prelude::*;

use crate::{
    A1, Column, ColumnMut, Constant, DeviceVec, Dispatch, Error, Executor, MFlag, ReadExpression,
    RowStorage, S1, StorageLayout, Transform, Zip,
    indexed::GatherInput,
    op::UnaryOp,
    output::{LowerOutputExpression, OutputExpression, SliceOutput, StageOutput},
    read::{Env0, Env1, LowerReadExpression},
    reduce::{ReductionOp, StageRead},
    scan::{InclusiveScanDispatch, inclusive_scan, inclusive_scan_u32, last_u32},
    storage::Concat,
    transform::materialize,
};

const BLOCK_SIZE: u32 = 256;

#[cubecl::cube(launch_unchecked)]
fn selected_indices_kernel(positions: &[u32], len: &[u32], indices: &mut [u32]) {
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize {
        let previous = if index == 0usize {
            0u32
        } else {
            positions[index - 1usize]
        };
        if positions[index] != previous {
            indices[(positions[index] - 1u32) as usize] = index as u32;
        }
    }
}

#[cubecl::cube(launch_unchecked)]
fn partition_permutation_kernel(
    positions: &[u32],
    selected_count: &[u32],
    len: &[u32],
    permutation: &mut [u32],
) {
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize {
        let previous = if index == 0usize {
            0u32
        } else {
            positions[index - 1usize]
        };
        let destination = if positions[index] != previous {
            positions[index] - 1u32
        } else {
            selected_count[0] + index as u32 - positions[index]
        };
        permutation[destination as usize] = index as u32;
    }
}

struct IsTrue;

#[cubecl::cube]
impl UnaryOp<MFlag> for IsTrue {
    type Output = MFlag;
    fn apply(input: MFlag) -> MFlag {
        crate::flag::from_bool(crate::flag::is_set(input))
    }
}

struct IsFalse;

#[cubecl::cube]
impl UnaryOp<MFlag> for IsFalse {
    type Output = MFlag;
    fn apply(input: MFlag) -> MFlag {
        crate::flag::from_bool(!crate::flag::is_set(input))
    }
}

struct SumU32;

#[cubecl::cube]
impl ReductionOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

/// Recursive fill over output leaves.
#[doc(hidden)]
pub trait FillOutput<R: Runtime>: OutputExpression + Sized {
    fn fill_output(self, exec: &Executor<R>, value: Self::Item) -> Result<(), Error>;
}

impl<R, T> FillOutput<R> for ColumnMut<T>
where
    R: Runtime,
    T: crate::MStorageElement + StorageLayout<StorageArity = S1>,
    Constant<T>: ReadExpression<Item = T, ReadArity = A1>
        + LowerReadExpression<Slots = Env1<T>>
        + StageRead<R, Env0>,
    ColumnMut<T>: OutputExpression<Item = T, StorageArity = S1>
        + LowerOutputExpression<Slots = Env1<T>>
        + StageOutput<R, Env0>,
{
    fn fill_output(self, exec: &Executor<R>, value: T) -> Result<(), Error> {
        let len = self.capacity();
        materialize(exec, Constant::new(value, len), self)
    }
}

impl<R, Left, Right> FillOutput<R> for Zip<Left, Right>
where
    R: Runtime,
    Left: FillOutput<R>,
    Right: FillOutput<R>,
    Zip<Left, Right>: OutputExpression,
    <Left::Item as StorageLayout>::StorageLeaves: Concat<
            <Right::Item as StorageLayout>::StorageLeaves,
            Output = <<Zip<Left, Right> as OutputExpression>::Item as StorageLayout>::StorageLeaves,
        >,
{
    fn fill_output(self, exec: &Executor<R>, value: Self::Item) -> Result<(), Error> {
        let left_len = self.0.logical_len()?;
        let right_len = self.1.logical_len()?;
        if left_len != right_len {
            return Err(Error::LengthMismatch {
                left: left_len,
                right: right_len,
            });
        }
        let (left, right) =
            <Left::Item as StorageLayout>::StorageLeaves::split(value.into_storage_leaves());
        self.0
            .fill_output(exec, Left::Item::from_storage_leaves(left))?;
        self.1
            .fill_output(exec, Right::Item::from_storage_leaves(right))
    }
}

impl<R, Output> FillOutput<R> for crate::output::Slice<R, Output>
where
    R: Runtime,
    Output: FillOutput<R>,
{
    fn fill_output(self, exec: &Executor<R>, value: Self::Item) -> Result<(), Error> {
        self.into_inner().fill_output(exec, value)
    }
}

/// Internal capability to consume a logical stencil expression.
#[doc(hidden)]
pub trait FlagInput<R: Runtime>: ReadExpression<Item = MFlag> + Sized {
    fn flag_len(&self) -> Result<usize, Error>;
    fn flag_extent(&self) -> Result<crate::extent::LogicalExtent, Error>;
    fn selected_control(self, exec: &Executor<R>) -> Result<SelectionControl<R>, Error>;
    fn rejected_control(self, exec: &Executor<R>) -> Result<SelectionControl<R>, Error>;
}

impl<R, Stencil> FlagInput<R> for Stencil
where
    R: Runtime,
    Stencil: ReadExpression<Item = MFlag> + LowerReadExpression + StageRead<R, Env0>,
    Transform<Stencil, IsTrue>:
        ReadExpression<Item = MFlag> + LowerReadExpression + StageRead<R, Env0>,
    Transform<Stencil, IsFalse>:
        ReadExpression<Item = MFlag> + LowerReadExpression + StageRead<R, Env0>,
    Dispatch<crate::A13, crate::S12>: InclusiveScanDispatch<
            R,
            Transform<Stencil, IsTrue>,
            ColumnMut<u32>,
            u32,
            crate::read::KernelReadSlots<
                <Transform<Stencil, IsTrue> as LowerReadExpression>::Slots,
            >,
            crate::output::KernelOutputSlots<Env1<u32>>,
            SumU32,
        >,
    Dispatch<crate::A13, crate::S12>: InclusiveScanDispatch<
            R,
            Transform<Stencil, IsFalse>,
            ColumnMut<u32>,
            u32,
            crate::read::KernelReadSlots<
                <Transform<Stencil, IsFalse> as LowerReadExpression>::Slots,
            >,
            crate::output::KernelOutputSlots<Env1<u32>>,
            SumU32,
        >,
    ColumnMut<u32>: StageOutput<R, Env0>,
{
    fn flag_len(&self) -> Result<usize, Error> {
        self.logical_len()
    }

    fn flag_extent(&self) -> Result<crate::extent::LogicalExtent, Error> {
        self.logical_extent()
    }

    fn selected_control(self, exec: &Executor<R>) -> Result<SelectionControl<R>, Error> {
        let len = self.logical_len()?;
        let extent = self.logical_extent()?;
        let mut positions = exec.alloc_row::<u32>(len);
        inclusive_scan(
            exec,
            Transform::new(self, IsTrue),
            SumU32,
            positions.slice_mut_usize(..),
        )?;
        positions.set_logical_extent(extent);
        SelectionControl::from_positions(exec, positions)
    }

    fn rejected_control(self, exec: &Executor<R>) -> Result<SelectionControl<R>, Error> {
        let len = self.logical_len()?;
        let extent = self.logical_extent()?;
        let mut positions = exec.alloc_row::<u32>(len);
        inclusive_scan(
            exec,
            Transform::new(self, IsFalse),
            SumU32,
            positions.slice_mut_usize(..),
        )?;
        positions.set_logical_extent(extent);
        SelectionControl::from_positions(exec, positions)
    }
}

/// Stable selected-row control shared by every payload that uses the same
/// flags.  Keeping this separate from the payload prevents by-key algorithms
/// from coupling key arity to value arity.
#[doc(hidden)]
pub struct SelectionControl<R: Runtime> {
    len: usize,
    source_extent: crate::extent::LogicalExtent,
    indices: DeviceVec<R, u32>,
    count: DeviceVec<R, u32>,
}

impl<R: Runtime> SelectionControl<R> {
    pub(crate) fn from_flags(exec: &Executor<R>, flags: DeviceVec<R, u32>) -> Result<Self, Error> {
        let positions = inclusive_scan_u32(exec, &flags)?;
        Self::from_positions(exec, positions)
    }

    pub(crate) fn from_positions(
        exec: &Executor<R>,
        positions: DeviceVec<R, u32>,
    ) -> Result<Self, Error> {
        let count = last_u32(exec, &positions)?;
        Self::from_positions_with_count(exec, &positions, count)
    }

    fn from_positions_with_count(
        exec: &Executor<R>,
        positions: &DeviceVec<R, u32>,
        count: DeviceVec<R, u32>,
    ) -> Result<Self, Error> {
        let source_extent = positions.logical_extent();
        let mut indices = exec.alloc_row::<u32>(positions.capacity());
        if positions.capacity() != 0 {
            let len_handle = positions.logical_extent().materialize(exec)?;
            unsafe {
                selected_indices_kernel::launch_unchecked::<R>(
                    exec.client(),
                    crate::launch::cube_count_1d(
                        positions.capacity().div_ceil(BLOCK_SIZE as usize),
                    )?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(positions.handle.clone(), positions.capacity()),
                    BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
                    BufferArg::from_raw_parts(indices.handle.clone(), indices.capacity()),
                );
            }
        }
        indices.set_logical_extent(crate::extent::LogicalExtent::from_device(
            &count,
            positions.capacity(),
        ));
        Ok(Self {
            len: positions.capacity(),
            source_extent,
            indices,
            count,
        })
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn count(&self) -> &DeviceVec<R, u32> {
        &self.count
    }

    pub(crate) fn indices(&self) -> &DeviceVec<R, u32> {
        &self.indices
    }

    pub(crate) fn source_extent(&self) -> crate::extent::LogicalExtent {
        self.source_extent.clone()
    }
}

/// Internal capability for copying selected rows from a readable expression.
#[doc(hidden)]
pub trait CopySelected<R: Runtime, Output>: ReadExpression + Sized {
    fn source_len(&self) -> Result<usize, Error>;
    fn source_extent(&self) -> Result<crate::extent::LogicalExtent, Error>;
    fn copy_selected(
        self,
        exec: &Executor<R>,
        control: &SelectionControl<R>,
        output: Output,
    ) -> Result<DeviceVec<R, u32>, Error>;
}

impl<R, Input, Output> CopySelected<R, Output> for Input
where
    R: Runtime,
    Input: crate::indexed::IndexedCopyInput<R, Column<crate::MIndex>, Output> + StageRead<R, Env0>,
    Output: OutputExpression,
{
    fn source_len(&self) -> Result<usize, Error> {
        self.logical_len()
    }

    fn source_extent(&self) -> Result<crate::extent::LogicalExtent, Error> {
        self.logical_extent()
    }

    fn copy_selected(
        self,
        exec: &Executor<R>,
        control: &SelectionControl<R>,
        output: Output,
    ) -> Result<DeviceVec<R, u32>, Error> {
        let source_len = self.logical_len()?;
        if source_len != control.len() {
            return Err(Error::LengthMismatch {
                left: source_len,
                right: control.len(),
            });
        }
        self.logical_extent()?.zipped(&control.source_extent())?;
        let output_len = output.logical_len()?;
        if output_len < control.len() {
            return Err(Error::OutputTooShort {
                input: control.len(),
                output: output_len,
            });
        }
        self.indexed_copy_selected(
            exec,
            control.indices.column(),
            None,
            Some(control.count()),
            true,
            output,
        )?;
        Ok(control.count().clone())
    }
}

/// Stably copies values whose stencil is nonzero.
pub(crate) fn copy_where<R, Input, Stencil, Output>(
    exec: &Executor<R>,
    input: Input,
    stencil: Stencil,
    output: Output,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Input: CopySelected<R, Output>,
    Stencil: FlagInput<R>,
{
    let input_len = input.source_len()?;
    let stencil_len = stencil.flag_len()?;
    if input_len != stencil_len {
        return Err(Error::LengthMismatch {
            left: input_len,
            right: stencil_len,
        });
    }
    input.source_extent()?.zipped(&stencil.flag_extent()?)?;
    let control = stencil.selected_control(exec)?;
    input.copy_selected(exec, &control, output)
}

/// Stably copies values whose stencil is zero.
pub(crate) fn remove_where<R, Input, Stencil, Output>(
    exec: &Executor<R>,
    input: Input,
    stencil: Stencil,
    output: Output,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Input: CopySelected<R, Output>,
    Stencil: FlagInput<R>,
{
    let input_len = input.source_len()?;
    let stencil_len = stencil.flag_len()?;
    if input_len != stencil_len {
        return Err(Error::LengthMismatch {
            left: input_len,
            right: stencil_len,
        });
    }
    input.source_extent()?.zipped(&stencil.flag_extent()?)?;
    let control = stencil.rejected_control(exec)?;
    input.copy_selected(exec, &control, output)
}

/// Stably partitions passing values before failing values.
pub(crate) fn partition<R, Input, Pred, Output>(
    exec: &Executor<R>,
    input: Input,
    _pred: Pred,
    output: Output,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Input: Clone
        + crate::predicate::PredicateInput<R, Pred>
        + CopySelected<R, Output>
        + GatherInput<R, Column<crate::MIndex>, Output>,
    Output: SliceOutput,
{
    let len = input.source_len()?;
    let output_len = output.logical_len()?;
    if output_len < len {
        return Err(Error::OutputTooShort {
            input: len,
            output: output_len,
        });
    }
    let positions = input.clone().predicate_positions(exec)?;
    let passing = last_u32(exec, &positions)?;
    let extent = input.source_extent()?;
    let mut permutation = exec.alloc_row::<u32>(len);
    permutation.set_logical_extent(extent.clone());
    if len != 0 {
        let len_handle = extent.materialize(exec)?;
        unsafe {
            partition_permutation_kernel::launch_unchecked::<R>(
                exec.client(),
                crate::launch::cube_count_1d(len.div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(positions.handle.clone(), positions.capacity()),
                BufferArg::from_raw_parts(passing.handle.clone(), 1),
                BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
                BufferArg::from_raw_parts(permutation.handle.clone(), permutation.capacity()),
            );
        }
        input.gather(exec, permutation.column(), output)?;
    }
    Ok(passing)
}

/// Fills every item in an output tree.
pub(crate) fn fill<R, Output>(
    exec: &Executor<R>,
    value: Output::Item,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Output: FillOutput<R>,
{
    output.fill_output(exec, value)
}

/// Replaces items whose logical stencil is true.
pub(crate) fn replace_where<R, Stencil, Output>(
    exec: &Executor<R>,
    value: <Output::Item as crate::allocation::ScratchStorage<R>>::Storage,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    crate::read::FixedRead<
        <<Output::Item as crate::allocation::ScratchStorage<R>>::Storage as RowStorage<R>>::Read,
    >: GatherInput<
            R,
            Constant<u32>,
            <<Output::Item as crate::allocation::ScratchStorage<R>>::Storage as RowStorage<R>>::Write,
        >,
    Stencil: FlagInput<R>,
    Output: OutputExpression,
    Output::Item: crate::allocation::ScratchStorage<R>,
    <Output::Item as crate::allocation::ScratchStorage<R>>::Storage: RowStorage<R>,
    <<Output::Item as crate::allocation::ScratchStorage<R>>::Storage as RowStorage<R>>::Write:
        FillOutput<R>,
    <<Output::Item as crate::allocation::ScratchStorage<R>>::Storage as RowStorage<R>>::Read:
        crate::indexed::IndexedCopyInput<R, Column<crate::MIndex>, Output>,
{
    let stencil_len = stencil.flag_len()?;
    let output_len = output.logical_len()?;
    if stencil_len != output_len {
        return Err(Error::LengthMismatch {
            left: stencil_len,
            right: output_len,
        });
    }
    let control = stencil.selected_control(exec)?;
    let replacements =
        <Output::Item as crate::allocation::ScratchStorage<R>>::alloc_scratch(exec, control.len());
    crate::read::FixedRead::new(value.read()).gather(
        exec,
        Constant::new(0, control.len()),
        replacements.write(),
    )?;
    crate::indexed::IndexedCopyInput::indexed_copy_selected(
        replacements.read(),
        exec,
        control.indices().column(),
        None,
        Some(control.count()),
        false,
        output,
    )
}

/// Internal capability for a transform selected by a logical stencil.
#[doc(hidden)]
pub trait TransformWhereInput<R: Runtime, Stencil, Output, Op>: ReadExpression + Sized {
    fn transform_where(
        self,
        exec: &Executor<R>,
        op: Op,
        stencil: Stencil,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Input, Stencil, Output, Op> TransformWhereInput<R, Stencil, Output, Op> for Input
where
    R: Runtime,
    Input: ReadExpression + StageRead<R, Env0>,
    Op: UnaryOp<Input::Item>,
    Transform<Input, Op>: ReadExpression<Item = Op::Output>,
    Transform<Input, Op>: crate::indexed::IndexedCopyInput<R, crate::Counting, Output>,
    Stencil: FlagInput<R>,
    Output: OutputExpression,
{
    fn transform_where(
        self,
        exec: &Executor<R>,
        op: Op,
        stencil: Stencil,
        output: Output,
    ) -> Result<(), Error> {
        let input_len = self.logical_len()?;
        let stencil_len = stencil.flag_len()?;
        let output_len = output.logical_len()?;
        if input_len != stencil_len {
            return Err(Error::LengthMismatch {
                left: input_len,
                right: stencil_len,
            });
        }
        if input_len != output_len {
            return Err(Error::LengthMismatch {
                left: input_len,
                right: output_len,
            });
        }
        self.logical_extent()?.zipped(&stencil.flag_extent()?)?;
        let control = stencil.selected_control(exec)?;
        crate::indexed::IndexedCopyInput::indexed_copy_selected(
            Transform::new(self, op),
            exec,
            crate::Counting::new(0, input_len),
            Some(control.indices()),
            Some(control.count()),
            false,
            output,
        )
    }
}

/// Applies `op` only where the logical stencil is true.
pub(crate) fn transform_where<R, Input, Stencil, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    stencil: Stencil,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: TransformWhereInput<R, Stencil, Output, Op>,
{
    input.transform_where(exec, op, stencil, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Counting, Permute, Zip};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn copy_where_preserves_flat_three_column_rows() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let a = exec.to_device(&[1_u32, 2, 3, 4]);
        let b = exec.to_device(&[10_f32, 20.0, 30.0, 40.0]);
        let c = exec.to_device(&[100_i32, 200, 300, 400]);
        let flags = exec.to_device(&[0_u32, 1, 1, 0]);
        let out_a = exec.to_device(&[0_u32; 4]);
        let out_b = exec.to_device(&[0_f32; 4]);
        let out_c = exec.to_device(&[0_i32; 4]);
        let input = Zip::new(a.column(), Zip::new(b.column(), c.column()));
        let output = Zip::new(
            Zip::new(out_a.slice_mut(..), out_b.slice_mut(..)),
            out_c.slice_mut(..),
        );

        let count = copy_where(&exec, input, flags.column(), output).unwrap();
        let count = exec.to_host(&count).unwrap()[0];
        assert_eq!(count, 2);
        assert_eq!(exec.to_host(&out_a.slice(..count)).unwrap(), vec![2, 3]);
        assert_eq!(
            exec.to_host(&out_b.slice(..count)).unwrap(),
            vec![20.0, 30.0]
        );
        assert_eq!(exec.to_host(&out_c.slice(..count)).unwrap(), vec![200, 300]);
    }

    #[test]
    fn fused_stencil_scan_produces_binary_positions() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let flags = exec.to_device(&[0_u32, 7, 3, 0]);
        let positions = exec.alloc_row::<u32>(4);
        inclusive_scan(
            &exec,
            Transform::new(flags.column(), IsTrue),
            SumU32,
            positions.slice_mut(..),
        )
        .unwrap();
        assert_eq!(exec.to_host(&positions).unwrap(), vec![0, 1, 2, 2]);
    }

    #[test]
    fn copy_where_on_seven_leaves_dispatches_eval8_after_permute() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let inputs: Vec<_> = (0_u32..7)
            .map(|base| exec.to_device(&[base * 10 + 1, base * 10 + 2, base * 10 + 3]))
            .collect();
        let outputs: Vec<_> = (0..7).map(|_| exec.to_device(&[0_u32; 3])).collect();
        let flags = exec.to_device(&[1_u32, 0, 1]);
        let input = Zip::new(
            inputs[0].column(),
            Zip::new(
                inputs[1].column(),
                Zip::new(
                    inputs[2].column(),
                    Zip::new(
                        inputs[3].column(),
                        Zip::new(
                            inputs[4].column(),
                            Zip::new(inputs[5].column(), inputs[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let output = Zip::new(
            Zip::new(
                Zip::new(
                    Zip::new(
                        Zip::new(
                            Zip::new(outputs[0].slice_mut(..), outputs[1].slice_mut(..)),
                            outputs[2].slice_mut(..),
                        ),
                        outputs[3].slice_mut(..),
                    ),
                    outputs[4].slice_mut(..),
                ),
                outputs[5].slice_mut(..),
            ),
            outputs[6].slice_mut(..),
        );

        let count = copy_where(&exec, input, flags.column(), output).unwrap();
        assert_eq!(exec.to_host(&count).unwrap(), vec![2]);
        for (column, output) in outputs.iter().enumerate() {
            assert_eq!(
                exec.to_host(&output.slice(..2)).unwrap(),
                vec![column as u32 * 10 + 1, column as u32 * 10 + 3]
            );
        }
    }

    #[test]
    fn remove_where_inverts_nonzero_stencil() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[10_u32, 20, 30, 40]);
        let flags = exec.to_device(&[0_u32, 1, 0, 1]);
        let output = exec.to_device(&[0_u32; 4]);
        let count =
            remove_where(&exec, input.column(), flags.column(), output.slice_mut(..)).unwrap();
        let count = exec.to_host(&count).unwrap()[0];
        assert_eq!(count, 2);
        assert_eq!(exec.to_host(&output.slice(..2)).unwrap(), vec![10, 30]);
    }

    struct IsEven;

    #[cubecl::cube]
    impl crate::op::PredicateOp<u32> for IsEven {
        fn apply(input: u32) -> MFlag {
            crate::flag::from_bool(input % 2u32 == 0u32)
        }
    }

    #[test]
    fn partition_is_stable_on_both_sides() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[3_u32, 2, 4, 1, 6, 5]);
        let output = exec.to_device(&[0_u32; 6]);
        let split = partition(&exec, input.column(), IsEven, output.slice_mut_usize(..)).unwrap();
        let split = exec.to_host(&split).unwrap()[0];
        assert_eq!(split, 3);
        assert_eq!(exec.to_host(&output).unwrap(), vec![2, 4, 6, 3, 1, 5]);
    }

    #[test]
    fn fill_and_replace_where_recurse_over_binary_output_tree() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let a = exec.to_device(&[0_u32; 4]);
        let b = exec.to_device(&[0_f32; 4]);
        let c = exec.to_device(&[0_i32; 4]);
        let output = || {
            Zip::new(
                Zip::new(a.slice_mut_usize(..), b.slice_mut_usize(..)),
                c.slice_mut_usize(..),
            )
        };
        fill(&exec, (7_u32, 2.5_f32, -3_i32), output()).unwrap();
        let flags = exec.to_device(&[0_u32, 1, 0, 1]);
        let value = exec.scalar((9_u32, 4.5_f32, -8_i32)).unwrap();
        replace_where(
            &exec,
            value.into_scratch_storage(),
            flags.column(),
            output(),
        )
        .unwrap();
        assert_eq!(exec.to_host(&a).unwrap(), vec![7, 9, 7, 9]);
        assert_eq!(exec.to_host(&b).unwrap(), vec![2.5, 4.5, 2.5, 4.5]);
        assert_eq!(exec.to_host(&c).unwrap(), vec![-3, -8, -3, -8]);
    }

    type Seven = (u32, u32, u32, u32, u32, u32, u32);

    struct IncrementSeven;

    #[cubecl::cube]
    impl UnaryOp<Seven> for IncrementSeven {
        type Output = Seven;
        fn apply(input: Seven) -> Seven {
            (
                input.0 + 1u32,
                input.1 + 1u32,
                input.2 + 1u32,
                input.3 + 1u32,
                input.4 + 1u32,
                input.5 + 1u32,
                input.6 + 1u32,
            )
        }
    }

    #[test]
    fn transform_where_normalizes_eval8_before_selected_storage7_copy() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let inputs: Vec<_> = (0_u32..7)
            .map(|base| exec.to_device(&[base * 10 + 1, base * 10 + 2, base * 10 + 3]))
            .collect();
        let outputs: Vec<_> = (0..7).map(|_| exec.to_device(&[100_u32; 3])).collect();
        let stencil = exec.to_device(&[1_u32, 0, 1]);
        let seven = Zip::new(
            inputs[0].column(),
            Zip::new(
                inputs[1].column(),
                Zip::new(
                    inputs[2].column(),
                    Zip::new(
                        inputs[3].column(),
                        Zip::new(
                            inputs[4].column(),
                            Zip::new(inputs[5].column(), inputs[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let input = Permute::new(seven, Counting::new(0, 3));
        let output = Zip::new(
            Zip::new(
                Zip::new(
                    Zip::new(
                        Zip::new(
                            Zip::new(outputs[0].slice_mut(..), outputs[1].slice_mut(..)),
                            outputs[2].slice_mut(..),
                        ),
                        outputs[3].slice_mut(..),
                    ),
                    outputs[4].slice_mut(..),
                ),
                outputs[5].slice_mut(..),
            ),
            outputs[6].slice_mut(..),
        );

        transform_where(&exec, input, IncrementSeven, stencil.column(), output).unwrap();
        for (column, output) in outputs.iter().enumerate() {
            assert_eq!(
                exec.to_host(output).unwrap(),
                vec![column as u32 * 10 + 2, 100, column as u32 * 10 + 4]
            );
        }
    }
}
