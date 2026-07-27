use cubecl::prelude::{CubeType, Runtime};
use std::marker::PhantomData;

use crate::{
    DeviceVec, Error, Executor, MAlloc, MIter, MIterMut, MStorage, MVec,
    op::{ExpandOp, UnaryOp},
};

struct Count<Op>(PhantomData<Op>);

#[cubecl::cube]
impl<Input, Op> UnaryOp<Input> for Count<Op>
where
    Input: CubeType,
    Op: ExpandOp<Input>,
{
    type Output = u32;

    fn apply(input: Input) -> u32 {
        Op::count(input)
    }
}

struct GenerateOperation<'a, R: Runtime, Input, Op> {
    exec: &'a Executor<R>,
    input: Input,
    element_offsets: &'a DeviceVec<R, u32>,
    owners: &'a DeviceVec<R, u32>,
    _op: PhantomData<Op>,
}

impl<R, Input, Op> crate::api::iter::OutputOperation<R, Op::Output>
    for GenerateOperation<'_, R, Input, Op>
where
    R: Runtime,
    Input: MIter<R>,
    Op: ExpandOp<Input::Item>,
{
    type Result = Result<(), Error>;

    fn run<Output>(self, output: Output) -> Self::Result
    where
        Op::Output: crate::api::iter::KernelRow + crate::allocation::ScratchStorage<R>,
        Output: crate::api::iter::ConcreteOutput<R, Op::Output>,
    {
        crate::expansion::generate::<R, _, _, Op>(
            self.exec,
            &crate::api::iter::lower_fixed::<R, _>(self.input),
            self.element_offsets,
            self.owners,
            &output,
        )
    }
}

pub(crate) fn expand<R, Input, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
) -> Result<(MVec<R, Op::Output>, DeviceVec<R, u32>), Error>
where
    R: Runtime,
    Input: MIter<R>,
    Op: ExpandOp<Input::Item>,
    Op::Output: MAlloc<R>,
{
    let _ = op;
    let input_len = input.capacity()?;
    let offset_count = input_len.checked_add(1).ok_or(Error::LengthTooLarge {
        len: input_len as usize + 1,
    })?;

    let counts: DeviceVec<R, u32> =
        crate::vector::map(exec, input.clone(), Count::<Op>(PhantomData))?;
    let positions = crate::scan::inclusive_scan_u32(exec, &counts)?;
    let output_len_storage = crate::scan::last_u32(exec, &positions)?;
    let output_len = crate::api::value::read::<R, u32>(exec, &output_len_storage)?;

    let element_offsets = exec.alloc::<u32>(offset_count);
    crate::vector::fill(exec, 0u32, element_offsets.slice_mut(..1))?;
    if input_len != 0 {
        crate::vector::copy(exec, positions.slice(..), element_offsets.slice_mut(1..))?;
    }

    let control = crate::seg::control::SegmentControl::from_materialized(
        exec,
        element_offsets.clone(),
        output_len,
    )?;
    let owners = control.ids(exec)?;
    let output = exec.alloc::<Op::Output>(output_len);
    output
        .slice_mut(..)
        .run_output_operation(GenerateOperation::<R, _, Op> {
            exec,
            input,
            element_offsets: &element_offsets,
            owners: &owners,
            _op: PhantomData,
        })?;

    Ok((output, element_offsets))
}

/// Expands each input row into zero or more output rows in stable order.
///
/// [`ExpandOp::count`] determines the size of each row's output range, and
/// [`ExpandOp::generate`] produces every item in that range. The sum of the
/// counts must fit in [`crate::MIndex`].
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op::ExpandOp, vector::flat_map};
///
/// struct Repeat;
///
/// #[cubecl::cube]
/// impl ExpandOp<u32> for Repeat {
///     type Output = u32;
///
///     fn count(input: u32) -> u32 {
///         input
///     }
///
///     fn generate(input: u32, local_index: u32) -> u32 {
///         input * 10 + local_index
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[2_u32, 0, 3]);
/// let output = flat_map(&exec, input.slice(..), Repeat).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![20, 21, 30, 31, 32]);
/// ```
pub fn flat_map<R, Input, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
) -> Result<MVec<R, Op::Output>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Op: ExpandOp<Input::Item>,
    Op::Output: MAlloc<R>,
{
    Ok(expand(exec, input, op)?.0)
}
