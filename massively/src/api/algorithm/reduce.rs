#![allow(private_bounds)]

use cubecl::prelude::Runtime;

use crate::{Error, Executor, MAlloc, MIter, MVal, Scalar, op::ReductionOp};

/// Reduces all input items, starting from `init`.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op, vector::reduce};
///
/// struct Add;
///
/// #[cubecl::cube]
/// impl op::ReductionOp<u32> for Add {
///     fn apply(lhs: u32, rhs: u32) -> u32 {
///         lhs + rhs
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3, 4]);
///
/// let sum = reduce(&exec, input.slice(..), 0_u32, Add).unwrap();
///
/// assert_eq!(sum.read(&exec).unwrap(), 10);
/// ```
pub fn reduce<R, Input, Op>(
    exec: &Executor<R>,
    input: Input,
    init: impl MVal<R, Input::Item>,
    op: Op,
) -> Result<Scalar<R, Input::Item>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Input::Item: MAlloc<R>,
    Op: ReductionOp<Input::Item>,
{
    reduce_value(exec, input, init.into_device(exec)?, op)
}

fn reduce_value<R, Input, Op>(
    exec: &Executor<R>,
    input: Input,
    init: Scalar<R, Input::Item>,
    op: Op,
) -> Result<Scalar<R, Input::Item>, Error>
where
    R: Runtime,
    Input: MIter<R>,
    Input::Item: MAlloc<R>,
    Op: ReductionOp<Input::Item>,
{
    let storage =
        <<Input::Item as MAlloc<R>>::Dispatch as crate::api::iter::ItemDispatch<R>>::reduce(
            exec,
            input,
            init.into_storage(),
            op,
        )?;
    Scalar::from_storage(storage)
}
