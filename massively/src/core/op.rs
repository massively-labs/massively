//! Operations embedded in read expressions.

use cubecl::prelude::*;

use crate::MIndex;

/// Compile-time unary operation used by [`crate::lazy::Map`].
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op, vector::map};
///
/// struct Square;
///
/// #[cubecl::cube]
/// impl op::UnaryOp<u32> for Square {
///     type Output = u32;
///
///     fn apply(value: u32) -> u32 {
///         value * value
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[2_u32, 3, 4]);
/// let output = map(&exec, input.slice(..), Square).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![4, 9, 16]);
/// ```
#[cubecl::cube]
pub trait UnaryOp<Input: CubeType>: 'static + Send + Sync {
    type Output: CubeType + Send + Sync + 'static;

    fn apply(input: Input) -> Self::Output;
}

/// Compile-time indexed expansion operation.
///
/// `count` returns the number of output items produced by one input item.
/// `generate` returns the item at `local_index`, where callers guarantee
/// `local_index < count(input)`.
#[cubecl::cube]
pub trait ExpandOp<Input: CubeType>: 'static + Send + Sync {
    type Output: CubeType + Send + Sync + 'static;

    fn count(input: Input) -> MIndex;

    fn generate(input: Input, local_index: MIndex) -> Self::Output;
}

/// Internal index-aware transform operation.
#[doc(hidden)]
#[cubecl::cube]
pub trait IndexedUnaryOp<Input: CubeType>: 'static + Send + Sync {
    type Output: CubeType + Send + Sync + 'static;

    fn apply(input: Input, index: u32) -> Self::Output;
}

/// Internal index-aware operation over an adjacent input pair.
#[doc(hidden)]
#[cubecl::cube]
pub trait IndexedBinaryOp<Input: CubeType>: 'static + Send + Sync {
    type Output: CubeType + Send + Sync + 'static;

    fn apply(previous: Input, current: Input, index: u32) -> Self::Output;
}

/// Identity operation.
///
/// # Examples
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, op, vector::map};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let input = exec.to_device(&[1_u32, 2, 3]);
/// let output = map(&exec, input.slice(..), op::Identity).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 3]);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Identity;

#[cubecl::cube]
impl<Input: CubeType + Send + Sync + 'static> UnaryOp<Input> for Identity {
    type Output = Input;

    fn apply(input: Input) -> Input {
        input
    }
}
