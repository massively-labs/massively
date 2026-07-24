use cubecl::prelude::Runtime;

use crate::{Error, Executor, MFlag, MIndex, MIter, MIterMut, MVec, op::BinaryPredicateOp};

/// Finds the first source item equal to any needle.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::find_first_of};
///
/// struct Equal;
///
/// #[cubecl::cube]
/// impl op::BinaryPredicateOp<u32> for Equal {
///     fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
///         massively::flag::from_bool(lhs == rhs)
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let source = exec.to_device(&[1_u32, 2, 3, 4]);
/// let needles = exec.to_device(&[7_u32, 3]);
///
/// let index = find_first_of(&exec, source.slice(..), needles.slice(..), Equal).unwrap();
///
/// assert_eq!(index, Some(2));
/// ```
pub fn find_first_of<R, Source, Needles, Equal>(
    exec: &Executor<R>,
    source: Source,
    needles: Needles,
    equal: Equal,
) -> Result<Option<MIndex>, Error>
where
    R: Runtime,
    Source: MIter<R>,
    Needles: MIter<R, Item = Source::Item>,
    Equal: BinaryPredicateOp<Source::Item>,
{
    let index = crate::search::find_first_of(
        exec,
        crate::api::iter::lower_fixed::<R, _>(source),
        crate::api::iter::lower_fixed::<R, _>(needles),
        equal,
    )?
    .read(exec)?;
    Ok((index != u32::MAX).then_some(index))
}

/// Finds the lower bound of each value.
///
/// `source` must be sorted according to `less`.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::lower_bound};
///
/// struct Less;
///
/// #[cubecl::cube]
/// impl op::BinaryPredicateOp<u32> for Less {
///     fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
///         massively::flag::from_bool(lhs < rhs)
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let source = exec.to_device(&[1_u32, 3, 5, 7]);
/// let values = exec.to_device(&[0_u32, 3, 6, 8]);
/// let output = lower_bound(&exec, source.slice(..), values.slice(..), Less).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![0, 1, 3, 4]);
/// ```
pub fn lower_bound<R, Source, Values, Less>(
    exec: &Executor<R>,
    source: Source,
    values: Values,
    less: Less,
) -> Result<MVec<R, MIndex>, Error>
where
    R: Runtime,
    Source: MIter<R>,
    Values: MIter<R, Item = Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    crate::search::lower_bounds_storage(
        exec,
        crate::api::iter::lower_fixed::<R, _>(source),
        crate::api::iter::lower_fixed::<R, _>(values),
        less,
    )
}

/// Finds lower bounds into caller-provided storage.
#[doc(hidden)]
pub(crate) fn lower_bound_into<R, Source, Values, Output, Less>(
    exec: &Executor<R>,
    source: Source,
    values: Values,
    less: Less,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Source: MIter<R>,
    Values: MIter<R, Item = Source::Item>,
    Output: MIterMut<R, Item = MIndex>,
    Less: BinaryPredicateOp<Source::Item>,
{
    let bounds = crate::search::lower_bounds_storage(
        exec,
        crate::api::iter::lower_fixed::<R, _>(source),
        crate::api::iter::lower_fixed::<R, _>(values),
        less,
    )?;
    crate::api::algorithm::transform::transform_into(
        exec,
        bounds.column(),
        crate::op::Identity,
        output,
    )
}

/// Finds the upper bound of each value.
///
/// `source` must be sorted according to `less`.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::upper_bound};
///
/// struct Less;
///
/// #[cubecl::cube]
/// impl op::BinaryPredicateOp<u32> for Less {
///     fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
///         massively::flag::from_bool(lhs < rhs)
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let source = exec.to_device(&[1_u32, 3, 5, 7]);
/// let values = exec.to_device(&[0_u32, 3, 6, 8]);
/// let output = upper_bound(&exec, source.slice(..), values.slice(..), Less).unwrap();
///
/// assert_eq!(exec.to_host(&output).unwrap(), vec![0, 2, 3, 4]);
/// ```
pub fn upper_bound<R, Source, Values, Less>(
    exec: &Executor<R>,
    source: Source,
    values: Values,
    less: Less,
) -> Result<MVec<R, MIndex>, Error>
where
    R: Runtime,
    Source: MIter<R>,
    Values: MIter<R, Item = Source::Item>,
    Less: BinaryPredicateOp<Source::Item>,
{
    crate::search::upper_bounds_storage(
        exec,
        crate::api::iter::lower_fixed::<R, _>(source),
        crate::api::iter::lower_fixed::<R, _>(values),
        less,
    )
}

/// Finds upper bounds into caller-provided storage.
#[doc(hidden)]
pub(crate) fn upper_bound_into<R, Source, Values, Output, Less>(
    exec: &Executor<R>,
    source: Source,
    values: Values,
    less: Less,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Source: MIter<R>,
    Values: MIter<R, Item = Source::Item>,
    Output: MIterMut<R, Item = MIndex>,
    Less: BinaryPredicateOp<Source::Item>,
{
    let bounds = crate::search::upper_bounds_storage(
        exec,
        crate::api::iter::lower_fixed::<R, _>(source),
        crate::api::iter::lower_fixed::<R, _>(values),
        less,
    )?;
    crate::api::algorithm::transform::transform_into(
        exec,
        bounds.column(),
        crate::op::Identity,
        output,
    )
}

/// Returns whether two ranges contain equal items.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::equal};
///
/// struct Equal;
///
/// #[cubecl::cube]
/// impl op::BinaryPredicateOp<u32> for Equal {
///     fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
///         massively::flag::from_bool(lhs == rhs)
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let left = exec.to_device(&[1_u32, 2, 3]);
/// let right = exec.to_device(&[1_u32, 2, 3]);
///
/// assert!(massively::flag::is_set(
///     equal(&exec, left.slice(..), right.slice(..), Equal).unwrap()
/// ));
/// ```
pub fn equal<R, Left, Right, Equal>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    equal: Equal,
) -> Result<MFlag, Error>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R, Item = Left::Item>,
    Equal: BinaryPredicateOp<Left::Item>,
{
    let left_len = left.len()?;
    let right_len = right.len()?;
    let mismatch = crate::search::equal(
        exec,
        crate::api::iter::lower_fixed::<R, _>(left),
        crate::api::iter::lower_fixed::<R, _>(right),
        equal,
    )?
    .read(exec)?;
    Ok(crate::flag::from_bool(
        left_len == right_len && mismatch >= left_len.min(right_len),
    ))
}

/// Returns the first mismatch.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::mismatch};
///
/// struct Equal;
///
/// #[cubecl::cube]
/// impl op::BinaryPredicateOp<u32> for Equal {
///     fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
///         massively::flag::from_bool(lhs == rhs)
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let left = exec.to_device(&[1_u32, 2, 3]);
/// let right = exec.to_device(&[1_u32, 4, 3]);
///
/// assert_eq!(mismatch(&exec, left.slice(..), right.slice(..), Equal).unwrap(), Some(1));
/// ```
pub fn mismatch<R, Left, Right, Equal>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    equal: Equal,
) -> Result<Option<MIndex>, Error>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R, Item = Left::Item>,
    Equal: BinaryPredicateOp<Left::Item>,
{
    let left_len = left.len()?;
    let right_len = right.len()?;
    let shared_len = left_len.min(right_len);
    let index = crate::search::mismatch(
        exec,
        crate::api::iter::lower_fixed::<R, _>(left),
        crate::api::iter::lower_fixed::<R, _>(right),
        equal,
    )?
    .read(exec)?;
    Ok((index < shared_len || left_len != right_len).then_some(index.min(shared_len)))
}

/// Lexicographically compares two ranges.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{op, Executor, vector::lexicographical_compare};
///
/// struct Less;
///
/// #[cubecl::cube]
/// impl op::BinaryPredicateOp<u32> for Less {
///     fn apply(lhs: u32, rhs: u32) -> massively::MFlag {
///         massively::flag::from_bool(lhs < rhs)
///     }
/// }
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let left = exec.to_device(&[1_u32, 2, 3]);
/// let right = exec.to_device(&[1_u32, 3, 0]);
///
/// assert!(massively::flag::is_set(
///     lexicographical_compare(&exec, left.slice(..), right.slice(..), Less).unwrap()
/// ));
/// ```
pub fn lexicographical_compare<R, Left, Right, Less>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    less: Less,
) -> Result<MFlag, Error>
where
    R: Runtime,
    Left: MIter<R>,
    Right: MIter<R, Item = Left::Item>,
    Less: BinaryPredicateOp<Left::Item>,
{
    Ok(crate::flag::from_bool(
        crate::search::lexicographical_compare(
            exec,
            crate::api::iter::lower_fixed::<R, _>(left),
            crate::api::iter::lower_fixed::<R, _>(right),
            less,
        )?
        .read(exec)?
            != 0,
    ))
}
