use cubecl::prelude::*;

use crate::{
    DeviceVec, Error, Executor, MIndex, MIter, MIterMut, op::BinaryPredicateOp, op::ReductionOp,
};

const BLOCK_SIZE: u32 = 256;

#[cubecl::cube(launch_unchecked)]
fn mark_segment_heads_kernel(offsets: &[u32], segment_count: &[u32], heads: &mut [u32]) {
    let segment = ABSOLUTE_POS as usize;
    if segment < segment_count[0] as usize {
        let start = offsets[segment] as usize;
        let end = offsets[segment + 1usize] as usize;
        if start < end && start < heads.len() {
            heads[start] = segment as u32 + 1u32;
        }
    }
}

#[cubecl::cube(launch_unchecked)]
fn reverse_indices_kernel(offsets: &[u32], ids: &[u32], len: &[u32], indices: &mut [u32]) {
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize {
        let segment = ids[index] as usize - 1usize;
        indices[index] = offsets[segment] + offsets[segment + 1usize] - 1u32 - index as u32;
    }
}

#[cubecl::cube(launch_unchecked)]
fn merge_head_flags_kernel(heads: &[u32], len: &[u32], flags: &mut [u32]) {
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize && heads[index] != 0u32 {
        flags[index] = 1u32;
    }
}

#[cubecl::cube(launch_unchecked)]
fn clear_head_flags_kernel(heads: &[u32], len: &[u32], flags: &mut [u32]) {
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize && heads[index] != 0u32 {
        flags[index] = 0u32;
    }
}

#[cubecl::cube(launch_unchecked)]
fn sorted_until_candidates_kernel(
    offsets: &[u32],
    heads: &[u32],
    ids: &[u32],
    breaks: &[u32],
    len: &[u32],
    candidates: &mut [u32],
) {
    let index = ABSOLUTE_POS as usize;
    if index < len[0] as usize {
        let segment = ids[index] as usize - 1usize;
        candidates[index] = if heads[index] == 0u32 && breaks[index] != 0u32 {
            index as u32 - offsets[segment]
        } else {
            4_294_967_295u32
        };
    }
}

#[cubecl::cube(launch_unchecked)]
fn finish_sorted_until_kernel(
    offsets: &[u32],
    reduced: &[u32],
    segment_count: &[u32],
    output: &mut [u32],
) {
    let segment = ABSOLUTE_POS as usize;
    if segment < segment_count[0] as usize {
        let candidate = reduced[segment];
        output[segment] = if candidate == 4_294_967_295u32 {
            offsets[segment + 1usize] - offsets[segment]
        } else {
            candidate
        };
    }
}

#[cubecl::cube(launch_unchecked)]
fn selected_offsets_kernel(
    input_offsets: &[u32],
    positions: &[u32],
    offset_count: &[u32],
    output_offsets: &mut [u32],
) {
    let index = ABSOLUTE_POS as usize;
    if index < offset_count[0] as usize {
        let end = input_offsets[index] as usize;
        output_offsets[index] = if end == 0usize {
            0u32
        } else {
            positions[end - 1usize]
        };
    }
}

pub(crate) struct MaxU32;

#[cubecl::cube]
impl ReductionOp<u32> for MaxU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        u32::max(lhs, rhs)
    }
}

pub(crate) struct MinU32;

#[cubecl::cube]
impl ReductionOp<u32> for MinU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        u32::min(lhs, rhs)
    }
}

pub(crate) struct SumU32;

#[cubecl::cube]
impl ReductionOp<u32> for SumU32 {
    fn apply(lhs: u32, rhs: u32) -> u32 {
        lhs + rhs
    }
}

pub(crate) struct LessU32;

#[cubecl::cube]
impl BinaryPredicateOp<u32> for LessU32 {
    fn apply(lhs: u32, rhs: u32) -> crate::MFlag {
        crate::flag::from_bool(lhs < rhs)
    }
}

pub(crate) struct SegmentOffsets<R: Runtime> {
    pub(crate) offsets: DeviceVec<R, u32>,
    pub(crate) segment_count: usize,
    pub(crate) value_len: usize,
}

impl<R: Runtime> SegmentOffsets<R> {
    pub(crate) fn new<Offsets>(
        exec: &Executor<R>,
        offsets: Offsets,
        value_len: usize,
    ) -> Result<Self, Error>
    where
        Offsets: MIter<R, Item = MIndex>,
    {
        let offsets = crate::api::iter::materialize_u32(exec, offsets)?;
        Self::from_materialized(offsets, value_len)
    }

    fn from_materialized(offsets: DeviceVec<R, u32>, value_len: usize) -> Result<Self, Error> {
        let Some(segment_count) = offsets.capacity().checked_sub(1) else {
            return Err(Error::LengthMismatch { left: 1, right: 0 });
        };
        Ok(Self {
            offsets,
            segment_count,
            value_len,
        })
    }

    pub(crate) fn compact<Input, Output, OutputOffsets>(
        &self,
        exec: &Executor<R>,
        input: Input,
        flags: DeviceVec<R, u32>,
        output: Output,
        output_offsets: OutputOffsets,
    ) -> Result<DeviceVec<R, u32>, Error>
    where
        Input: crate::core::facade::KernelInput<R, Item = Output::Item>,
        Output: MIterMut<R>,
        OutputOffsets: MIterMut<R, Item = MIndex>,
    {
        compact_with_offsets(
            exec,
            &self.offsets,
            self.segment_count,
            self.value_len,
            input,
            flags,
            output,
            output_offsets,
        )
    }
}

pub(crate) struct SegmentControl<R: Runtime> {
    pub(crate) offsets: DeviceVec<R, u32>,
    pub(crate) heads: DeviceVec<R, u32>,
    pub(crate) segment_count: usize,
    pub(crate) value_len: usize,
}

impl<R: Runtime> SegmentControl<R> {
    pub(crate) fn new<Offsets>(
        exec: &Executor<R>,
        offsets: Offsets,
        value_len: usize,
    ) -> Result<Self, Error>
    where
        Offsets: MIter<R, Item = MIndex>,
    {
        let offsets = SegmentOffsets::new(exec, offsets, value_len)?;
        Self::from_offsets(exec, offsets)
    }

    pub(crate) fn from_materialized(
        exec: &Executor<R>,
        offsets: DeviceVec<R, u32>,
        value_len: usize,
    ) -> Result<Self, Error> {
        let offsets = SegmentOffsets::from_materialized(offsets, value_len)?;
        Self::from_offsets(exec, offsets)
    }

    fn from_offsets(exec: &Executor<R>, offsets: SegmentOffsets<R>) -> Result<Self, Error> {
        let SegmentOffsets {
            offsets,
            segment_count,
            value_len,
        } = offsets;
        let zero = exec.value(0u32)?;
        let heads = exec.full_value(value_len, &zero)?;

        if segment_count != 0 && value_len != 0 {
            let segment_count_u32 = u32::try_from(segment_count)
                .map_err(|_| Error::LengthTooLarge { len: segment_count })?;
            let segment_count_handle = exec
                .client()
                .create_from_slice(u32::as_bytes(&[segment_count_u32]));
            unsafe {
                mark_segment_heads_kernel::launch_unchecked::<R>(
                    exec.client(),
                    crate::launch::cube_count_1d(segment_count.div_ceil(BLOCK_SIZE as usize))?,
                    CubeDim::new_1d(BLOCK_SIZE),
                    BufferArg::from_raw_parts(offsets.handle.clone(), offsets.capacity()),
                    BufferArg::from_raw_parts(segment_count_handle, 1),
                    BufferArg::from_raw_parts(heads.handle.clone(), heads.capacity()),
                );
            }
        }

        Ok(Self {
            offsets,
            heads,
            segment_count,
            value_len,
        })
    }

    /// Expands sparse segment heads into one source-segment id per value.
    ///
    /// Most segmented algorithms consume head flags directly and avoid this
    /// full-length scan. Only operations that need random access to both
    /// segment bounds request ids.
    pub(crate) fn ids(&self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        let ids = exec.alloc::<u32>(self.value_len);
        if self.value_len != 0 {
            crate::vector::inclusive_scan_into(
                exec,
                self.heads.slice(..),
                MaxU32,
                ids.slice_mut(..),
            )?;
        }
        Ok(ids)
    }

    pub(crate) fn reverse_indices(&self, exec: &Executor<R>) -> Result<DeviceVec<R, u32>, Error> {
        let indices = exec.alloc::<u32>(self.value_len);
        if self.value_len == 0 {
            return Ok(indices);
        }
        let ids = self.ids(exec)?;
        let len = u32::try_from(self.value_len).map_err(|_| Error::LengthTooLarge {
            len: self.value_len,
        })?;
        let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len]));
        unsafe {
            reverse_indices_kernel::launch_unchecked::<R>(
                exec.client(),
                crate::launch::cube_count_1d(self.value_len.div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(self.offsets.handle.clone(), self.offsets.capacity()),
                BufferArg::from_raw_parts(ids.handle.clone(), ids.capacity()),
                BufferArg::from_raw_parts(len_handle, 1),
                BufferArg::from_raw_parts(indices.handle.clone(), indices.capacity()),
            );
        }
        Ok(indices)
    }

    pub(crate) fn merge_heads(
        &self,
        exec: &Executor<R>,
        flags: &DeviceVec<R, u32>,
    ) -> Result<(), Error> {
        if flags.capacity() != self.value_len {
            return Err(Error::LengthMismatch {
                left: self.value_len,
                right: flags.capacity(),
            });
        }
        if self.value_len == 0 {
            return Ok(());
        }
        let len = u32::try_from(self.value_len).map_err(|_| Error::LengthTooLarge {
            len: self.value_len,
        })?;
        let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len]));
        unsafe {
            merge_head_flags_kernel::launch_unchecked::<R>(
                exec.client(),
                crate::launch::cube_count_1d(self.value_len.div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(self.heads.handle.clone(), self.heads.capacity()),
                BufferArg::from_raw_parts(len_handle, 1),
                BufferArg::from_raw_parts(flags.handle.clone(), flags.capacity()),
            );
        }
        Ok(())
    }

    pub(crate) fn clear_heads(
        &self,
        exec: &Executor<R>,
        flags: &DeviceVec<R, u32>,
    ) -> Result<(), Error> {
        if flags.capacity() != self.value_len {
            return Err(Error::LengthMismatch {
                left: self.value_len,
                right: flags.capacity(),
            });
        }
        if self.value_len == 0 {
            return Ok(());
        }
        let len = u32::try_from(self.value_len).map_err(|_| Error::LengthTooLarge {
            len: self.value_len,
        })?;
        let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len]));
        unsafe {
            clear_head_flags_kernel::launch_unchecked::<R>(
                exec.client(),
                crate::launch::cube_count_1d(self.value_len.div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(self.heads.handle.clone(), self.heads.capacity()),
                BufferArg::from_raw_parts(len_handle, 1),
                BufferArg::from_raw_parts(flags.handle.clone(), flags.capacity()),
            );
        }
        Ok(())
    }

    pub(crate) fn sorted_until_candidates(
        &self,
        exec: &Executor<R>,
        breaks: &DeviceVec<R, u32>,
    ) -> Result<DeviceVec<R, u32>, Error> {
        if breaks.capacity() != self.value_len {
            return Err(Error::LengthMismatch {
                left: self.value_len,
                right: breaks.capacity(),
            });
        }
        let candidates = exec.alloc::<u32>(self.value_len);
        if self.value_len == 0 {
            return Ok(candidates);
        }
        let ids = self.ids(exec)?;
        let len = u32::try_from(self.value_len).map_err(|_| Error::LengthTooLarge {
            len: self.value_len,
        })?;
        let len_handle = exec.client().create_from_slice(u32::as_bytes(&[len]));
        unsafe {
            sorted_until_candidates_kernel::launch_unchecked::<R>(
                exec.client(),
                crate::launch::cube_count_1d(self.value_len.div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(self.offsets.handle.clone(), self.offsets.capacity()),
                BufferArg::from_raw_parts(self.heads.handle.clone(), self.heads.capacity()),
                BufferArg::from_raw_parts(ids.handle.clone(), ids.capacity()),
                BufferArg::from_raw_parts(breaks.handle.clone(), breaks.capacity()),
                BufferArg::from_raw_parts(len_handle, 1),
                BufferArg::from_raw_parts(candidates.handle.clone(), candidates.capacity()),
            );
        }
        Ok(candidates)
    }

    pub(crate) fn finish_sorted_until(
        &self,
        exec: &Executor<R>,
        reduced: &DeviceVec<R, u32>,
    ) -> Result<DeviceVec<R, u32>, Error> {
        if reduced.capacity() != self.segment_count {
            return Err(Error::LengthMismatch {
                left: self.segment_count,
                right: reduced.capacity(),
            });
        }
        let output = exec.alloc::<u32>(self.segment_count);
        if self.segment_count == 0 {
            return Ok(output);
        }
        let segment_count =
            u32::try_from(self.segment_count).map_err(|_| Error::LengthTooLarge {
                len: self.segment_count,
            })?;
        let segment_count_handle = exec
            .client()
            .create_from_slice(u32::as_bytes(&[segment_count]));
        unsafe {
            finish_sorted_until_kernel::launch_unchecked::<R>(
                exec.client(),
                crate::launch::cube_count_1d(self.segment_count.div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(self.offsets.handle.clone(), self.offsets.capacity()),
                BufferArg::from_raw_parts(reduced.handle.clone(), reduced.capacity()),
                BufferArg::from_raw_parts(segment_count_handle, 1),
                BufferArg::from_raw_parts(output.handle.clone(), output.capacity()),
            );
        }
        Ok(output)
    }

    pub(crate) fn compact<Input, Output, OutputOffsets>(
        &self,
        exec: &Executor<R>,
        input: Input,
        flags: DeviceVec<R, u32>,
        output: Output,
        output_offsets: OutputOffsets,
    ) -> Result<DeviceVec<R, u32>, Error>
    where
        Input: crate::core::facade::KernelInput<R, Item = Output::Item>,
        Output: MIterMut<R>,
        OutputOffsets: MIterMut<R, Item = MIndex>,
    {
        compact_with_offsets(
            exec,
            &self.offsets,
            self.segment_count,
            self.value_len,
            input,
            flags,
            output,
            output_offsets,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn compact_with_offsets<R, Input, Output, OutputOffsets>(
    exec: &Executor<R>,
    offsets: &DeviceVec<R, u32>,
    segment_count: usize,
    value_len: usize,
    input: Input,
    flags: DeviceVec<R, u32>,
    output: Output,
    output_offsets: OutputOffsets,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Input: crate::core::facade::KernelInput<R, Item = Output::Item>,
    Output: MIterMut<R>,
    OutputOffsets: MIterMut<R, Item = MIndex>,
{
    let positions = crate::core::scan::inclusive_scan_u32(exec, &flags)?;
    let offset_count = segment_count + 1;
    let selected_offsets = if value_len == 0 {
        let zero = exec.value(0u32)?;
        exec.full_value(offset_count, &zero)?
    } else {
        let selected_offsets = exec.alloc::<u32>(offset_count);
        let offset_count_u32 =
            u32::try_from(offset_count).map_err(|_| Error::LengthTooLarge { len: offset_count })?;
        let offset_count_handle = exec
            .client()
            .create_from_slice(u32::as_bytes(&[offset_count_u32]));
        unsafe {
            selected_offsets_kernel::launch_unchecked::<R>(
                exec.client(),
                crate::launch::cube_count_1d(offset_count.div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(offsets.handle.clone(), offsets.capacity()),
                BufferArg::from_raw_parts(positions.handle.clone(), positions.capacity()),
                BufferArg::from_raw_parts(offset_count_handle, 1),
                BufferArg::from_raw_parts(
                    selected_offsets.handle.clone(),
                    selected_offsets.capacity(),
                ),
            );
        }
        selected_offsets
    };
    crate::api::algorithm::transform::transform_into(
        exec,
        selected_offsets.slice(..),
        crate::op::Identity,
        output_offsets,
    )?;

    let selection = crate::selection::SelectionControl::from_positions(exec, positions)?;
    crate::vector::apply_permutation_prefix_into(
        exec,
        input,
        selection.indices().column(),
        selection.count(),
        output,
    )?;
    Ok(selection.count().clone())
}
