//! Ordered set algorithms composed from bounds, merge, and stable selection.

use cubecl::prelude::*;

use crate::{
    DeviceVec, Error, Executor, ReadExpression, StorageLayout,
    eval::Eval13,
    read::{Env0, LowerReadExpression, PaddedReadSlots},
    reduce::{StageRead, StagedBindings},
    selection::{CopySelected, SelectionControl},
    storage::{Decompose, MutableLeaves, MutableLeavesExpand, Recompose},
};

const OCCURRENCE_SIZE: u32 = 64;
const OCCURRENCE_ITEMS: usize = 4;
const OCCURRENCE_TILE: usize = OCCURRENCE_SIZE as usize * OCCURRENCE_ITEMS;

#[cubecl::cube]
trait OccurrenceRule: 'static + Send + Sync {
    fn keep(rank: u32, other_count: u32) -> u32;
}

struct UnionExtra;
struct IntersectionKeep;
struct DifferenceKeep;

#[cubecl::cube]
impl OccurrenceRule for UnionExtra {
    fn keep(rank: u32, other_count: u32) -> u32 {
        if rank >= other_count { 1u32 } else { 0u32 }
    }
}

#[cubecl::cube]
impl OccurrenceRule for IntersectionKeep {
    fn keep(rank: u32, other_count: u32) -> u32 {
        if rank < other_count { 1u32 } else { 0u32 }
    }
}

#[cubecl::cube]
impl OccurrenceRule for DifferenceKeep {
    fn keep(rank: u32, other_count: u32) -> u32 {
        if rank >= other_count { 1u32 } else { 0u32 }
    }
}

macro_rules! define_occurrence_flags_direct_kernel {
    (
        $name:ident,$eval:ident,$method:ident;
        [$( $own_leaf:ident:$own_slot:ident ),+];
        [$( $other_leaf:ident:$other_slot:ident ),+]
    ) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $own_leaf: CubePrimitive, )+
            $( $other_leaf: CubePrimitive, )+
            Leaves: CubeType + MutableLeaves + Send + Sync + 'static,
            Own: $eval<Item, $( $own_leaf ),+>,
            Other: $eval<Item, $( $other_leaf ),+>,
            Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
            Less: crate::op::BinaryPredicateOp<Item>,
            Rule: OccurrenceRule,
        >(
            $( $own_slot: &[$own_leaf], )+
            own_offsets: &[u32],
            $( $other_slot: &[$other_leaf], )+
            other_offsets: &[u32],
            own_len: &[u32],
            other_len: &[u32],
            flags: &mut [u32],
        ) {
            let own_length = own_len[0] as usize;
            let other_length = other_len[0] as usize;
            let tile_start = CUBE_POS as usize * OCCURRENCE_TILE;
            if tile_start < own_length {
                let tile_end = if tile_start + OCCURRENCE_TILE < own_length {
                    tile_start + OCCURRENCE_TILE
                } else {
                    own_length
                };
                let mut windows = Shared::<[u32]>::new_slice(3usize);
                if UNIT_POS == 0u32 {
                    let first = Layout::decompose(Own::$method(
                        $( $own_slot, )+
                        own_offsets,
                        tile_start,
                    ))
                    .into_cells();
                    let last = Layout::decompose(Own::$method(
                        $( $own_slot, )+
                        own_offsets,
                        tile_end - 1usize,
                    ))
                    .into_cells();

                    let own_begin = RuntimeCell::<usize>::new(0usize);
                    let own_begin_high = RuntimeCell::<usize>::new(tile_start + 1usize);
                    while own_begin.read() < own_begin_high.read() {
                        let middle = (own_begin.read() + own_begin_high.read()) / 2usize;
                        if crate::ordering::binary_predicate::<Item, Less>(
                            Own::$method($( $own_slot, )+ own_offsets, middle),
                            Layout::recompose(Leaves::read(&first)),
                        ) {
                            own_begin.store(middle + 1usize);
                        } else {
                            own_begin_high.store(middle);
                        }
                    }

                    let other_begin = RuntimeCell::<usize>::new(0usize);
                    let other_begin_high = RuntimeCell::<usize>::new(other_length);
                    while other_begin.read() < other_begin_high.read() {
                        let middle = (other_begin.read() + other_begin_high.read()) / 2usize;
                        if crate::ordering::binary_predicate::<Item, Less>(
                            Other::$method($( $other_slot, )+ other_offsets, middle),
                            Layout::recompose(Leaves::read(&first)),
                        ) {
                            other_begin.store(middle + 1usize);
                        } else {
                            other_begin_high.store(middle);
                        }
                    }

                    let other_end = RuntimeCell::<usize>::new(other_begin.read());
                    let other_end_high = RuntimeCell::<usize>::new(other_length);
                    while other_end.read() < other_end_high.read() {
                        let middle = (other_end.read() + other_end_high.read()) / 2usize;
                        if !crate::ordering::binary_predicate::<Item, Less>(
                            Layout::recompose(Leaves::read(&last)),
                            Other::$method($( $other_slot, )+ other_offsets, middle),
                        ) {
                            other_end.store(middle + 1usize);
                        } else {
                            other_end_high.store(middle);
                        }
                    }

                    windows[0] = own_begin.read() as u32;
                    windows[1] = other_begin.read() as u32;
                    windows[2] = other_end.read() as u32;
                }
                sync_cube();

                for item in 0usize..OCCURRENCE_ITEMS {
                    let index = tile_start + UNIT_POS as usize + item * OCCURRENCE_SIZE as usize;
                    if index < tile_end {
                        let query = Layout::decompose(Own::$method(
                            $( $own_slot, )+
                            own_offsets,
                            index,
                        ))
                        .into_cells();

                        let own_low = RuntimeCell::<usize>::new(windows[0] as usize);
                        let own_high = RuntimeCell::<usize>::new(index + 1usize);
                        while own_low.read() < own_high.read() {
                            let middle = (own_low.read() + own_high.read()) / 2usize;
                            if crate::ordering::binary_predicate::<Item, Less>(
                                Own::$method($( $own_slot, )+ own_offsets, middle),
                                Layout::recompose(Leaves::read(&query)),
                            ) {
                                own_low.store(middle + 1usize);
                            } else {
                                own_high.store(middle);
                            }
                        }

                        let other_low = RuntimeCell::<usize>::new(windows[1] as usize);
                        let other_low_high = RuntimeCell::<usize>::new(windows[2] as usize);
                        while other_low.read() < other_low_high.read() {
                            let middle = (other_low.read() + other_low_high.read()) / 2usize;
                            if crate::ordering::binary_predicate::<Item, Less>(
                                Other::$method($( $other_slot, )+ other_offsets, middle),
                                Layout::recompose(Leaves::read(&query)),
                            ) {
                                other_low.store(middle + 1usize);
                            } else {
                                other_low_high.store(middle);
                            }
                        }

                        let other_high = RuntimeCell::<usize>::new(other_low.read());
                        let other_high_end = RuntimeCell::<usize>::new(windows[2] as usize);
                        while other_high.read() < other_high_end.read() {
                            let middle = (other_high.read() + other_high_end.read()) / 2usize;
                            if !crate::ordering::binary_predicate::<Item, Less>(
                                Layout::recompose(Leaves::read(&query)),
                                Other::$method($( $other_slot, )+ other_offsets, middle),
                            ) {
                                other_high.store(middle + 1usize);
                            } else {
                                other_high_end.store(middle);
                            }
                        }

                        flags[index] = Rule::keep(
                            index as u32 - own_low.read() as u32,
                            (other_high.read() - other_low.read()) as u32,
                        );
                    }
                }
            }
        }
    };
}

define_occurrence_flags_direct_kernel!(occurrence_flags_direct_a13,Eval13,eval13;
    [L0:own0,L1:own1,L2:own2,L3:own3,L4:own4,L5:own5,L6:own6,L7:own7,L8:own8,L9:own9,L10:own10,L11:own11,L12:own12];
    [R0:other0,R1:other1,R2:other2,R3:other3,R4:other4,R5:other5,R6:other6,R7:other7,R8:other8,R9:other9,R10:other10,R11:other11,R12:other12]
);

trait OccurrenceFlagsDirectInput<R: Runtime, Other>: ReadExpression + Sized {
    fn occurrence_flags_direct<Less, Rule>(
        self,
        exec: &Executor<R>,
        other: Other,
    ) -> Result<DeviceVec<R, u32>, Error>
    where
        Less: crate::op::BinaryPredicateOp<Self::Item>,
        Rule: OccurrenceRule;
}

impl<R, Own, Other> OccurrenceFlagsDirectInput<R, Other> for Own
where
    R: Runtime,
    Own: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Other: ReadExpression<Item = Own::Item> + LowerReadExpression + StageRead<R, Env0>,
    Own::Item: StorageLayout,
    <Own::Item as StorageLayout>::StorageLeaves: CubeType + MutableLeaves + Send + Sync + 'static,
    <Own::Item as StorageLayout>::DeviceLayout: Decompose<Own::Item, Leaves = <Own::Item as StorageLayout>::StorageLeaves>
        + Recompose<Own::Item, Leaves = <Own::Item as StorageLayout>::StorageLeaves>,
{
    fn occurrence_flags_direct<Less, Rule>(
        self,
        exec: &Executor<R>,
        other: Other,
    ) -> Result<DeviceVec<R, u32>, Error>
    where
        Less: crate::op::BinaryPredicateOp<Self::Item>,
        Rule: OccurrenceRule,
    {
        let capacity = self.logical_len()?;
        let extent = self.logical_extent()?;
        let mut flags = exec.alloc_row::<u32>(capacity);
        flags.set_logical_extent(extent.clone());
        if capacity == 0 {
            return Ok(flags);
        }

        let mut own_reads = StagedBindings::new();
        self.stage_at(exec.client(), exec.id(), &mut own_reads)?;
        own_reads.pad_to_thirteen(exec.client());
        let mut other_reads = StagedBindings::new();
        other.stage_at(exec.client(), exec.id(), &mut other_reads)?;
        other_reads.pad_to_thirteen(exec.client());
        let own_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&own_reads.offsets));
        let other_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&other_reads.offsets));
        let own_len = extent.materialize(exec)?;
        let other_len = other.logical_extent()?.materialize(exec)?;

        unsafe {
            occurrence_flags_direct_a13::launch_unchecked::<
                Own::Item,
                <Own::Slots as PaddedReadSlots>::L0,
                <Own::Slots as PaddedReadSlots>::L1,
                <Own::Slots as PaddedReadSlots>::L2,
                <Own::Slots as PaddedReadSlots>::L3,
                <Own::Slots as PaddedReadSlots>::L4,
                <Own::Slots as PaddedReadSlots>::L5,
                <Own::Slots as PaddedReadSlots>::L6,
                <Own::Slots as PaddedReadSlots>::L7,
                <Own::Slots as PaddedReadSlots>::L8,
                <Own::Slots as PaddedReadSlots>::L9,
                <Own::Slots as PaddedReadSlots>::L10,
                <Own::Slots as PaddedReadSlots>::L11,
                <Own::Slots as PaddedReadSlots>::L12,
                <Other::Slots as PaddedReadSlots>::L0,
                <Other::Slots as PaddedReadSlots>::L1,
                <Other::Slots as PaddedReadSlots>::L2,
                <Other::Slots as PaddedReadSlots>::L3,
                <Other::Slots as PaddedReadSlots>::L4,
                <Other::Slots as PaddedReadSlots>::L5,
                <Other::Slots as PaddedReadSlots>::L6,
                <Other::Slots as PaddedReadSlots>::L7,
                <Other::Slots as PaddedReadSlots>::L8,
                <Other::Slots as PaddedReadSlots>::L9,
                <Other::Slots as PaddedReadSlots>::L10,
                <Other::Slots as PaddedReadSlots>::L11,
                <Other::Slots as PaddedReadSlots>::L12,
                <Own::Item as StorageLayout>::StorageLeaves,
                Own::DeviceExpr,
                Other::DeviceExpr,
                <Own::Item as StorageLayout>::DeviceLayout,
                Less,
                Rule,
                R,
            >(
                exec.client(),
                crate::launch::cube_count_1d(capacity.div_ceil(OCCURRENCE_TILE))?,
                CubeDim::new_1d(OCCURRENCE_SIZE),
                BufferArg::from_raw_parts(own_reads.slots[0].0.clone(), own_reads.slots[0].1),
                BufferArg::from_raw_parts(own_reads.slots[1].0.clone(), own_reads.slots[1].1),
                BufferArg::from_raw_parts(own_reads.slots[2].0.clone(), own_reads.slots[2].1),
                BufferArg::from_raw_parts(own_reads.slots[3].0.clone(), own_reads.slots[3].1),
                BufferArg::from_raw_parts(own_reads.slots[4].0.clone(), own_reads.slots[4].1),
                BufferArg::from_raw_parts(own_reads.slots[5].0.clone(), own_reads.slots[5].1),
                BufferArg::from_raw_parts(own_reads.slots[6].0.clone(), own_reads.slots[6].1),
                BufferArg::from_raw_parts(own_reads.slots[7].0.clone(), own_reads.slots[7].1),
                BufferArg::from_raw_parts(own_reads.slots[8].0.clone(), own_reads.slots[8].1),
                BufferArg::from_raw_parts(own_reads.slots[9].0.clone(), own_reads.slots[9].1),
                BufferArg::from_raw_parts(own_reads.slots[10].0.clone(), own_reads.slots[10].1),
                BufferArg::from_raw_parts(own_reads.slots[11].0.clone(), own_reads.slots[11].1),
                BufferArg::from_raw_parts(own_reads.slots[12].0.clone(), own_reads.slots[12].1),
                BufferArg::from_raw_parts(own_offsets, own_reads.offsets.len()),
                BufferArg::from_raw_parts(other_reads.slots[0].0.clone(), other_reads.slots[0].1),
                BufferArg::from_raw_parts(other_reads.slots[1].0.clone(), other_reads.slots[1].1),
                BufferArg::from_raw_parts(other_reads.slots[2].0.clone(), other_reads.slots[2].1),
                BufferArg::from_raw_parts(other_reads.slots[3].0.clone(), other_reads.slots[3].1),
                BufferArg::from_raw_parts(other_reads.slots[4].0.clone(), other_reads.slots[4].1),
                BufferArg::from_raw_parts(other_reads.slots[5].0.clone(), other_reads.slots[5].1),
                BufferArg::from_raw_parts(other_reads.slots[6].0.clone(), other_reads.slots[6].1),
                BufferArg::from_raw_parts(other_reads.slots[7].0.clone(), other_reads.slots[7].1),
                BufferArg::from_raw_parts(other_reads.slots[8].0.clone(), other_reads.slots[8].1),
                BufferArg::from_raw_parts(other_reads.slots[9].0.clone(), other_reads.slots[9].1),
                BufferArg::from_raw_parts(other_reads.slots[10].0.clone(), other_reads.slots[10].1),
                BufferArg::from_raw_parts(other_reads.slots[11].0.clone(), other_reads.slots[11].1),
                BufferArg::from_raw_parts(other_reads.slots[12].0.clone(), other_reads.slots[12].1),
                BufferArg::from_raw_parts(other_offsets, other_reads.offsets.len()),
                BufferArg::from_raw_parts(own_len.handle.clone(), 1),
                BufferArg::from_raw_parts(other_len.handle.clone(), 1),
                BufferArg::from_raw_parts(flags.handle.clone(), flags.capacity()),
            );
        }
        Ok(flags)
    }
}

/// Runs an ordered multiset operation on two lowered read expressions.
///
/// `mode` is 0 for union, 1 for intersection, and 2 for left difference.
pub(crate) fn set<R, Left, Right, Less, Output>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    _less: Less,
    output: Output,
    mode: u8,
) -> Result<DeviceVec<R, u32>, Error>
where
    R: Runtime,
    Left: crate::core::facade::KernelInput<R>
        + CopySelected<R, Output>
        + crate::merge::MergeDirectInput<R, Right, Output, Less>,
    Right: crate::core::facade::KernelInput<R> + ReadExpression<Item = Left::Item>,
    Left::Item: crate::api::iter::KernelRow,
    Less: crate::op::BinaryPredicateOp<Left::Item>,
    Output:
        crate::core::facade::KernelOutput<R> + crate::output::OutputExpression<Item = Left::Item>,
{
    if mode == 0 {
        let right_extra = right
            .clone()
            .occurrence_flags_direct::<Less, UnionExtra>(exec, left.clone())?;
        let selection = SelectionControl::from_flags(exec, right_extra)?;
        let total_capacity = left
            .logical_len()?
            .checked_add(selection.indices().capacity())
            .ok_or(Error::LengthTooLarge { len: usize::MAX })?;
        let output_extent = crate::extent::LogicalExtent::add(
            exec,
            &left.logical_extent()?,
            &selection.indices().logical_extent(),
            total_capacity,
        )?;
        crate::merge::merge_direct_selected_right(
            exec,
            left,
            right,
            selection.indices(),
            _less,
            output,
        )?;
        return output_extent.materialize(exec);
    }

    let flags = if mode == 1 {
        left.clone()
            .occurrence_flags_direct::<Less, IntersectionKeep>(exec, right.clone())?
    } else {
        left.clone()
            .occurrence_flags_direct::<Less, DifferenceKeep>(exec, right)?
    };
    let control = SelectionControl::from_flags(exec, flags)?;
    left.copy_selected(exec, &control, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::BinaryPredicateOp;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    struct LessU32;

    #[cubecl::cube]
    impl BinaryPredicateOp<u32> for LessU32 {
        fn apply(lhs: u32, rhs: u32) -> crate::MFlag {
            crate::flag::from_bool(lhs < rhs)
        }
    }

    #[test]
    fn ordered_sets_preserve_standard_multiplicity() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let left = exec.to_device(&[1_u32, 2, 2, 2, 4]);
        let right = exec.to_device(&[2_u32, 2, 3, 4, 4]);

        let union = exec.to_device(&[0_u32; 10]);
        let union_len = crate::api::algorithm::set_union_into(
            &exec,
            left.column(),
            right.column(),
            LessU32,
            union.slice_mut(..),
        )
        .unwrap();
        let union_len = union_len.read(&exec).unwrap();
        assert_eq!(
            exec.to_host(&union.slice(..union_len)).unwrap(),
            vec![1, 2, 2, 2, 3, 4, 4]
        );

        let intersection = exec.to_device(&[0_u32; 5]);
        let intersection_len = crate::api::algorithm::set_intersection_into(
            &exec,
            left.column(),
            right.column(),
            LessU32,
            intersection.slice_mut(..),
        )
        .unwrap();
        let intersection_len = intersection_len.read(&exec).unwrap();
        assert_eq!(
            exec.to_host(&intersection.slice(..intersection_len))
                .unwrap(),
            vec![2, 2, 4]
        );

        let difference = exec.to_device(&[0_u32; 5]);
        let difference_len = crate::api::algorithm::set_difference_into(
            &exec,
            left.column(),
            right.column(),
            LessU32,
            difference.slice_mut(..),
        )
        .unwrap();
        let difference_len = difference_len.read(&exec).unwrap();
        assert_eq!(
            exec.to_host(&difference.slice(..difference_len)).unwrap(),
            vec![1, 2]
        );
    }
}
