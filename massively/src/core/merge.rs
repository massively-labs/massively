//! Stable two-range merge control and arity-independent payload application.

use core::marker::PhantomData;

use cubecl::prelude::*;

use crate::{
    A13, DeviceVec, Error, Executor, MStorageElement, ReadExpression, StorageLayout,
    eval::Eval13,
    ordering::BinaryPredicateOp,
    output::{LowerOutputExpression, OutputBindings, OutputExpression, StageOutput},
    read::{Env0, Env13, LowerReadExpression, PaddedReadSlots},
    reduce::{StageRead, StagedBindings},
    storage::{
        Decompose, Recompose, SharedLeaves, SharedLeavesExpand, StorePadded12, StorePadded12Expand,
    },
};

const BLOCK_SIZE: u32 = 256;
const MERGE_SIZE: u32 = 64;
const MERGE_ITEMS: usize = 4;
const MERGE_TILE: usize = MERGE_SIZE as usize * MERGE_ITEMS;

macro_rules! define_merge_control_kernel {
    ($name:ident,$eval:ident,$method:ident; [$( $left_leaf:ident:$left_slot:ident:$right_leaf:ident:$right_slot:ident ),+]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $left_leaf: CubePrimitive, )+
            $( $right_leaf: CubePrimitive, )+
            Left: $eval<Item, $( $left_leaf ),+>,
            Right: $eval<Item, $( $right_leaf ),+>,
            Less: BinaryPredicateOp<Item>,
        >(
            $( $left_slot: &[$left_leaf], )+
            left_offsets: &[u32],
            $( $right_slot: &[$right_leaf], )+
            right_offsets: &[u32],
            left_length: &[u32],
            right_length: &[u32],
            parameters: &[u32],
            permutation: &mut [u32],
        ) {
            let left_len = left_length[0] as usize;
            let right_len = right_length[0] as usize;
            let right_base = parameters[0];
            let total = left_len + right_len;
            let tile_start = (CUBE_POS as usize) * MERGE_TILE;
            if tile_start < total {
                let tile_end = if tile_start + MERGE_TILE < total {
                    tile_start + MERGE_TILE
                } else {
                    total
                };
                let mut partition = Shared::<[u32]>::new_slice(4usize);
                if UNIT_POS == 0u32 {
                    let begin_low_init = if tile_start > right_len {
                        tile_start - right_len
                    } else {
                        0usize
                    };
                    let begin_high_init = if tile_start < left_len {
                        tile_start
                    } else {
                        left_len
                    };
                    let begin_low = RuntimeCell::<usize>::new(begin_low_init);
                    let begin_high = RuntimeCell::<usize>::new(begin_high_init);
                    while begin_low.read() < begin_high.read() {
                        let left_rank = (begin_low.read() + begin_high.read()) / 2usize;
                        let right_rank = tile_start - left_rank;
                        if left_rank < left_len
                            && right_rank > 0usize
                            && !crate::ordering::binary_predicate::<Item, Less>(
                                Right::$method(
                                    $( $right_slot, )+
                                    right_offsets,
                                    right_rank - 1usize,
                                ),
                                Left::$method($( $left_slot, )+ left_offsets, left_rank),
                            )
                        {
                            begin_low.store(left_rank + 1usize);
                        } else {
                            begin_high.store(left_rank);
                        }
                    }

                    let end_low_init = if tile_end > right_len {
                        tile_end - right_len
                    } else {
                        0usize
                    };
                    let end_high_init = if tile_end < left_len {
                        tile_end
                    } else {
                        left_len
                    };
                    let end_low = RuntimeCell::<usize>::new(end_low_init);
                    let end_high = RuntimeCell::<usize>::new(end_high_init);
                    while end_low.read() < end_high.read() {
                        let left_rank = (end_low.read() + end_high.read()) / 2usize;
                        let right_rank = tile_end - left_rank;
                        if left_rank < left_len
                            && right_rank > 0usize
                            && !crate::ordering::binary_predicate::<Item, Less>(
                                Right::$method(
                                    $( $right_slot, )+
                                    right_offsets,
                                    right_rank - 1usize,
                                ),
                                Left::$method($( $left_slot, )+ left_offsets, left_rank),
                            )
                        {
                            end_low.store(left_rank + 1usize);
                        } else {
                            end_high.store(left_rank);
                        }
                    }

                    let left_begin = begin_low.read();
                    let right_begin = tile_start - left_begin;
                    partition[0] = left_begin as u32;
                    partition[1] = right_begin as u32;
                    partition[2] = (end_low.read() - left_begin) as u32;
                    partition[3] = ((tile_end - end_low.read()) - right_begin) as u32;
                }
                sync_cube();

                let left_begin = partition[0] as usize;
                let right_begin = partition[1] as usize;
                let left_count = partition[2] as usize;
                let right_count = partition[3] as usize;
                let tile_len = left_count + right_count;
                let local_start = UNIT_POS as usize * MERGE_ITEMS;
                if local_start < tile_len {
                    let local_end = if local_start + MERGE_ITEMS < tile_len {
                        local_start + MERGE_ITEMS
                    } else {
                        tile_len
                    };
                    let local_low_init = if local_start > right_count {
                        local_start - right_count
                    } else {
                        0usize
                    };
                    let local_high_init = if local_start < left_count {
                        local_start
                    } else {
                        left_count
                    };
                    let local_low = RuntimeCell::<usize>::new(local_low_init);
                    let local_high = RuntimeCell::<usize>::new(local_high_init);
                    while local_low.read() < local_high.read() {
                        let left_rank = (local_low.read() + local_high.read()) / 2usize;
                        let right_rank = local_start - left_rank;
                        if left_rank < left_count
                            && right_rank > 0usize
                            && !crate::ordering::binary_predicate::<Item, Less>(
                                Right::$method(
                                    $( $right_slot, )+
                                    right_offsets,
                                    right_begin + right_rank - 1usize,
                                ),
                                Left::$method(
                                    $( $left_slot, )+
                                    left_offsets,
                                    left_begin + left_rank,
                                ),
                            )
                        {
                            local_low.store(left_rank + 1usize);
                        } else {
                            local_high.store(left_rank);
                        }
                    }

                    let left_rank = RuntimeCell::<usize>::new(local_low.read());
                    let right_rank = RuntimeCell::<usize>::new(local_start - local_low.read());
                    let cursor = RuntimeCell::<usize>::new(local_start);
                    while cursor.read() < local_end {
                        let take_left = left_rank.read() < left_count
                            && (right_rank.read() >= right_count
                                || !crate::ordering::binary_predicate::<Item, Less>(
                                    Right::$method(
                                        $( $right_slot, )+
                                        right_offsets,
                                        right_begin + right_rank.read(),
                                    ),
                                    Left::$method(
                                        $( $left_slot, )+
                                        left_offsets,
                                        left_begin + left_rank.read(),
                                    ),
                                ));
                        let encoded = if take_left {
                            let encoded = (left_begin + left_rank.read()) as u32;
                            left_rank.store(left_rank.read() + 1usize);
                            encoded
                        } else {
                            let encoded = right_base + (right_begin + right_rank.read()) as u32;
                            right_rank.store(right_rank.read() + 1usize);
                            encoded
                        };
                        permutation[tile_start + cursor.read()] = encoded;
                        cursor.store(cursor.read() + 1usize);
                    }
                }
            }
        }
    };
}

define_merge_control_kernel!(merge_control_a13,Eval13,eval13; [LL0:left0:RL0:right0,LL1:left1:RL1:right1,LL2:left2:RL2:right2,LL3:left3:RL3:right3,LL4:left4:RL4:right4,LL5:left5:RL5:right5,LL6:left6:RL6:right6,LL7:left7:RL7:right7,LL8:left8:RL8:right8,LL9:left9:RL9:right9,LL10:left10:RL10:right10,LL11:left11:RL11:right11,LL12:left12:RL12:right12]);

macro_rules! define_merge_direct_kernel {
    ($name:ident,$eval:ident,$method:ident; [$( $left_leaf:ident:$left_slot:ident:$right_leaf:ident:$right_slot:ident ),+]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        #[allow(clippy::too_many_arguments)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $left_leaf: CubePrimitive, )+
            $( $right_leaf: CubePrimitive, )+
            O0: CubePrimitive,
            O1: CubePrimitive,
            O2: CubePrimitive,
            O3: CubePrimitive,
            O4: CubePrimitive,
            O5: CubePrimitive,
            O6: CubePrimitive,
            O7: CubePrimitive,
            O8: CubePrimitive,
            O9: CubePrimitive,
            O10: CubePrimitive,
            O11: CubePrimitive,
            Leaves: CubeType
                + Send
                + Sync
                + 'static
                + SharedLeaves
                + StorePadded12<
                    O0 = O0,
                    O1 = O1,
                    O2 = O2,
                    O3 = O3,
                    O4 = O4,
                    O5 = O5,
                    O6 = O6,
                    O7 = O7,
                    O8 = O8,
                    O9 = O9,
                    O10 = O10,
                    O11 = O11,
                >,
            Left: $eval<Item, $( $left_leaf ),+>,
            Right: $eval<Item, $( $right_leaf ),+>,
            Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
            Less: BinaryPredicateOp<Item>,
        >(
            $( $left_slot: &[$left_leaf], )+
            left_offsets: &[u32],
            $( $right_slot: &[$right_leaf], )+
            right_offsets: &[u32],
            right_positions: &[u32],
            #[comptime] select_right: bool,
            left_length: &[u32],
            right_length: &[u32],
            out0: &mut [O0],
            out1: &mut [O1],
            out2: &mut [O2],
            out3: &mut [O3],
            out4: &mut [O4],
            out5: &mut [O5],
            out6: &mut [O6],
            out7: &mut [O7],
            out8: &mut [O8],
            out9: &mut [O9],
            out10: &mut [O10],
            out11: &mut [O11],
            write_offsets: &[u32],
        ) {
            let left_len = left_length[0] as usize;
            let right_len = right_length[0] as usize;
            let total = left_len + right_len;
            let tile_start = (CUBE_POS as usize) * MERGE_TILE;
            if tile_start < total {
                let tile_end = if tile_start + MERGE_TILE < total {
                    tile_start + MERGE_TILE
                } else {
                    total
                };
                let mut partition = Shared::<[u32]>::new_slice(4usize);
                if UNIT_POS == 0u32 {
                    let begin_low_init = if tile_start > right_len {
                        tile_start - right_len
                    } else {
                        0usize
                    };
                    let begin_high_init = if tile_start < left_len {
                        tile_start
                    } else {
                        left_len
                    };
                    let begin_low = RuntimeCell::<usize>::new(begin_low_init);
                    let begin_high = RuntimeCell::<usize>::new(begin_high_init);
                    while begin_low.read() < begin_high.read() {
                        let left_rank = (begin_low.read() + begin_high.read()) / 2usize;
                        let right_rank = tile_start - left_rank;
                        let right_position = if right_rank > 0usize {
                            right_rank - 1usize
                        } else {
                            0usize
                        };
                        let right_index = if select_right {
                            right_positions[right_position] as usize
                        } else {
                            right_position
                        };
                        if left_rank < left_len
                            && right_rank > 0usize
                            && !crate::ordering::binary_predicate::<Item, Less>(
                                Right::$method(
                                    $( $right_slot, )+
                                    right_offsets,
                                    right_index,
                                ),
                                Left::$method($( $left_slot, )+ left_offsets, left_rank),
                            )
                        {
                            begin_low.store(left_rank + 1usize);
                        } else {
                            begin_high.store(left_rank);
                        }
                    }

                    let end_low_init = if tile_end > right_len {
                        tile_end - right_len
                    } else {
                        0usize
                    };
                    let end_high_init = if tile_end < left_len {
                        tile_end
                    } else {
                        left_len
                    };
                    let end_low = RuntimeCell::<usize>::new(end_low_init);
                    let end_high = RuntimeCell::<usize>::new(end_high_init);
                    while end_low.read() < end_high.read() {
                        let left_rank = (end_low.read() + end_high.read()) / 2usize;
                        let right_rank = tile_end - left_rank;
                        let right_position = if right_rank > 0usize {
                            right_rank - 1usize
                        } else {
                            0usize
                        };
                        let right_index = if select_right {
                            right_positions[right_position] as usize
                        } else {
                            right_position
                        };
                        if left_rank < left_len
                            && right_rank > 0usize
                            && !crate::ordering::binary_predicate::<Item, Less>(
                                Right::$method(
                                    $( $right_slot, )+
                                    right_offsets,
                                    right_index,
                                ),
                                Left::$method($( $left_slot, )+ left_offsets, left_rank),
                            )
                        {
                            end_low.store(left_rank + 1usize);
                        } else {
                            end_high.store(left_rank);
                        }
                    }

                    let left_begin = begin_low.read();
                    let right_begin = tile_start - left_begin;
                    partition[0] = left_begin as u32;
                    partition[1] = right_begin as u32;
                    partition[2] = (end_low.read() - left_begin) as u32;
                    partition[3] = ((tile_end - end_low.read()) - right_begin) as u32;
                }
                sync_cube();

                let left_begin = partition[0] as usize;
                let right_begin = partition[1] as usize;
                let left_count = partition[2] as usize;
                let right_count = partition[3] as usize;
                let tile_len = left_count + right_count;
                let mut shared = Leaves::new_shared(MERGE_TILE);
                let load_position = RuntimeCell::<usize>::new(UNIT_POS as usize);
                while load_position.read() < tile_len {
                    if load_position.read() < left_count {
                        Layout::decompose(Left::$method(
                            $( $left_slot, )+
                            left_offsets,
                            left_begin + load_position.read(),
                        ))
                        .store_shared(&mut shared, load_position.read());
                    } else {
                        let right_index = right_begin + load_position.read() - left_count;
                        let right_index = if select_right {
                            right_positions[right_index] as usize
                        } else {
                            right_index
                        };
                        Layout::decompose(Right::$method(
                            $( $right_slot, )+
                            right_offsets,
                            right_index,
                        ))
                        .store_shared(&mut shared, load_position.read());
                    }
                    load_position.store(load_position.read() + MERGE_SIZE as usize);
                }
                sync_cube();

                let local_start = UNIT_POS as usize * MERGE_ITEMS;
                if local_start < tile_len {
                    let local_end = if local_start + MERGE_ITEMS < tile_len {
                        local_start + MERGE_ITEMS
                    } else {
                        tile_len
                    };
                    let local_low_init = if local_start > right_count {
                        local_start - right_count
                    } else {
                        0usize
                    };
                    let local_high_init = if local_start < left_count {
                        local_start
                    } else {
                        left_count
                    };
                    let local_low = RuntimeCell::<usize>::new(local_low_init);
                    let local_high = RuntimeCell::<usize>::new(local_high_init);
                    while local_low.read() < local_high.read() {
                        let left_rank = (local_low.read() + local_high.read()) / 2usize;
                        let right_rank = local_start - left_rank;
                        if left_rank < left_count
                            && right_rank > 0usize
                            && !crate::ordering::binary_predicate::<Item, Less>(
                                Layout::recompose(Leaves::load_shared(
                                    &shared,
                                    left_count + right_rank - 1usize,
                                )),
                                Layout::recompose(Leaves::load_shared(&shared, left_rank)),
                            )
                        {
                            local_low.store(left_rank + 1usize);
                        } else {
                            local_high.store(left_rank);
                        }
                    }

                    let left_rank = RuntimeCell::<usize>::new(local_low.read());
                    let right_rank = RuntimeCell::<usize>::new(local_start - local_low.read());
                    let cursor = RuntimeCell::<usize>::new(local_start);
                    while cursor.read() < local_end {
                        let take_left = left_rank.read() < left_count
                            && (right_rank.read() >= right_count
                                || !crate::ordering::binary_predicate::<Item, Less>(
                                    Layout::recompose(Leaves::load_shared(
                                        &shared,
                                        left_count + right_rank.read(),
                                    )),
                                    Layout::recompose(Leaves::load_shared(
                                        &shared,
                                        left_rank.read(),
                                    )),
                                ));
                        let source = if take_left {
                            let source = left_rank.read();
                            left_rank.store(source + 1usize);
                            source
                        } else {
                            let source = left_count + right_rank.read();
                            right_rank.store(right_rank.read() + 1usize);
                            source
                        };
                        Leaves::load_shared(&shared, source).store_padded(
                            out0,
                            out1,
                            out2,
                            out3,
                            out4,
                            out5,
                            out6,
                            out7,
                            out8,
                            out9,
                            out10,
                            out11,
                            write_offsets,
                            tile_start + cursor.read(),
                        );
                        cursor.store(cursor.read() + 1usize);
                    }
                }
            }
        }
    };
}

define_merge_direct_kernel!(merge_direct_a13,Eval13,eval13; [LL0:left0:RL0:right0,LL1:left1:RL1:right1,LL2:left2:RL2:right2,LL3:left3:RL3:right3,LL4:left4:RL4:right4,LL5:left5:RL5:right5,LL6:left6:RL6:right6,LL7:left7:RL7:right7,LL8:left8:RL8:right8,LL9:left9:RL9:right9,LL10:left10:RL10:right10,LL11:left11:RL11:right11,LL12:left12:RL12:right12]);

pub(crate) trait MergeDirectInput<R: Runtime, Right, Output, Less>: ReadExpression {
    fn merge_direct(
        &self,
        exec: &Executor<R>,
        right: &Right,
        right_positions: Option<&DeviceVec<R, u32>>,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Left, Right, Output, Less> MergeDirectInput<R, Right, Output, Less> for Left
where
    R: Runtime,
    Left: ReadExpression<Item = Output::Item> + LowerReadExpression + StageRead<R, Env0>,
    Right: ReadExpression<Item = Output::Item> + LowerReadExpression + StageRead<R, Env0>,
    Less: BinaryPredicateOp<Output::Item>,
    <Output::Item as StorageLayout>::StorageLeaves: SharedLeaves + StorePadded12,
    <<Output::Item as StorageLayout>::StorageLeaves as CubeType>::ExpandType: StorePadded12Expand,
    <Output::Item as StorageLayout>::DeviceLayout:
        Recompose<Output::Item, Leaves = <Output::Item as StorageLayout>::StorageLeaves>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots:
        crate::output::PaddedOutputSlots<Leaves = <Output::Item as StorageLayout>::StorageLeaves>,
{
    fn merge_direct(
        &self,
        exec: &Executor<R>,
        right: &Right,
        right_positions: Option<&DeviceVec<R, u32>>,
        output: Output,
    ) -> Result<(), Error> {
        let left_capacity = self.logical_len()?;
        let right_capacity = match right_positions {
            Some(positions) => positions.capacity(),
            None => right.logical_len()?,
        };
        let total_capacity = left_capacity
            .checked_add(right_capacity)
            .ok_or(Error::LengthTooLarge { len: usize::MAX })?;
        let left_extent = self.logical_extent()?;
        let right_extent = match right_positions {
            Some(positions) => positions.logical_extent(),
            None => right.logical_extent()?,
        };
        let required_output = left_extent
            .upper_bound()
            .checked_add(right_extent.upper_bound())
            .ok_or(Error::LengthTooLarge { len: usize::MAX })?;
        let output_len = output.logical_len()?;
        if output_len < required_output {
            return Err(Error::OutputTooShort {
                input: required_output,
                output: output_len,
            });
        }
        if total_capacity == 0 {
            return Ok(());
        }

        let mut left_reads = StagedBindings::new();
        self.stage_at(exec.client(), exec.id(), &mut left_reads)?;
        left_reads.pad_to_thirteen(exec.client());
        let mut right_reads = StagedBindings::new();
        right.stage_at(exec.client(), exec.id(), &mut right_reads)?;
        right_reads.pad_to_thirteen(exec.client());
        let mut writes = OutputBindings::new();
        output.stage_output(exec.id(), &mut writes)?;
        writes.pad_to_twelve(exec.client());

        let left_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&left_reads.offsets));
        let right_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&right_reads.offsets));
        let write_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&writes.offsets));
        let left_length = left_extent.materialize(exec)?;
        let right_length = right_extent.materialize(exec)?;
        let (right_positions_handle, right_positions_len, select_right) = match right_positions {
            Some(positions) => (positions.handle.clone(), positions.capacity(), true),
            None => (right_length.handle.clone(), 1usize, false),
        };

        unsafe {
            merge_direct_a13::launch_unchecked::<
                Output::Item,
                <Left::Slots as PaddedReadSlots>::L0,
                <Left::Slots as PaddedReadSlots>::L1,
                <Left::Slots as PaddedReadSlots>::L2,
                <Left::Slots as PaddedReadSlots>::L3,
                <Left::Slots as PaddedReadSlots>::L4,
                <Left::Slots as PaddedReadSlots>::L5,
                <Left::Slots as PaddedReadSlots>::L6,
                <Left::Slots as PaddedReadSlots>::L7,
                <Left::Slots as PaddedReadSlots>::L8,
                <Left::Slots as PaddedReadSlots>::L9,
                <Left::Slots as PaddedReadSlots>::L10,
                <Left::Slots as PaddedReadSlots>::L11,
                <Left::Slots as PaddedReadSlots>::L12,
                <Right::Slots as PaddedReadSlots>::L0,
                <Right::Slots as PaddedReadSlots>::L1,
                <Right::Slots as PaddedReadSlots>::L2,
                <Right::Slots as PaddedReadSlots>::L3,
                <Right::Slots as PaddedReadSlots>::L4,
                <Right::Slots as PaddedReadSlots>::L5,
                <Right::Slots as PaddedReadSlots>::L6,
                <Right::Slots as PaddedReadSlots>::L7,
                <Right::Slots as PaddedReadSlots>::L8,
                <Right::Slots as PaddedReadSlots>::L9,
                <Right::Slots as PaddedReadSlots>::L10,
                <Right::Slots as PaddedReadSlots>::L11,
                <Right::Slots as PaddedReadSlots>::L12,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O0,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O1,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O2,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O3,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O4,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O5,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O6,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O7,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O8,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O9,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O10,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O11,
                <Output::Item as StorageLayout>::StorageLeaves,
                Left::DeviceExpr,
                Right::DeviceExpr,
                <Output::Item as StorageLayout>::DeviceLayout,
                Less,
                R,
            >(
                exec.client(),
                crate::launch::cube_count_1d(total_capacity.div_ceil(MERGE_TILE))?,
                CubeDim::new_1d(MERGE_SIZE),
                BufferArg::from_raw_parts(left_reads.slots[0].0.clone(), left_reads.slots[0].1),
                BufferArg::from_raw_parts(left_reads.slots[1].0.clone(), left_reads.slots[1].1),
                BufferArg::from_raw_parts(left_reads.slots[2].0.clone(), left_reads.slots[2].1),
                BufferArg::from_raw_parts(left_reads.slots[3].0.clone(), left_reads.slots[3].1),
                BufferArg::from_raw_parts(left_reads.slots[4].0.clone(), left_reads.slots[4].1),
                BufferArg::from_raw_parts(left_reads.slots[5].0.clone(), left_reads.slots[5].1),
                BufferArg::from_raw_parts(left_reads.slots[6].0.clone(), left_reads.slots[6].1),
                BufferArg::from_raw_parts(left_reads.slots[7].0.clone(), left_reads.slots[7].1),
                BufferArg::from_raw_parts(left_reads.slots[8].0.clone(), left_reads.slots[8].1),
                BufferArg::from_raw_parts(left_reads.slots[9].0.clone(), left_reads.slots[9].1),
                BufferArg::from_raw_parts(left_reads.slots[10].0.clone(), left_reads.slots[10].1),
                BufferArg::from_raw_parts(left_reads.slots[11].0.clone(), left_reads.slots[11].1),
                BufferArg::from_raw_parts(left_reads.slots[12].0.clone(), left_reads.slots[12].1),
                BufferArg::from_raw_parts(left_offsets, left_reads.offsets.len()),
                BufferArg::from_raw_parts(right_reads.slots[0].0.clone(), right_reads.slots[0].1),
                BufferArg::from_raw_parts(right_reads.slots[1].0.clone(), right_reads.slots[1].1),
                BufferArg::from_raw_parts(right_reads.slots[2].0.clone(), right_reads.slots[2].1),
                BufferArg::from_raw_parts(right_reads.slots[3].0.clone(), right_reads.slots[3].1),
                BufferArg::from_raw_parts(right_reads.slots[4].0.clone(), right_reads.slots[4].1),
                BufferArg::from_raw_parts(right_reads.slots[5].0.clone(), right_reads.slots[5].1),
                BufferArg::from_raw_parts(right_reads.slots[6].0.clone(), right_reads.slots[6].1),
                BufferArg::from_raw_parts(right_reads.slots[7].0.clone(), right_reads.slots[7].1),
                BufferArg::from_raw_parts(right_reads.slots[8].0.clone(), right_reads.slots[8].1),
                BufferArg::from_raw_parts(right_reads.slots[9].0.clone(), right_reads.slots[9].1),
                BufferArg::from_raw_parts(right_reads.slots[10].0.clone(), right_reads.slots[10].1),
                BufferArg::from_raw_parts(right_reads.slots[11].0.clone(), right_reads.slots[11].1),
                BufferArg::from_raw_parts(right_reads.slots[12].0.clone(), right_reads.slots[12].1),
                BufferArg::from_raw_parts(right_offsets, right_reads.offsets.len()),
                BufferArg::from_raw_parts(right_positions_handle, right_positions_len),
                select_right,
                BufferArg::from_raw_parts(left_length.handle.clone(), 1),
                BufferArg::from_raw_parts(right_length.handle.clone(), 1),
                BufferArg::from_raw_parts(writes.slots[0].0.clone(), writes.slots[0].1),
                BufferArg::from_raw_parts(writes.slots[1].0.clone(), writes.slots[1].1),
                BufferArg::from_raw_parts(writes.slots[2].0.clone(), writes.slots[2].1),
                BufferArg::from_raw_parts(writes.slots[3].0.clone(), writes.slots[3].1),
                BufferArg::from_raw_parts(writes.slots[4].0.clone(), writes.slots[4].1),
                BufferArg::from_raw_parts(writes.slots[5].0.clone(), writes.slots[5].1),
                BufferArg::from_raw_parts(writes.slots[6].0.clone(), writes.slots[6].1),
                BufferArg::from_raw_parts(writes.slots[7].0.clone(), writes.slots[7].1),
                BufferArg::from_raw_parts(writes.slots[8].0.clone(), writes.slots[8].1),
                BufferArg::from_raw_parts(writes.slots[9].0.clone(), writes.slots[9].1),
                BufferArg::from_raw_parts(writes.slots[10].0.clone(), writes.slots[10].1),
                BufferArg::from_raw_parts(writes.slots[11].0.clone(), writes.slots[11].1),
                BufferArg::from_raw_parts(write_offsets, writes.offsets.len()),
            );
        }
        Ok(())
    }
}

pub(crate) fn merge_direct<R, Left, Right, Less, Output>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    _less: Less,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Left: MergeDirectInput<R, Right, Output, Less>,
{
    left.merge_direct(exec, &right, None, output)
}

pub(crate) fn merge_direct_selected_right<R, Left, Right, Less, Output>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    right_positions: &DeviceVec<R, u32>,
    _less: Less,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Left: MergeDirectInput<R, Right, Output, Less>,
{
    left.merge_direct(exec, &right, Some(right_positions), output)
}

#[cubecl::cube(launch_unchecked, explicit_define)]
fn merge_apply_a13<
    Item: CubeType + Send + Sync + 'static,
    L0: CubePrimitive,
    L1: CubePrimitive,
    L2: CubePrimitive,
    L3: CubePrimitive,
    L4: CubePrimitive,
    L5: CubePrimitive,
    L6: CubePrimitive,
    L7: CubePrimitive,
    L8: CubePrimitive,
    L9: CubePrimitive,
    L10: CubePrimitive,
    L11: CubePrimitive,
    L12: CubePrimitive,
    R0: CubePrimitive,
    R1: CubePrimitive,
    R2: CubePrimitive,
    R3: CubePrimitive,
    R4: CubePrimitive,
    R5: CubePrimitive,
    R6: CubePrimitive,
    R7: CubePrimitive,
    R8: CubePrimitive,
    R9: CubePrimitive,
    R10: CubePrimitive,
    R11: CubePrimitive,
    R12: CubePrimitive,
    O0: CubePrimitive,
    O1: CubePrimitive,
    O2: CubePrimitive,
    O3: CubePrimitive,
    O4: CubePrimitive,
    O5: CubePrimitive,
    O6: CubePrimitive,
    O7: CubePrimitive,
    O8: CubePrimitive,
    O9: CubePrimitive,
    O10: CubePrimitive,
    O11: CubePrimitive,
    Leaves: CubeType
        + Send
        + Sync
        + 'static
        + StorePadded12<
            O0 = O0,
            O1 = O1,
            O2 = O2,
            O3 = O3,
            O4 = O4,
            O5 = O5,
            O6 = O6,
            O7 = O7,
            O8 = O8,
            O9 = O9,
            O10 = O10,
            O11 = O11,
        >,
    LeftExpr: Eval13<Item, L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>,
    RightExpr: Eval13<Item, R0, R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11, R12>,
    Layout: Decompose<Item, Leaves = Leaves>,
>(
    left0: &[L0],
    left1: &[L1],
    left2: &[L2],
    left3: &[L3],
    left4: &[L4],
    left5: &[L5],
    left6: &[L6],
    left7: &[L7],
    left8: &[L8],
    left9: &[L9],
    left10: &[L10],
    left11: &[L11],
    left12: &[L12],
    left_offsets: &[u32],
    right0: &[R0],
    right1: &[R1],
    right2: &[R2],
    right3: &[R3],
    right4: &[R4],
    right5: &[R5],
    right6: &[R6],
    right7: &[R7],
    right8: &[R8],
    right9: &[R9],
    right10: &[R10],
    right11: &[R11],
    right12: &[R12],
    right_offsets: &[u32],
    permutation: &[u32],
    right_base: &[u32],
    active_len: &[u32],
    out0: &mut [O0],
    out1: &mut [O1],
    out2: &mut [O2],
    out3: &mut [O3],
    out4: &mut [O4],
    out5: &mut [O5],
    out6: &mut [O6],
    out7: &mut [O7],
    out8: &mut [O8],
    out9: &mut [O9],
    out10: &mut [O10],
    out11: &mut [O11],
    write_offsets: &[u32],
) {
    let output_position = ABSOLUTE_POS as usize;
    if output_position < active_len[0] as usize {
        let encoded = permutation[output_position];
        if encoded < right_base[0] {
            Layout::decompose(LeftExpr::eval13(
                left0,
                left1,
                left2,
                left3,
                left4,
                left5,
                left6,
                left7,
                left8,
                left9,
                left10,
                left11,
                left12,
                left_offsets,
                encoded as usize,
            ))
            .store_padded(
                out0,
                out1,
                out2,
                out3,
                out4,
                out5,
                out6,
                out7,
                out8,
                out9,
                out10,
                out11,
                write_offsets,
                output_position,
            );
        } else {
            Layout::decompose(RightExpr::eval13(
                right0,
                right1,
                right2,
                right3,
                right4,
                right5,
                right6,
                right7,
                right8,
                right9,
                right10,
                right11,
                right12,
                right_offsets,
                (encoded - right_base[0]) as usize,
            ))
            .store_padded(
                out0,
                out1,
                out2,
                out3,
                out4,
                out5,
                out6,
                out7,
                out8,
                out9,
                out10,
                out11,
                write_offsets,
                output_position,
            );
        }
    }
}

pub(crate) trait MergeApplyInput<R: Runtime, Right, Output>: ReadExpression {
    fn merge_apply(
        &self,
        exec: &Executor<R>,
        right: &Right,
        control: &MergeControl<R>,
        output: Output,
    ) -> Result<(), Error>;
}

impl<R, Left, Right, Output> MergeApplyInput<R, Right, Output> for Left
where
    R: Runtime,
    Left: ReadExpression<Item = Output::Item> + LowerReadExpression + StageRead<R, Env0>,
    Right: ReadExpression<Item = Output::Item> + LowerReadExpression + StageRead<R, Env0>,
    <Output::Item as StorageLayout>::StorageLeaves: StorePadded12,
    <<Output::Item as StorageLayout>::StorageLeaves as CubeType>::ExpandType: StorePadded12Expand,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots:
        crate::output::PaddedOutputSlots<Leaves = <Output::Item as StorageLayout>::StorageLeaves>,
{
    fn merge_apply(
        &self,
        exec: &Executor<R>,
        right: &Right,
        control: &MergeControl<R>,
        output: Output,
    ) -> Result<(), Error> {
        let operation_len = control.permutation.capacity();
        let active_extent = control.permutation.logical_extent();
        if output.logical_len()? < active_extent.upper_bound() {
            return Err(Error::OutputTooShort {
                input: active_extent.upper_bound(),
                output: output.logical_len()?,
            });
        }
        if operation_len == 0 {
            return Ok(());
        }

        let mut left_reads = StagedBindings::new();
        self.stage_at(exec.client(), exec.id(), &mut left_reads)?;
        left_reads.pad_to_thirteen(exec.client());
        let mut right_reads = StagedBindings::new();
        right.stage_at(exec.client(), exec.id(), &mut right_reads)?;
        right_reads.pad_to_thirteen(exec.client());
        let mut writes = OutputBindings::new();
        output.stage_output(exec.id(), &mut writes)?;
        writes.pad_to_twelve(exec.client());

        let left_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&left_reads.offsets));
        let right_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&right_reads.offsets));
        let write_offsets = exec
            .client()
            .create_from_slice(u32::as_bytes(&writes.offsets));
        let right_base =
            u32::try_from(control.left_capacity).map_err(|_| Error::LengthTooLarge {
                len: control.left_capacity,
            })?;
        let right_base = exec
            .client()
            .create_from_slice(u32::as_bytes(&[right_base]));
        let active_len = active_extent.materialize(exec)?;

        unsafe {
            merge_apply_a13::launch_unchecked::<
                Output::Item,
                <Left::Slots as PaddedReadSlots>::L0,
                <Left::Slots as PaddedReadSlots>::L1,
                <Left::Slots as PaddedReadSlots>::L2,
                <Left::Slots as PaddedReadSlots>::L3,
                <Left::Slots as PaddedReadSlots>::L4,
                <Left::Slots as PaddedReadSlots>::L5,
                <Left::Slots as PaddedReadSlots>::L6,
                <Left::Slots as PaddedReadSlots>::L7,
                <Left::Slots as PaddedReadSlots>::L8,
                <Left::Slots as PaddedReadSlots>::L9,
                <Left::Slots as PaddedReadSlots>::L10,
                <Left::Slots as PaddedReadSlots>::L11,
                <Left::Slots as PaddedReadSlots>::L12,
                <Right::Slots as PaddedReadSlots>::L0,
                <Right::Slots as PaddedReadSlots>::L1,
                <Right::Slots as PaddedReadSlots>::L2,
                <Right::Slots as PaddedReadSlots>::L3,
                <Right::Slots as PaddedReadSlots>::L4,
                <Right::Slots as PaddedReadSlots>::L5,
                <Right::Slots as PaddedReadSlots>::L6,
                <Right::Slots as PaddedReadSlots>::L7,
                <Right::Slots as PaddedReadSlots>::L8,
                <Right::Slots as PaddedReadSlots>::L9,
                <Right::Slots as PaddedReadSlots>::L10,
                <Right::Slots as PaddedReadSlots>::L11,
                <Right::Slots as PaddedReadSlots>::L12,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O0,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O1,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O2,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O3,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O4,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O5,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O6,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O7,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O8,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O9,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O10,
                <<Output::Item as StorageLayout>::StorageLeaves as StorePadded12>::O11,
                <Output::Item as StorageLayout>::StorageLeaves,
                Left::DeviceExpr,
                Right::DeviceExpr,
                <Output::Item as StorageLayout>::DeviceLayout,
                R,
            >(
                exec.client(),
                crate::launch::cube_count_1d(operation_len.div_ceil(BLOCK_SIZE as usize))?,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(left_reads.slots[0].0.clone(), left_reads.slots[0].1),
                BufferArg::from_raw_parts(left_reads.slots[1].0.clone(), left_reads.slots[1].1),
                BufferArg::from_raw_parts(left_reads.slots[2].0.clone(), left_reads.slots[2].1),
                BufferArg::from_raw_parts(left_reads.slots[3].0.clone(), left_reads.slots[3].1),
                BufferArg::from_raw_parts(left_reads.slots[4].0.clone(), left_reads.slots[4].1),
                BufferArg::from_raw_parts(left_reads.slots[5].0.clone(), left_reads.slots[5].1),
                BufferArg::from_raw_parts(left_reads.slots[6].0.clone(), left_reads.slots[6].1),
                BufferArg::from_raw_parts(left_reads.slots[7].0.clone(), left_reads.slots[7].1),
                BufferArg::from_raw_parts(left_reads.slots[8].0.clone(), left_reads.slots[8].1),
                BufferArg::from_raw_parts(left_reads.slots[9].0.clone(), left_reads.slots[9].1),
                BufferArg::from_raw_parts(left_reads.slots[10].0.clone(), left_reads.slots[10].1),
                BufferArg::from_raw_parts(left_reads.slots[11].0.clone(), left_reads.slots[11].1),
                BufferArg::from_raw_parts(left_reads.slots[12].0.clone(), left_reads.slots[12].1),
                BufferArg::from_raw_parts(left_offsets, left_reads.offsets.len()),
                BufferArg::from_raw_parts(right_reads.slots[0].0.clone(), right_reads.slots[0].1),
                BufferArg::from_raw_parts(right_reads.slots[1].0.clone(), right_reads.slots[1].1),
                BufferArg::from_raw_parts(right_reads.slots[2].0.clone(), right_reads.slots[2].1),
                BufferArg::from_raw_parts(right_reads.slots[3].0.clone(), right_reads.slots[3].1),
                BufferArg::from_raw_parts(right_reads.slots[4].0.clone(), right_reads.slots[4].1),
                BufferArg::from_raw_parts(right_reads.slots[5].0.clone(), right_reads.slots[5].1),
                BufferArg::from_raw_parts(right_reads.slots[6].0.clone(), right_reads.slots[6].1),
                BufferArg::from_raw_parts(right_reads.slots[7].0.clone(), right_reads.slots[7].1),
                BufferArg::from_raw_parts(right_reads.slots[8].0.clone(), right_reads.slots[8].1),
                BufferArg::from_raw_parts(right_reads.slots[9].0.clone(), right_reads.slots[9].1),
                BufferArg::from_raw_parts(right_reads.slots[10].0.clone(), right_reads.slots[10].1),
                BufferArg::from_raw_parts(right_reads.slots[11].0.clone(), right_reads.slots[11].1),
                BufferArg::from_raw_parts(right_reads.slots[12].0.clone(), right_reads.slots[12].1),
                BufferArg::from_raw_parts(right_offsets, right_reads.offsets.len()),
                BufferArg::from_raw_parts(
                    control.permutation.handle.clone(),
                    control.permutation.capacity(),
                ),
                BufferArg::from_raw_parts(right_base, 1),
                BufferArg::from_raw_parts(active_len.handle.clone(), 1),
                BufferArg::from_raw_parts(writes.slots[0].0.clone(), writes.slots[0].1),
                BufferArg::from_raw_parts(writes.slots[1].0.clone(), writes.slots[1].1),
                BufferArg::from_raw_parts(writes.slots[2].0.clone(), writes.slots[2].1),
                BufferArg::from_raw_parts(writes.slots[3].0.clone(), writes.slots[3].1),
                BufferArg::from_raw_parts(writes.slots[4].0.clone(), writes.slots[4].1),
                BufferArg::from_raw_parts(writes.slots[5].0.clone(), writes.slots[5].1),
                BufferArg::from_raw_parts(writes.slots[6].0.clone(), writes.slots[6].1),
                BufferArg::from_raw_parts(writes.slots[7].0.clone(), writes.slots[7].1),
                BufferArg::from_raw_parts(writes.slots[8].0.clone(), writes.slots[8].1),
                BufferArg::from_raw_parts(writes.slots[9].0.clone(), writes.slots[9].1),
                BufferArg::from_raw_parts(writes.slots[10].0.clone(), writes.slots[10].1),
                BufferArg::from_raw_parts(writes.slots[11].0.clone(), writes.slots[11].1),
                BufferArg::from_raw_parts(write_offsets, writes.offsets.len()),
            );
        }
        Ok(())
    }
}

pub(crate) struct MergeDispatch<Storage>(PhantomData<fn() -> Storage>);

pub(crate) trait MergeControlDispatch<R, Left, Right, Item, LeftSlots, RightSlots, Less>
where
    R: Runtime,
{
    fn run(exec: &Executor<R>, left: &Left, right: &Right) -> Result<MergeControl<R>, Error>;
}

macro_rules! impl_merge_control_dispatch {
    ($storage:ty,$arity:ty,$eval:ident,$kernel:ident; [$( $left_leaf:ident:$left_index:literal:$right_leaf:ident:$right_index:literal ),+]) => {
        impl<R, Left, Right, Item, Less, $( $left_leaf, )+ $( $right_leaf ),+>
            MergeControlDispatch<
                R,
                Left,
                Right,
                Item,
                Env13<$( $left_leaf ),+>,
                Env13<$( $right_leaf ),+>,
                Less,
            >
            for MergeDispatch<$storage>
        where
            R: Runtime,
            Item: CubeType + Send + Sync + 'static,
            Less: BinaryPredicateOp<Item>,
            $( $left_leaf: MStorageElement, )+
            $( $right_leaf: MStorageElement, )+
            Left: ReadExpression<Item = Item, ReadArity = $arity>
                + LowerReadExpression<Slots = Env13<$( $left_leaf ),+>>
                + StageRead<R, Env0>,
            Right: ReadExpression<Item = Item, ReadArity = $arity>
                + LowerReadExpression<Slots = Env13<$( $right_leaf ),+>>
                + StageRead<R, Env0>,
            Left::DeviceExpr: $eval<Item, $( $left_leaf ),+>,
            Right::DeviceExpr: $eval<Item, $( $right_leaf ),+>,
        {
            fn run(exec: &Executor<R>, left: &Left, right: &Right) -> Result<MergeControl<R>, Error> {
                let left_capacity = left.logical_len()?;
                let right_capacity = right.logical_len()?;
                let total_capacity = left_capacity.checked_add(right_capacity).ok_or(Error::LengthTooLarge { len: usize::MAX })?;
                let left_extent = left.logical_extent()?;
                let right_extent = right.logical_extent()?;
                let total_extent = crate::extent::LogicalExtent::add(
                    exec,
                    &left_extent,
                    &right_extent,
                    total_capacity,
                )?;
                let mut permutation = exec.alloc_row::<u32>(total_capacity);
                permutation.set_logical_extent(total_extent);
                if total_capacity == 0 {
                    return Ok(MergeControl {
                        permutation,
                        left_capacity,
                        right_capacity,
                        left_extent,
                        right_extent,
                    });
                }
                let mut left_bindings = StagedBindings::new();
                let mut right_bindings = StagedBindings::new();
                left.stage_at(exec.client(), exec.id(), &mut left_bindings)?;
                right.stage_at(exec.client(), exec.id(), &mut right_bindings)?;
                let left_offsets = exec.client().create_from_slice(u32::as_bytes(&left_bindings.offsets));
                let right_offsets = exec.client().create_from_slice(u32::as_bytes(&right_bindings.offsets));
                let left_length = left_extent.materialize(exec)?;
                let right_length = right_extent.materialize(exec)?;
                let right_base = u32::try_from(left_capacity)
                    .map_err(|_| Error::LengthTooLarge { len: left_capacity })?;
                let parameters = exec.client().create_from_slice(u32::as_bytes(&[right_base]));
                unsafe {
                    $kernel::launch_unchecked::<Item, $( $left_leaf, )+ $( $right_leaf, )+ Left::DeviceExpr, Right::DeviceExpr, Less, R>(
                        exec.client(),
                        crate::launch::cube_count_1d(total_capacity.div_ceil(MERGE_TILE))?,
                        CubeDim::new_1d(MERGE_SIZE),
                        $( BufferArg::from_raw_parts(left_bindings.slots[$left_index].0.clone(), left_bindings.slots[$left_index].1), )+
                        BufferArg::from_raw_parts(left_offsets, left_bindings.offsets.len()),
                        $( BufferArg::from_raw_parts(right_bindings.slots[$right_index].0.clone(), right_bindings.slots[$right_index].1), )+
                        BufferArg::from_raw_parts(right_offsets, right_bindings.offsets.len()),
                        BufferArg::from_raw_parts(left_length.handle.clone(), 1),
                        BufferArg::from_raw_parts(right_length.handle.clone(), 1),
                        BufferArg::from_raw_parts(parameters, 1),
                        BufferArg::from_raw_parts(
                            permutation.handle.clone(),
                            permutation.capacity(),
                        ),
                    );
                }
                Ok(MergeControl {
                    permutation,
                    left_capacity,
                    right_capacity,
                    left_extent,
                    right_extent,
                })
            }
        }
    };
}

impl_merge_control_dispatch!(crate::S12,A13,Eval13,merge_control_a13; [LL0:0:RL0:0,LL1:1:RL1:1,LL2:2:RL2:2,LL3:3:RL3:3,LL4:4:RL4:4,LL5:5:RL5:5,LL6:6:RL6:6,LL7:7:RL7:7,LL8:8:RL8:8,LL9:9:RL9:9,LL10:10:RL10:10,LL11:11:RL11:11,LL12:12:RL12:12]);

/// Stable merge permutation over a conceptual `left || right` payload.
#[doc(hidden)]
pub struct MergeControl<R: Runtime> {
    pub(crate) permutation: DeviceVec<R, u32>,
    pub(crate) left_capacity: usize,
    pub(crate) right_capacity: usize,
    pub(crate) left_extent: crate::extent::LogicalExtent,
    pub(crate) right_extent: crate::extent::LogicalExtent,
}

pub(crate) fn merge_control_fixed<R, Left, Right, Less>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    _less: Less,
) -> Result<MergeControl<R>, Error>
where
    R: Runtime,
    Left: ReadExpression<ReadArity = A13> + LowerReadExpression + StageRead<R, Env0>,
    Right: ReadExpression<Item = Left::Item, ReadArity = A13>
        + LowerReadExpression
        + StageRead<R, Env0>,
    MergeDispatch<crate::S12>:
        MergeControlDispatch<R, Left, Right, Left::Item, Left::Slots, Right::Slots, Less>,
{
    <MergeDispatch<crate::S12> as MergeControlDispatch<
        R,
        Left,
        Right,
        Left::Item,
        Left::Slots,
        Right::Slots,
        Less,
    >>::run(exec, &left, &right)
}

/// Applies a merge permutation directly to two fixed-ABI read expressions.
///
/// Key and payload dispatch remain independent, while lazy payloads avoid an
/// otherwise unnecessary pair of normalization copies.
pub(crate) fn apply_fixed<R, Left, Right, Output>(
    exec: &Executor<R>,
    left: Left,
    right: Right,
    control: &MergeControl<R>,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Left: MergeApplyInput<R, Right, Output> + StageRead<R, Env0>,
    Right: ReadExpression<Item = Left::Item> + StageRead<R, Env0>,
{
    let left_capacity = left.logical_len()?;
    let right_capacity = right.logical_len()?;
    if left_capacity != control.left_capacity || right_capacity != control.right_capacity {
        return Err(Error::LengthMismatch {
            left: left_capacity + right_capacity,
            right: control.left_capacity + control.right_capacity,
        });
    }
    left.logical_extent()?.zipped(&control.left_extent)?;
    right.logical_extent()?.zipped(&control.right_extent)?;
    left.merge_apply(exec, &right, control, output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Zip;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    struct LessU32;

    #[cubecl::cube]
    impl BinaryPredicateOp<u32> for LessU32 {
        fn apply(lhs: u32, rhs: u32) -> crate::MFlag {
            crate::flag::from_bool(lhs < rhs)
        }
    }

    #[test]
    fn merge_is_stable_and_payloads_reuse_one_control() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let left = exec.to_device(&[1_u32, 2, 2, 5]);
        let right = exec.to_device(&[2_u32, 3, 4]);
        let output = exec.to_device(&[0_u32; 7]);
        merge_direct(
            &exec,
            crate::api::iter::lower_fixed::<WgpuRuntime, _>(left.column()),
            crate::api::iter::lower_fixed::<WgpuRuntime, _>(right.column()),
            LessU32,
            output.slice_mut(..),
        )
        .unwrap();
        assert_eq!(exec.to_host(&output).unwrap(), vec![1, 2, 2, 2, 3, 4, 5]);

        let left_values = exec.to_device(&[10_u32, 20, 21, 50]);
        let right_values = exec.to_device(&[200_u32, 300, 400]);
        let out_keys = exec.to_device(&[0_u32; 7]);
        let out_values = exec.to_device(&[0_u32; 7]);
        let control = merge_control_fixed(
            &exec,
            crate::api::iter::lower_fixed::<WgpuRuntime, _>(left.column()),
            crate::api::iter::lower_fixed::<WgpuRuntime, _>(right.column()),
            LessU32,
        )
        .unwrap();
        apply_fixed(
            &exec,
            crate::api::iter::lower_fixed::<WgpuRuntime, _>(left.column()),
            crate::api::iter::lower_fixed::<WgpuRuntime, _>(right.column()),
            &control,
            out_keys.slice_mut(..),
        )
        .unwrap();
        apply_fixed(
            &exec,
            crate::api::iter::lower_fixed::<WgpuRuntime, _>(left_values.column()),
            crate::api::iter::lower_fixed::<WgpuRuntime, _>(right_values.column()),
            &control,
            out_values.slice_mut(..),
        )
        .unwrap();
        assert_eq!(exec.to_host(&out_keys).unwrap(), vec![1, 2, 2, 2, 3, 4, 5]);
        assert_eq!(
            exec.to_host(&out_values).unwrap(),
            vec![10, 20, 21, 200, 300, 400, 50]
        );

        // Keep a binary output in the monomorphization surface as well.
        let pair_out = Zip::new(
            exec.to_device(&[0_u32; 7]).slice_mut(..),
            exec.to_device(&[0_u32; 7]).slice_mut(..),
        );
        let pair_left = Zip::new(left.column(), left_values.column());
        let pair_right = Zip::new(right.column(), right_values.column());
        struct LessPair;
        #[cubecl::cube]
        impl BinaryPredicateOp<(u32, u32)> for LessPair {
            fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> crate::MFlag {
                crate::flag::from_bool(lhs.0 < rhs.0)
            }
        }
        merge_direct(
            &exec,
            crate::api::iter::lower_fixed::<WgpuRuntime, _>(pair_left),
            crate::api::iter::lower_fixed::<WgpuRuntime, _>(pair_right),
            LessPair,
            pair_out,
        )
        .unwrap();
    }
}
