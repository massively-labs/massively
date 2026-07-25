//! Generic stable comparison sort over flat-row SoA storage.
//!
//! The first kernel creates stable block-local runs. Later kernels merge
//! those runs using sequential reads and writes.

use cubecl::prelude::*;

use crate::{
    Dispatch, Error, Executor, MStorageElement, ReadExpression, RowAlloc, RowStorage,
    StorageLayout,
    eval::{Eval1, Eval2, Eval3, Eval4, Eval5, Eval6, Eval7, Eval8, Eval9, Eval10, Eval11, Eval12},
    launch::cube_count_1d,
    output::{LowerOutputExpression, OutputBindings, OutputExpression, StageOutput},
    read::{
        Env0, Env1, Env2, Env3, Env4, Env5, Env6, Env7, Env8, Env9, Env10, Env11, Env12,
        LowerReadExpression,
    },
    reduce::{StageRead, StagedBindings},
    storage::{
        Decompose, Last, LoadLeaves1, LoadLeaves2, LoadLeaves3, LoadLeaves4, LoadLeaves5,
        LoadLeaves6, LoadLeaves7, LoadLeaves8, LoadLeaves9, LoadLeaves10, LoadLeaves11,
        LoadLeaves12, More, Recompose, StoreLeaves1, StoreLeaves1Expand, StoreLeaves2,
        StoreLeaves2Expand, StoreLeaves3, StoreLeaves3Expand, StoreLeaves4, StoreLeaves4Expand,
        StoreLeaves5, StoreLeaves5Expand, StoreLeaves6, StoreLeaves6Expand, StoreLeaves7,
        StoreLeaves7Expand, StoreLeaves8, StoreLeaves8Expand, StoreLeaves9, StoreLeaves9Expand,
        StoreLeaves10, StoreLeaves10Expand, StoreLeaves11, StoreLeaves11Expand, StoreLeaves12,
        StoreLeaves12Expand,
    },
};

use super::BinaryPredicateOp;

/// One workgroup creates one initial sorted run of this many items.
pub(crate) const BLOCK_ITEMS: usize = 128;
const BLOCK_SIZE: u32 = BLOCK_ITEMS as u32;

/// Storage-shape dispatch for the optimized sort.
///
/// This capability lives on the physical leaf list, whose type identifies one
/// exact storage width.  Semantic callers therefore keep a universal API while
/// the generated kernel contains only the leaves that hold the item.
pub(crate) trait SortLeaves<R: Runtime, Item>: Sized
where
    Item: RowAlloc<R, StorageLeaves = Self>,
{
    fn sort_storage<Less>(
        exec: &Executor<R>,
        input: Item::RowStorage,
    ) -> Result<Item::RowStorage, Error>
    where
        Less: BinaryPredicateOp<Item>;
}

macro_rules! define_sort_kernels {
    (
        $block:ident, $merge:ident, $eval:ident, $method:ident;
        [$( $leaf:ident : $slot:ident : $out:ident : $shared_a:ident : $shared_b:ident ),+];
        $leaves:ty, $load:ident, $store:ident;
        $merge_items:expr
    ) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $block<
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive, )+
            Leaves: CubeType + Send + Sync + 'static
                + $load<$( $leaf ),+>
                + $store<$( $leaf ),+>,
            Expr: $eval<Item, $( $leaf ),+>,
            Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
            Less: BinaryPredicateOp<Item>,
        >(
            $( $slot: &[$leaf], )+
            read_offsets: &[u32],
            logical_len_buffer: &[u32],
            zero_offsets: &[u32],
            $( $out: &mut [$leaf], )+
            write_offsets: &[u32],
        ) {
            let local = UNIT_POS as usize;
            let cube_dim = BLOCK_SIZE as usize;
            let global = (CUBE_POS as usize) * cube_dim + local;
            let tile_start = (CUBE_POS as usize) * cube_dim;
            let logical_len = logical_len_buffer[0] as usize;
            let tile_len = if logical_len > tile_start {
                let remaining = logical_len - tile_start;
                if remaining < cube_dim { remaining } else { cube_dim }
            } else {
                0usize
            };

            $( let mut $shared_a = Shared::<[$leaf]>::new_slice(cube_dim); )+
            $( let mut $shared_b = Shared::<[$leaf]>::new_slice(cube_dim); )+

            if local < tile_len {
                Layout::decompose(Expr::$method($( $slot, )+ read_offsets, global)).store(
                    $( &mut $shared_a, )+ zero_offsets, local,
                );
            }
            sync_cube();

            let width = RuntimeCell::<usize>::new(1usize);
            let source_a = RuntimeCell::<u32>::new(1u32);
            while width.read() < cube_dim {
                if local < tile_len {
                    let pair_width = width.read() * 2usize;
                    let base = (local / pair_width) * pair_width;
                    let left_remaining = tile_len - base;
                    let left_len = if left_remaining < width.read() {
                        left_remaining
                    } else {
                        width.read()
                    };
                    let right_start = base + left_len;
                    let right_remaining = tile_len - right_start;
                    let right_len = if right_remaining < width.read() {
                        right_remaining
                    } else {
                        width.read()
                    };
                    let rank = local - base;
                    let low_init = if rank > right_len { rank - right_len } else { 0usize };
                    let high_init = if rank < left_len { rank } else { left_len };
                    let low = RuntimeCell::<usize>::new(low_init);
                    let high = RuntimeCell::<usize>::new(high_init);

                    if source_a.read() != 0u32 {
                        while low.read() < high.read() {
                            let left_rank = (low.read() + high.read()) / 2usize;
                            let right_rank = rank - left_rank;
                            if left_rank < left_len
                                && right_rank > 0usize
                                && !crate::ordering::binary_predicate::<Item, Less>(
                                    Layout::recompose(Leaves::load($( &$shared_a, )+ zero_offsets, right_start + right_rank - 1usize)),
                                    Layout::recompose(Leaves::load($( &$shared_a, )+ zero_offsets, base + left_rank)),
                                )
                            {
                                low.store(left_rank + 1usize);
                            } else {
                                high.store(left_rank);
                            }
                        }
                        let left_rank = low.read();
                        let right_rank = rank - left_rank;
                        if left_rank < left_len {
                            if right_rank >= right_len
                                || !crate::ordering::binary_predicate::<Item, Less>(
                                    Layout::recompose(Leaves::load($( &$shared_a, )+ zero_offsets, right_start + right_rank)),
                                    Layout::recompose(Leaves::load($( &$shared_a, )+ zero_offsets, base + left_rank)),
                                )
                            {
                                Leaves::load($( &$shared_a, )+ zero_offsets, base + left_rank).store(
                                    $( &mut $shared_b, )+ zero_offsets, local,
                                );
                            } else {
                                Leaves::load($( &$shared_a, )+ zero_offsets, right_start + right_rank).store(
                                    $( &mut $shared_b, )+ zero_offsets, local,
                                );
                            }
                        } else {
                            Leaves::load($( &$shared_a, )+ zero_offsets, right_start + right_rank).store(
                                $( &mut $shared_b, )+ zero_offsets, local,
                            );
                        }
                    } else {
                        while low.read() < high.read() {
                            let left_rank = (low.read() + high.read()) / 2usize;
                            let right_rank = rank - left_rank;
                            if left_rank < left_len
                                && right_rank > 0usize
                                && !crate::ordering::binary_predicate::<Item, Less>(
                                    Layout::recompose(Leaves::load($( &$shared_b, )+ zero_offsets, right_start + right_rank - 1usize)),
                                    Layout::recompose(Leaves::load($( &$shared_b, )+ zero_offsets, base + left_rank)),
                                )
                            {
                                low.store(left_rank + 1usize);
                            } else {
                                high.store(left_rank);
                            }
                        }
                        let left_rank = low.read();
                        let right_rank = rank - left_rank;
                        if left_rank < left_len {
                            if right_rank >= right_len
                                || !crate::ordering::binary_predicate::<Item, Less>(
                                    Layout::recompose(Leaves::load($( &$shared_b, )+ zero_offsets, right_start + right_rank)),
                                    Layout::recompose(Leaves::load($( &$shared_b, )+ zero_offsets, base + left_rank)),
                                )
                            {
                                Leaves::load($( &$shared_b, )+ zero_offsets, base + left_rank).store(
                                    $( &mut $shared_a, )+ zero_offsets, local,
                                );
                            } else {
                                Leaves::load($( &$shared_b, )+ zero_offsets, right_start + right_rank).store(
                                    $( &mut $shared_a, )+ zero_offsets, local,
                                );
                            }
                        } else {
                            Leaves::load($( &$shared_b, )+ zero_offsets, right_start + right_rank).store(
                                $( &mut $shared_a, )+ zero_offsets, local,
                            );
                        }
                    }
                }
                sync_cube();
                source_a.store(1u32 - source_a.read());
                width.store(width.read() * 2usize);
            }

            if local < tile_len {
                if source_a.read() != 0u32 {
                    Leaves::load($( &$shared_a, )+ zero_offsets, local).store(
                        $( $out, )+ write_offsets, global,
                    );
                } else {
                    Leaves::load($( &$shared_b, )+ zero_offsets, local).store(
                        $( $out, )+ write_offsets, global,
                    );
                }
            }
        }

        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $merge<
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive, )+
            Leaves: CubeType + Send + Sync + 'static
                + $load<$( $leaf ),+>
                + $store<$( $leaf ),+>,
            Expr: $eval<Item, $( $leaf ),+>,
            Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
            Less: BinaryPredicateOp<Item>,
        >(
            $( $slot: &[$leaf], )+
            read_offsets: &[u32],
            logical_len_buffer: &[u32],
            params: &[u32],
            zero_offsets: &[u32],
            $( $out: &mut [$leaf], )+
            write_offsets: &[u32],
        ) {
            let logical_len = logical_len_buffer[0] as usize;
            let run_width = params[0] as usize;
            let pair_width = if logical_len == 0usize {
                1usize
            } else if logical_len <= run_width
                || run_width > logical_len - run_width
            {
                logical_len
            } else {
                run_width * 2usize
            };
            let merge_tile_items = BLOCK_ITEMS * $merge_items;
            let tiles_per_pair = (pair_width + merge_tile_items - 1usize) / merge_tile_items;
            let pair = (CUBE_POS as usize) / tiles_per_pair;
            let tile = (CUBE_POS as usize) % tiles_per_pair;
            let base = pair * pair_width;

            if base < logical_len {
                let pair_len = if logical_len - base < pair_width {
                    logical_len - base
                } else {
                    pair_width
                };
                let left_len = if pair_len < run_width { pair_len } else { run_width };
                let right_start = base + left_len;
                let right_len = pair_len - left_len;
                let tile_rank_start = tile * merge_tile_items;

                if tile_rank_start < pair_len {
                    let tile_rank_end = if tile_rank_start + merge_tile_items < pair_len {
                        tile_rank_start + merge_tile_items
                    } else {
                        pair_len
                    };

                    // Both tile boundaries are uniform for the workgroup.  Compute
                    // them once; repeating these global-memory binary searches in
                    // every lane is particularly expensive for wide comparators.
                    let mut partition = Shared::<[u32]>::new_slice(4usize);
                    if UNIT_POS == 0u32 {
                        let begin_low_init = if tile_rank_start > right_len {
                            tile_rank_start - right_len
                        } else {
                            0usize
                        };
                        let begin_high_init = if tile_rank_start < left_len {
                            tile_rank_start
                        } else {
                            left_len
                        };
                        let begin_low = RuntimeCell::<usize>::new(begin_low_init);
                        let begin_high = RuntimeCell::<usize>::new(begin_high_init);
                        while begin_low.read() < begin_high.read() {
                            let left_rank = (begin_low.read() + begin_high.read()) / 2usize;
                            let right_rank = tile_rank_start - left_rank;
                            if left_rank < left_len
                                && right_rank > 0usize
                                && !crate::ordering::binary_predicate::<Item, Less>(
                                    Expr::$method($( $slot, )+ read_offsets, right_start + right_rank - 1usize),
                                    Expr::$method($( $slot, )+ read_offsets, base + left_rank),
                                )
                            {
                                begin_low.store(left_rank + 1usize);
                            } else {
                                begin_high.store(left_rank);
                            }
                        }

                        let end_low_init = if tile_rank_end > right_len {
                            tile_rank_end - right_len
                        } else {
                            0usize
                        };
                        let end_high_init = if tile_rank_end < left_len {
                            tile_rank_end
                        } else {
                            left_len
                        };
                        let end_low = RuntimeCell::<usize>::new(end_low_init);
                        let end_high = RuntimeCell::<usize>::new(end_high_init);
                        while end_low.read() < end_high.read() {
                            let left_rank = (end_low.read() + end_high.read()) / 2usize;
                            let right_rank = tile_rank_end - left_rank;
                            if left_rank < left_len
                                && right_rank > 0usize
                                && !crate::ordering::binary_predicate::<Item, Less>(
                                    Expr::$method($( $slot, )+ read_offsets, right_start + right_rank - 1usize),
                                    Expr::$method($( $slot, )+ read_offsets, base + left_rank),
                                )
                            {
                                end_low.store(left_rank + 1usize);
                            } else {
                                end_high.store(left_rank);
                            }
                        }

                        let left_begin = begin_low.read();
                        let right_begin = tile_rank_start - left_begin;
                        partition[0] = left_begin as u32;
                        partition[1] = right_begin as u32;
                        partition[2] = (end_low.read() - left_begin) as u32;
                        partition[3] = ((tile_rank_end - end_low.read()) - right_begin) as u32;
                    }
                    sync_cube();

                    let left_begin = partition[0] as usize;
                    let right_begin = partition[1] as usize;
                    let left_count = partition[2] as usize;
                    let right_count = partition[3] as usize;
                    let tile_len = left_count + right_count;

                    $( let mut $shared_a = Shared::<[$leaf]>::new_slice(BLOCK_ITEMS * $merge_items); )+
                    let load_pos = RuntimeCell::<usize>::new(UNIT_POS as usize);
                    while load_pos.read() < tile_len {
                        let source = if load_pos.read() < left_count {
                            base + left_begin + load_pos.read()
                        } else {
                            right_start + right_begin + load_pos.read() - left_count
                        };
                        Layout::decompose(Expr::$method($( $slot, )+ read_offsets, source)).store(
                            $( &mut $shared_a, )+ zero_offsets, load_pos.read(),
                        );
                        load_pos.store(load_pos.read() + BLOCK_ITEMS);
                    }
                    sync_cube();

                    let local_start = (UNIT_POS as usize) * $merge_items;
                    if local_start < tile_len {
                        let local_end = if local_start + $merge_items < tile_len {
                            local_start + $merge_items
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
                                    Layout::recompose(Leaves::load(
                                        $( &$shared_a, )+ zero_offsets,
                                        left_count + right_rank - 1usize,
                                    )),
                                    Layout::recompose(Leaves::load(
                                        $( &$shared_a, )+ zero_offsets, left_rank,
                                    )),
                                )
                            {
                                local_low.store(left_rank + 1usize);
                            } else {
                                local_high.store(left_rank);
                            }
                        }

                        let left_rank = RuntimeCell::<usize>::new(local_low.read());
                        let right_rank = RuntimeCell::<usize>::new(local_start - local_low.read());
                        let local_cursor = RuntimeCell::<usize>::new(local_start);
                        while local_cursor.read() < local_end {
                            let output = base + tile_rank_start + local_cursor.read();
                            if left_rank.read() < left_count
                                && (right_rank.read() >= right_count
                                    || !crate::ordering::binary_predicate::<Item, Less>(
                                        Layout::recompose(Leaves::load(
                                            $( &$shared_a, )+ zero_offsets,
                                            left_count + right_rank.read(),
                                        )),
                                        Layout::recompose(Leaves::load(
                                            $( &$shared_a, )+ zero_offsets, left_rank.read(),
                                        )))
                                    )
                            {
                                Leaves::load(
                                    $( &$shared_a, )+ zero_offsets, left_rank.read(),
                                ).store($( $out, )+ write_offsets, output);
                                left_rank.store(left_rank.read() + 1usize);
                            } else {
                                let source = left_count + right_rank.read();
                                Leaves::load($( &$shared_a, )+ zero_offsets, source).store(
                                    $( $out, )+ write_offsets, output,
                                );
                                right_rank.store(right_rank.read() + 1usize);
                            }
                            local_cursor.store(local_cursor.read() + 1usize);
                        }
                    }
                }
            }
        }
    };
}

define_sort_kernels!(block_sort_s1, merge_runs_s1, Eval1, eval1; [L0:slot0:out0:shared_a0:shared_b0]; Last<L0>, LoadLeaves1, StoreLeaves1; 16usize);
define_sort_kernels!(block_sort_s2, merge_runs_s2, Eval2, eval2; [L0:slot0:out0:shared_a0:shared_b0,L1:slot1:out1:shared_a1:shared_b1]; More<L0,Last<L1>>, LoadLeaves2, StoreLeaves2; 16usize);
define_sort_kernels!(block_sort_s3, merge_runs_s3, Eval3, eval3; [L0:slot0:out0:shared_a0:shared_b0,L1:slot1:out1:shared_a1:shared_b1,L2:slot2:out2:shared_a2:shared_b2]; More<L0,More<L1,Last<L2>>>, LoadLeaves3, StoreLeaves3; 8usize);
define_sort_kernels!(block_sort_s4, merge_runs_s4, Eval4, eval4; [L0:slot0:out0:shared_a0:shared_b0,L1:slot1:out1:shared_a1:shared_b1,L2:slot2:out2:shared_a2:shared_b2,L3:slot3:out3:shared_a3:shared_b3]; More<L0,More<L1,More<L2,Last<L3>>>>, LoadLeaves4, StoreLeaves4; 8usize);
define_sort_kernels!(block_sort_s5, merge_runs_s5, Eval5, eval5; [L0:slot0:out0:shared_a0:shared_b0,L1:slot1:out1:shared_a1:shared_b1,L2:slot2:out2:shared_a2:shared_b2,L3:slot3:out3:shared_a3:shared_b3,L4:slot4:out4:shared_a4:shared_b4]; More<L0,More<L1,More<L2,More<L3,Last<L4>>>>>, LoadLeaves5, StoreLeaves5; 8usize);
define_sort_kernels!(block_sort_s6, merge_runs_s6, Eval6, eval6; [L0:slot0:out0:shared_a0:shared_b0,L1:slot1:out1:shared_a1:shared_b1,L2:slot2:out2:shared_a2:shared_b2,L3:slot3:out3:shared_a3:shared_b3,L4:slot4:out4:shared_a4:shared_b4,L5:slot5:out5:shared_a5:shared_b5]; More<L0,More<L1,More<L2,More<L3,More<L4,Last<L5>>>>>>, LoadLeaves6, StoreLeaves6; 8usize);
define_sort_kernels!(block_sort_s7, merge_runs_s7, Eval7, eval7; [L0:slot0:out0:shared_a0:shared_b0,L1:slot1:out1:shared_a1:shared_b1,L2:slot2:out2:shared_a2:shared_b2,L3:slot3:out3:shared_a3:shared_b3,L4:slot4:out4:shared_a4:shared_b4,L5:slot5:out5:shared_a5:shared_b5,L6:slot6:out6:shared_a6:shared_b6]; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,Last<L6>>>>>>>, LoadLeaves7, StoreLeaves7; 4usize);
define_sort_kernels!(block_sort_s8, merge_runs_s8, Eval8, eval8; [L0:slot0:out0:shared_a0:shared_b0,L1:slot1:out1:shared_a1:shared_b1,L2:slot2:out2:shared_a2:shared_b2,L3:slot3:out3:shared_a3:shared_b3,L4:slot4:out4:shared_a4:shared_b4,L5:slot5:out5:shared_a5:shared_b5,L6:slot6:out6:shared_a6:shared_b6,L7:slot7:out7:shared_a7:shared_b7]; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,Last<L7>>>>>>>>, LoadLeaves8, StoreLeaves8; 4usize);
define_sort_kernels!(block_sort_s9, merge_runs_s9, Eval9, eval9; [L0:slot0:out0:shared_a0:shared_b0,L1:slot1:out1:shared_a1:shared_b1,L2:slot2:out2:shared_a2:shared_b2,L3:slot3:out3:shared_a3:shared_b3,L4:slot4:out4:shared_a4:shared_b4,L5:slot5:out5:shared_a5:shared_b5,L6:slot6:out6:shared_a6:shared_b6,L7:slot7:out7:shared_a7:shared_b7,L8:slot8:out8:shared_a8:shared_b8]; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,Last<L8>>>>>>>>>, LoadLeaves9, StoreLeaves9; 4usize);
define_sort_kernels!(block_sort_s10, merge_runs_s10, Eval10, eval10; [L0:slot0:out0:shared_a0:shared_b0,L1:slot1:out1:shared_a1:shared_b1,L2:slot2:out2:shared_a2:shared_b2,L3:slot3:out3:shared_a3:shared_b3,L4:slot4:out4:shared_a4:shared_b4,L5:slot5:out5:shared_a5:shared_b5,L6:slot6:out6:shared_a6:shared_b6,L7:slot7:out7:shared_a7:shared_b7,L8:slot8:out8:shared_a8:shared_b8,L9:slot9:out9:shared_a9:shared_b9]; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,More<L8,Last<L9>>>>>>>>>>, LoadLeaves10, StoreLeaves10; 4usize);
define_sort_kernels!(block_sort_s11, merge_runs_s11, Eval11, eval11; [L0:slot0:out0:shared_a0:shared_b0,L1:slot1:out1:shared_a1:shared_b1,L2:slot2:out2:shared_a2:shared_b2,L3:slot3:out3:shared_a3:shared_b3,L4:slot4:out4:shared_a4:shared_b4,L5:slot5:out5:shared_a5:shared_b5,L6:slot6:out6:shared_a6:shared_b6,L7:slot7:out7:shared_a7:shared_b7,L8:slot8:out8:shared_a8:shared_b8,L9:slot9:out9:shared_a9:shared_b9,L10:slot10:out10:shared_a10:shared_b10]; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,More<L8,More<L9,Last<L10>>>>>>>>>>>, LoadLeaves11, StoreLeaves11; 4usize);
define_sort_kernels!(block_sort_s12, merge_runs_s12, Eval12, eval12; [L0:slot0:out0:shared_a0:shared_b0,L1:slot1:out1:shared_a1:shared_b1,L2:slot2:out2:shared_a2:shared_b2,L3:slot3:out3:shared_a3:shared_b3,L4:slot4:out4:shared_a4:shared_b4,L5:slot5:out5:shared_a5:shared_b5,L6:slot6:out6:shared_a6:shared_b6,L7:slot7:out7:shared_a7:shared_b7,L8:slot8:out8:shared_a8:shared_b8,L9:slot9:out9:shared_a9:shared_b9,L10:slot10:out10:shared_a10:shared_b10,L11:slot11:out11:shared_a11:shared_b11]; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,More<L8,More<L9,More<L10,Last<L11>>>>>>>>>>>>, LoadLeaves12, StoreLeaves12; 4usize);

trait SortPassDispatch<R, Input, Output, Item, ReadSlots, WriteSlots, Less>
where
    R: Runtime,
{
    fn block(exec: &Executor<R>, input: &Input, output: &Output) -> Result<(), Error>;

    fn merge(exec: &Executor<R>, input: &Input, output: &Output, width: usize)
    -> Result<(), Error>;
}

macro_rules! impl_sort_pass_dispatch {
    (
        $read_arity:ty, $storage:ty, $eval:ident, $block:ident, $merge:ident;
        [$( $leaf:ident : $index:literal ),+], $env:ty;
        $leaves:ty;
        $merge_items:expr
    ) => {
        impl<R, Input, Output, Item, Less, $( $leaf ),+>
            SortPassDispatch<R, Input, Output, Item, $env, $env, Less>
            for Dispatch<$read_arity, crate::S1>
        where
            R: Runtime,
            Item: StorageLayout<StorageArity = $storage, StorageLeaves = $leaves>
                + Send
                + Sync
                + 'static,
            Item::DeviceLayout:
                Decompose<Item, Leaves = $leaves> + Recompose<Item, Leaves = $leaves>,
            Less: BinaryPredicateOp<Item>,
            $( $leaf: MStorageElement, )+
            Input: ReadExpression<Item = Item, ReadArity = $read_arity>
                + LowerReadExpression<Slots = $env>
                + StageRead<R, Env0>,
            Input::DeviceExpr: $eval<Item, $( $leaf ),+>,
            Output: OutputExpression<StorageArity = $storage>
                + LowerOutputExpression<Slots = $env>
                + StageOutput<R, Env0>,
        {
            fn block(
                exec: &Executor<R>,
                input: &Input,
                output: &Output,
            ) -> Result<(), Error> {
                let capacity = input.logical_len()?;
                if output.logical_len()? < capacity {
                    return Err(Error::OutputTooShort {
                        input: capacity,
                        output: output.logical_len()?,
                    });
                }
                if capacity == 0 {
                    return Ok(());
                }
                let logical_len = input.logical_extent()?.materialize(exec)?;
                let mut reads = StagedBindings::new();
                input.stage_at(exec.client(), exec.id(), &mut reads)?;
                let mut writes = OutputBindings::new();
                output.stage_output(exec.id(), &mut writes)?;
                let read_offsets = exec.client().create_from_slice(u32::as_bytes(&reads.offsets));
                let write_offsets = exec.client().create_from_slice(u32::as_bytes(&writes.offsets));
                let zero_offsets = vec![$( { let _ = stringify!($leaf); 0u32 } ),+];
                let zero_offsets_handle =
                    exec.client().create_from_slice(u32::as_bytes(&zero_offsets));
                unsafe {
                    $block::launch_unchecked::<
                        Item,
                        $( $leaf, )+
                        Item::StorageLeaves,
                        Input::DeviceExpr,
                        Item::DeviceLayout,
                        Less,
                        R,
                    >(
                        exec.client(),
                        cube_count_1d(capacity.div_ceil(BLOCK_ITEMS))?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(reads.slots[$index].0.clone(), reads.slots[$index].1), )+
                        BufferArg::from_raw_parts(read_offsets, reads.offsets.len()),
                        BufferArg::from_raw_parts(logical_len.handle.clone(), 1),
                        BufferArg::from_raw_parts(zero_offsets_handle, zero_offsets.len()),
                        $( BufferArg::from_raw_parts(writes.slots[$index].0.clone(), writes.slots[$index].1), )+
                        BufferArg::from_raw_parts(write_offsets, writes.offsets.len()),
                    );
                }
                Ok(())
            }

            fn merge(
                exec: &Executor<R>,
                input: &Input,
                output: &Output,
                width: usize,
            ) -> Result<(), Error> {
                let capacity = input.logical_len()?;
                if capacity == 0 {
                    return Ok(());
                }
                let logical_len = input.logical_extent()?.materialize(exec)?;
                let width_u32 = u32::try_from(width)
                    .map_err(|_| Error::LengthTooLarge { len: width })?;
                let mut reads = StagedBindings::new();
                input.stage_at(exec.client(), exec.id(), &mut reads)?;
                let mut writes = OutputBindings::new();
                output.stage_output(exec.id(), &mut writes)?;
                let read_offsets = exec.client().create_from_slice(u32::as_bytes(&reads.offsets));
                let write_offsets = exec.client().create_from_slice(u32::as_bytes(&writes.offsets));
                let params = exec
                    .client()
                    .create_from_slice(u32::as_bytes(&[width_u32]));
                let zero_offsets = vec![$( { let _ = stringify!($leaf); 0u32 } ),+];
                let zero_offsets_handle =
                    exec.client().create_from_slice(u32::as_bytes(&zero_offsets));
                let pair_width = width.saturating_mul(2).min(capacity);
                let pairs = capacity.div_ceil(pair_width);
                let tiles_per_pair = pair_width.div_ceil(BLOCK_ITEMS * $merge_items);
                let cubes = pairs.saturating_mul(tiles_per_pair).max(1);
                unsafe {
                    $merge::launch_unchecked::<
                        Item,
                        $( $leaf, )+
                        Item::StorageLeaves,
                        Input::DeviceExpr,
                        Item::DeviceLayout,
                        Less,
                        R,
                    >(
                        exec.client(),
                        cube_count_1d(cubes)?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(reads.slots[$index].0.clone(), reads.slots[$index].1), )+
                        BufferArg::from_raw_parts(read_offsets, reads.offsets.len()),
                        BufferArg::from_raw_parts(logical_len.handle.clone(), 1),
                        BufferArg::from_raw_parts(params, 1),
                        BufferArg::from_raw_parts(zero_offsets_handle, zero_offsets.len()),
                        $( BufferArg::from_raw_parts(writes.slots[$index].0.clone(), writes.slots[$index].1), )+
                        BufferArg::from_raw_parts(write_offsets, writes.offsets.len()),
                    );
                }
                Ok(())
            }
        }
    };
}

impl_sort_pass_dispatch!(crate::A1, crate::S1, Eval1, block_sort_s1, merge_runs_s1; [L0:0], Env1<L0>; Last<L0>; 16usize);
impl_sort_pass_dispatch!(crate::A2, crate::S2, Eval2, block_sort_s2, merge_runs_s2; [L0:0,L1:1], Env2<L0,L1>; More<L0,Last<L1>>; 16usize);
impl_sort_pass_dispatch!(crate::A3, crate::S3, Eval3, block_sort_s3, merge_runs_s3; [L0:0,L1:1,L2:2], Env3<L0,L1,L2>; More<L0,More<L1,Last<L2>>>; 8usize);
impl_sort_pass_dispatch!(crate::A4, crate::S4, Eval4, block_sort_s4, merge_runs_s4; [L0:0,L1:1,L2:2,L3:3], Env4<L0,L1,L2,L3>; More<L0,More<L1,More<L2,Last<L3>>>>; 8usize);
impl_sort_pass_dispatch!(crate::A5, crate::S5, Eval5, block_sort_s5, merge_runs_s5; [L0:0,L1:1,L2:2,L3:3,L4:4], Env5<L0,L1,L2,L3,L4>; More<L0,More<L1,More<L2,More<L3,Last<L4>>>>>; 8usize);
impl_sort_pass_dispatch!(crate::A6, crate::S6, Eval6, block_sort_s6, merge_runs_s6; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5], Env6<L0,L1,L2,L3,L4,L5>; More<L0,More<L1,More<L2,More<L3,More<L4,Last<L5>>>>>>; 8usize);
impl_sort_pass_dispatch!(crate::A7, crate::S7, Eval7, block_sort_s7, merge_runs_s7; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6], Env7<L0,L1,L2,L3,L4,L5,L6>; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,Last<L6>>>>>>>; 4usize);
impl_sort_pass_dispatch!(crate::A8, crate::S8, Eval8, block_sort_s8, merge_runs_s8; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7], Env8<L0,L1,L2,L3,L4,L5,L6,L7>; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,Last<L7>>>>>>>>; 4usize);
impl_sort_pass_dispatch!(crate::A9, crate::S9, Eval9, block_sort_s9, merge_runs_s9; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8], Env9<L0,L1,L2,L3,L4,L5,L6,L7,L8>; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,Last<L8>>>>>>>>>; 4usize);
impl_sort_pass_dispatch!(crate::A10, crate::S10, Eval10, block_sort_s10, merge_runs_s10; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8,L9:9], Env10<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9>; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,More<L8,Last<L9>>>>>>>>>>; 4usize);
impl_sort_pass_dispatch!(crate::A11, crate::S11, Eval11, block_sort_s11, merge_runs_s11; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8,L9:9,L10:10], Env11<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10>; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,More<L8,More<L9,Last<L10>>>>>>>>>>>; 4usize);
impl_sort_pass_dispatch!(crate::A12, crate::S12, Eval12, block_sort_s12, merge_runs_s12; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8,L9:9,L10:10,L11:11], Env12<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11>; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,More<L8,More<L9,More<L10,Last<L11>>>>>>>>>>>>; 4usize);

/// Capability for sorting an owned flat-row storage value.
pub(crate) trait SortStorageItem<R: Runtime, Less>: RowAlloc<R> + Sized {
    fn sort_storage(exec: &Executor<R>, input: Self::RowStorage)
    -> Result<Self::RowStorage, Error>;
}

impl<R, Item, Less> SortStorageItem<R, Less> for Item
where
    R: Runtime,
    Item: RowAlloc<R> + StorageLayout + Send + Sync + 'static,
    Less: BinaryPredicateOp<Item>,
    <Item as RowAlloc<R>>::RowStorage: RowStorage<R>,
    <<Item as RowAlloc<R>>::RowStorage as RowStorage<R>>::Read:
        ReadExpression<Item = Item> + LowerReadExpression + StageRead<R, Env0>,
    <<Item as RowAlloc<R>>::RowStorage as RowStorage<R>>::Write: OutputExpression<StorageArity = Item::StorageArity>
        + LowerOutputExpression
        + StageOutput<R, Env0>,
    Dispatch<
        <<<Item as RowAlloc<R>>::RowStorage as RowStorage<R>>::Read as ReadExpression>::ReadArity,
        crate::S1,
    >: SortPassDispatch<
            R,
            <<Item as RowAlloc<R>>::RowStorage as RowStorage<R>>::Read,
            <<Item as RowAlloc<R>>::RowStorage as RowStorage<R>>::Write,
            Item,
            <<<Item as RowAlloc<R>>::RowStorage as RowStorage<R>>::Read as LowerReadExpression>::Slots,
            <<<Item as RowAlloc<R>>::RowStorage as RowStorage<R>>::Write as LowerOutputExpression>::Slots,
            Less,
        >,
{
    fn sort_storage(
        exec: &Executor<R>,
        input: Self::RowStorage,
    ) -> Result<Self::RowStorage, Error> {
        let len = input.len()?;
        let extent = input.logical_extent();
        if len == 0 {
            return Ok(input);
        }

        let mut current_keys = exec.alloc_row::<Item>(len);
        current_keys.set_logical_extent(extent.clone());
        let input_read = input.read();
        <Dispatch<
            <<<Item as RowAlloc<R>>::RowStorage as RowStorage<R>>::Read as ReadExpression>::ReadArity,
            crate::S1,
        > as SortPassDispatch<R, _, _, Item, _, _, Less>>::block(
            exec,
            &input_read,
            &current_keys.write(),
        )?;

        if len > BLOCK_ITEMS {
            let mut next_keys = exec.alloc_row::<Item>(len);
            next_keys.set_logical_extent(extent.clone());
            let mut width = BLOCK_ITEMS;
            while width < len {
                let current_read = current_keys.read();
                <Dispatch<
                    <<<Item as RowAlloc<R>>::RowStorage as RowStorage<R>>::Read as ReadExpression>::ReadArity,
                    crate::S1,
                > as SortPassDispatch<
                    R,
                    _,
                    _,
                    Item,
                    _,
                    _,
                    Less,
                >>::merge(
                    exec,
                    &current_read,
                    &next_keys.write(),
                    width,
                )?;
                core::mem::swap(&mut current_keys, &mut next_keys);
                width = width.saturating_mul(2);
            }
        }

        Ok(current_keys)
    }
}

macro_rules! impl_sort_leaves {
    ($read_arity:ty, $arity:ty, $eval:ident; $env:ty; $leaves:ty; $storage:ty; $( $leaf:ident ),+) => {
        impl<R, Item, $( $leaf ),+> SortLeaves<R, Item> for $leaves
        where
            R: Runtime,
            $( $leaf: MStorageElement, )+
            Item: RowAlloc<
                    R,
                    StorageArity = $arity,
                    StorageLeaves = $leaves,
                > + StorageLayout
                + Send
                + Sync
                + 'static,
            <Item as RowAlloc<R>>::RowStorage:
                RowStorage<R, ReadSlots = $env, WriteSlots = $env>,
            <<Item as RowAlloc<R>>::RowStorage as RowStorage<R>>::Read:
                ReadExpression<Item = Item, ReadArity = $read_arity>
                    + LowerReadExpression<Slots = $env>
                    + StageRead<R, Env0>,
            <<<Item as RowAlloc<R>>::RowStorage as RowStorage<R>>::Read as LowerReadExpression>::DeviceExpr:
                $eval<Item, $( $leaf ),+>,
            Item::DeviceLayout:
                Decompose<Item, Leaves = $leaves> + Recompose<Item, Leaves = $leaves>,
            <<Item as RowAlloc<R>>::RowStorage as RowStorage<R>>::Write:
                OutputExpression<StorageArity = $arity>
                    + LowerOutputExpression<Slots = $env>
                    + StageOutput<R, Env0>,
        {
            fn sort_storage<Less>(
                exec: &Executor<R>,
                input: Item::RowStorage,
            ) -> Result<Item::RowStorage, Error>
            where
                Less: BinaryPredicateOp<Item>,
            {
                <Item as SortStorageItem<R, Less>>::sort_storage(exec, input)
            }
        }
    };
}

impl_sort_leaves!(crate::A1, crate::S1, Eval1; Env1<L0>; Last<L0>; crate::DeviceVec<R,L0>; L0);
impl_sort_leaves!(crate::A2, crate::S2, Eval2; Env2<L0,L1>; More<L0,Last<L1>>; crate::Zip<crate::DeviceVec<R,L0>,crate::DeviceVec<R,L1>>; L0,L1);
impl_sort_leaves!(crate::A3, crate::S3, Eval3; Env3<L0,L1,L2>; More<L0,More<L1,Last<L2>>>; crate::Zip<crate::Zip<crate::DeviceVec<R,L0>,crate::DeviceVec<R,L1>>,crate::DeviceVec<R,L2>>; L0,L1,L2);
impl_sort_leaves!(crate::A4, crate::S4, Eval4; Env4<L0,L1,L2,L3>; More<L0,More<L1,More<L2,Last<L3>>>>; crate::Zip<crate::Zip<crate::Zip<crate::DeviceVec<R,L0>,crate::DeviceVec<R,L1>>,crate::DeviceVec<R,L2>>,crate::DeviceVec<R,L3>>; L0,L1,L2,L3);
impl_sort_leaves!(crate::A5, crate::S5, Eval5; Env5<L0,L1,L2,L3,L4>; More<L0,More<L1,More<L2,More<L3,Last<L4>>>>>; crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::DeviceVec<R,L0>,crate::DeviceVec<R,L1>>,crate::DeviceVec<R,L2>>,crate::DeviceVec<R,L3>>,crate::DeviceVec<R,L4>>; L0,L1,L2,L3,L4);
impl_sort_leaves!(crate::A6, crate::S6, Eval6; Env6<L0,L1,L2,L3,L4,L5>; More<L0,More<L1,More<L2,More<L3,More<L4,Last<L5>>>>>>; crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::DeviceVec<R,L0>,crate::DeviceVec<R,L1>>,crate::DeviceVec<R,L2>>,crate::DeviceVec<R,L3>>,crate::DeviceVec<R,L4>>,crate::DeviceVec<R,L5>>; L0,L1,L2,L3,L4,L5);
impl_sort_leaves!(crate::A7, crate::S7, Eval7; Env7<L0,L1,L2,L3,L4,L5,L6>; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,Last<L6>>>>>>>; crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::DeviceVec<R,L0>,crate::DeviceVec<R,L1>>,crate::DeviceVec<R,L2>>,crate::DeviceVec<R,L3>>,crate::DeviceVec<R,L4>>,crate::DeviceVec<R,L5>>,crate::DeviceVec<R,L6>>; L0,L1,L2,L3,L4,L5,L6);
impl_sort_leaves!(crate::A8, crate::S8, Eval8; Env8<L0,L1,L2,L3,L4,L5,L6,L7>; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,Last<L7>>>>>>>>; crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::DeviceVec<R,L0>,crate::DeviceVec<R,L1>>,crate::DeviceVec<R,L2>>,crate::DeviceVec<R,L3>>,crate::DeviceVec<R,L4>>,crate::DeviceVec<R,L5>>,crate::DeviceVec<R,L6>>,crate::DeviceVec<R,L7>>; L0,L1,L2,L3,L4,L5,L6,L7);
impl_sort_leaves!(crate::A9, crate::S9, Eval9; Env9<L0,L1,L2,L3,L4,L5,L6,L7,L8>; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,Last<L8>>>>>>>>>; crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::DeviceVec<R,L0>,crate::DeviceVec<R,L1>>,crate::DeviceVec<R,L2>>,crate::DeviceVec<R,L3>>,crate::DeviceVec<R,L4>>,crate::DeviceVec<R,L5>>,crate::DeviceVec<R,L6>>,crate::DeviceVec<R,L7>>,crate::DeviceVec<R,L8>>; L0,L1,L2,L3,L4,L5,L6,L7,L8);
impl_sort_leaves!(crate::A10, crate::S10, Eval10; Env10<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9>; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,More<L8,Last<L9>>>>>>>>>>; crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::DeviceVec<R,L0>,crate::DeviceVec<R,L1>>,crate::DeviceVec<R,L2>>,crate::DeviceVec<R,L3>>,crate::DeviceVec<R,L4>>,crate::DeviceVec<R,L5>>,crate::DeviceVec<R,L6>>,crate::DeviceVec<R,L7>>,crate::DeviceVec<R,L8>>,crate::DeviceVec<R,L9>>; L0,L1,L2,L3,L4,L5,L6,L7,L8,L9);
impl_sort_leaves!(crate::A11, crate::S11, Eval11; Env11<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10>; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,More<L8,More<L9,Last<L10>>>>>>>>>>>; crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::DeviceVec<R,L0>,crate::DeviceVec<R,L1>>,crate::DeviceVec<R,L2>>,crate::DeviceVec<R,L3>>,crate::DeviceVec<R,L4>>,crate::DeviceVec<R,L5>>,crate::DeviceVec<R,L6>>,crate::DeviceVec<R,L7>>,crate::DeviceVec<R,L8>>,crate::DeviceVec<R,L9>>,crate::DeviceVec<R,L10>>; L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10);
impl_sort_leaves!(crate::A12, crate::S12, Eval12; Env12<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11>; More<L0,More<L1,More<L2,More<L3,More<L4,More<L5,More<L6,More<L7,More<L8,More<L9,More<L10,Last<L11>>>>>>>>>>>>; crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::Zip<crate::DeviceVec<R,L0>,crate::DeviceVec<R,L1>>,crate::DeviceVec<R,L2>>,crate::DeviceVec<R,L3>>,crate::DeviceVec<R,L4>>,crate::DeviceVec<R,L5>>,crate::DeviceVec<R,L6>>,crate::DeviceVec<R,L7>>,crate::DeviceVec<R,L8>>,crate::DeviceVec<R,L9>>,crate::DeviceVec<R,L10>>,crate::DeviceVec<R,L11>>; L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11);
