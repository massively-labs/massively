//! Reusable prefix-scan control primitives.

use cubecl::prelude::*;

use crate::{
    A13, DeviceVec, Dispatch, Error, Executor, MStorageElement, ReadExpression, RowStorage, S12,
    StorageLayout,
    eval::Eval13,
    launch::cube_count_1d,
    output::{
        LowerOutputExpression, OutputBindings, OutputExpression, PaddedOutputSlots, SliceOutput,
        StageOutput,
    },
    read::{Adjacent, Env0, Env12, Env13, KernelReadSlots, LowerReadExpression, PaddedReadSlots},
    reduce::{ReductionOp, StageRead, StagedBindings},
    selection::FillOutput,
    storage::{
        Decompose, LoadMutPadded12, LoadPadded12, MutableLeaves, PlaneShuffleLeaves, Recompose,
        SharedLeaves, StorePadded12, StorePadded12Expand,
    },
    transform::materialize,
};

const BLOCK_SIZE: u32 = 256;

type FixedScanStorage<R, Item> = <Item as crate::allocation::ScratchStorage<R>>::Storage;
type FixedScanRead<R, Item> =
    crate::read::FixedRead<<FixedScanStorage<R, Item> as RowStorage<R>>::Read>;
type FixedScanOutput<R, Item> = <FixedScanStorage<R, Item> as RowStorage<R>>::Write;

#[cubecl::cube(launch_unchecked, explicit_define)]
fn u32_block_inclusive_scan_kernel(
    input: &[u32],
    len: &[u32],
    output: &mut [u32],
    block_sums: &mut [u32],
) {
    let unit = UNIT_POS as usize;
    let cube_dim = BLOCK_SIZE as usize;
    let global = (CUBE_POS as usize) * cube_dim + unit;
    let logical_len = len[0] as usize;
    let value = RuntimeCell::<u32>::new(if global < logical_len {
        input[global]
    } else {
        0u32
    });
    let valid = RuntimeCell::<u32>::new(if global < logical_len { 1u32 } else { 0u32 });

    let offset = RuntimeCell::<u32>::new(1u32);
    while offset.read() < PLANE_DIM {
        let left = plane_shuffle_up(value.read(), offset.read());
        let left_valid = plane_shuffle_up(valid.read(), offset.read());
        if UNIT_POS_PLANE >= offset.read() && left_valid != 0u32 {
            value.store(left + value.read());
            valid.store(1u32);
        }
        offset.store(offset.read() * 2u32);
    }

    let mut plane_values = Shared::<[u32]>::new_slice(cube_dim);
    let mut plane_valid = Shared::<[u32]>::new_slice(cube_dim);
    if UNIT_POS_PLANE + 1u32 == PLANE_DIM {
        plane_values[PLANE_POS as usize] = value.read();
        plane_valid[PLANE_POS as usize] = valid.read();
    }
    sync_cube();

    if unit == 0usize {
        let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
        let prefix = RuntimeCell::<u32>::new(plane_values[0]);
        let prefix_valid = RuntimeCell::<u32>::new(plane_valid[0]);
        let plane = RuntimeCell::<u32>::new(1u32);
        while plane.read() < plane_count {
            let index = plane.read() as usize;
            if plane_valid[index] != 0u32 {
                if prefix_valid.read() != 0u32 {
                    prefix.store(prefix.read() + plane_values[index]);
                } else {
                    prefix.store(plane_values[index]);
                    prefix_valid.store(1u32);
                }
            }
            plane_values[index] = prefix.read();
            plane.store(plane.read() + 1u32);
        }
    }
    sync_cube();

    if PLANE_POS > 0u32 && valid.read() != 0u32 {
        value.store(plane_values[PLANE_POS as usize - 1usize] + value.read());
    }

    if global < logical_len {
        output[global] = value.read();
    }
    if unit == 0usize {
        let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
        block_sums[CUBE_POS as usize] = plane_values[plane_count as usize - 1usize];
    }
}

#[cubecl::cube(launch_unchecked, explicit_define)]
fn u32_add_block_prefix_kernel(block_prefixes: &[u32], len: &[u32], output: &mut [u32]) {
    let block = CUBE_POS as usize;
    let global = block * BLOCK_SIZE as usize + UNIT_POS as usize;
    if block > 0usize && global < len[0] as usize {
        output[global] += block_prefixes[block - 1usize];
    }
}

#[cubecl::cube(launch_unchecked)]
fn copy_last_kernel(input: &[u32], len: &[u32], output: &mut [u32]) {
    if ABSOLUTE_POS == 0 {
        output[0] = if len[0] == 0u32 {
            0u32
        } else {
            input[len[0] as usize - 1usize]
        };
    }
}

#[cubecl::cube]
#[allow(clippy::too_many_arguments)]
fn scan_value_padded12<Item, O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, Leaves, Layout, Op>(
    value: Item,
    valid_value: u32,
    exclusive: bool,
    logical_len: usize,
    global: usize,
    unit: usize,
    block: usize,
    zero_offsets: &[u32],
    output_offsets: &[u32],
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
    sum0: &mut [O0],
    sum1: &mut [O1],
    sum2: &mut [O2],
    sum3: &mut [O3],
    sum4: &mut [O4],
    sum5: &mut [O5],
    sum6: &mut [O6],
    sum7: &mut [O7],
    sum8: &mut [O8],
    sum9: &mut [O9],
    sum10: &mut [O10],
    sum11: &mut [O11],
) where
    Item: CubeType + Send + Sync + 'static,
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
    Leaves: SharedLeaves
        + MutableLeaves
        + PlaneShuffleLeaves
        + LoadMutPadded12<
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
        > + StorePadded12<
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
        > + Send
        + Sync
        + 'static,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
{
    let cube_dim = BLOCK_SIZE as usize;
    let mut shared = Leaves::new_shared(cube_dim);
    let mut valid = Shared::<[u32]>::new_slice(cube_dim);
    let cells = Leaves::into_cells(Layout::decompose(value));
    let is_valid = RuntimeCell::<u32>::new(valid_value);
    let offset = RuntimeCell::<u32>::new(1u32);
    while offset.read() < PLANE_DIM {
        let left_cells = Leaves::into_cells(Leaves::shuffle_leaves_up(
            Leaves::read(&cells),
            offset.read(),
        ));
        let left_valid = plane_shuffle_up(is_valid.read(), offset.read());
        if UNIT_POS_PLANE >= offset.read() && left_valid != 0u32 {
            if is_valid.read() != 0u32 {
                let combined = Layout::decompose(Op::apply(
                    Layout::recompose(Leaves::read(&left_cells)),
                    Layout::recompose(Leaves::read(&cells)),
                ));
                Leaves::store(&cells, combined);
            } else {
                Leaves::store(&cells, Leaves::read(&left_cells));
                is_valid.store(1u32);
            }
        }
        offset.store(offset.read() * 2u32);
    }
    if UNIT_POS_PLANE + 1u32 == PLANE_DIM {
        Leaves::store_shared(Leaves::read(&cells), &mut shared, PLANE_POS as usize);
        valid[PLANE_POS as usize] = is_valid.read();
    }
    sync_cube();
    if unit == 0usize {
        let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
        let plane_cells = Leaves::into_cells(Leaves::load_shared(&shared, 0usize));
        let plane_is_valid = RuntimeCell::<u32>::new(valid[0]);
        let plane = RuntimeCell::<u32>::new(1u32);
        while plane.read() < plane_count {
            let index = plane.read() as usize;
            if valid[index] != 0u32 {
                if plane_is_valid.read() != 0u32 {
                    let combined = Layout::decompose(Op::apply(
                        Layout::recompose(Leaves::read(&plane_cells)),
                        Layout::recompose(Leaves::load_shared(&shared, index)),
                    ));
                    Leaves::store(&plane_cells, combined);
                } else {
                    Leaves::store(&plane_cells, Leaves::load_shared(&shared, index));
                    plane_is_valid.store(1u32);
                }
            }
            Leaves::store_shared(Leaves::read(&plane_cells), &mut shared, index);
            plane.store(plane.read() + 1u32);
        }
    }
    sync_cube();
    if PLANE_POS > 0u32 && is_valid.read() != 0u32 {
        let prefix = Leaves::load_shared(&shared, PLANE_POS as usize - 1usize);
        let combined = Layout::decompose(Op::apply(
            Layout::recompose(prefix),
            Layout::recompose(Leaves::read(&cells)),
        ));
        Leaves::store(&cells, combined);
    }
    if exclusive {
        let previous_cells =
            Leaves::into_cells(Leaves::shuffle_leaves_up(Leaves::read(&cells), 1u32));
        if UNIT_POS_PLANE == 0u32 && PLANE_POS > 0u32 {
            Leaves::store(
                &previous_cells,
                Leaves::load_shared(&shared, PLANE_POS as usize - 1usize),
            );
        }
        if unit > 0usize && global < logical_len {
            if block == 0usize {
                let initial = Layout::recompose(Leaves::load_mut_padded(
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
                    output_offsets,
                    0usize,
                ));
                let combined = Layout::decompose(Op::apply(
                    initial,
                    Layout::recompose(Leaves::read(&previous_cells)),
                ));
                Leaves::store(&previous_cells, combined);
            }
            Leaves::read(&previous_cells).store_padded(
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
                output_offsets,
                global,
            );
        }
    } else if global < logical_len {
        Leaves::read(&cells).store_padded(
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
            output_offsets,
            global,
        );
    }
    if unit == 0usize {
        let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
        Leaves::load_shared(&shared, plane_count as usize - 1usize).store_padded(
            sum0,
            sum1,
            sum2,
            sum3,
            sum4,
            sum5,
            sum6,
            sum7,
            sum8,
            sum9,
            sum10,
            sum11,
            zero_offsets,
            block,
        );
    }
}

macro_rules! define_padded_scan_kernel {
    ($name:ident,$eval:ident,$method:ident; [$( $leaf:ident:$slot:ident ),+]) => {
        #[cubecl::cube(launch_unchecked, explicit_define)]
        fn $name<
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive, )+
            O0: CubePrimitive, O1: CubePrimitive, O2: CubePrimitive, O3: CubePrimitive,
            O4: CubePrimitive, O5: CubePrimitive, O6: CubePrimitive, O7: CubePrimitive,
            O8: CubePrimitive, O9: CubePrimitive, O10: CubePrimitive, O11: CubePrimitive,
            Leaves: SharedLeaves
                + MutableLeaves
                + PlaneShuffleLeaves
                + LoadMutPadded12<
                    O0 = O0, O1 = O1, O2 = O2, O3 = O3, O4 = O4, O5 = O5,
                    O6 = O6, O7 = O7, O8 = O8, O9 = O9, O10 = O10, O11 = O11,
                >
                + StorePadded12<
                    O0 = O0, O1 = O1, O2 = O2, O3 = O3, O4 = O4, O5 = O5,
                    O6 = O6, O7 = O7, O8 = O8, O9 = O9, O10 = O10, O11 = O11,
                >
                + Send + Sync + 'static,
            Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
            Expr: $eval<Item, $( $leaf ),+>,
            Op: ReductionOp<Item>,
        >(
            $( $slot: &[$leaf], )+
            read_offsets: &[u32],
            len: &[u32],
            #[comptime] exclusive: bool,
            zero_offsets: &[u32],
            output_offsets: &[u32],
            out0: &mut [O0], out1: &mut [O1], out2: &mut [O2], out3: &mut [O3],
            out4: &mut [O4], out5: &mut [O5], out6: &mut [O6], out7: &mut [O7],
            out8: &mut [O8], out9: &mut [O9], out10: &mut [O10], out11: &mut [O11],
            sum0: &mut [O0], sum1: &mut [O1], sum2: &mut [O2], sum3: &mut [O3],
            sum4: &mut [O4], sum5: &mut [O5], sum6: &mut [O6], sum7: &mut [O7],
            sum8: &mut [O8], sum9: &mut [O9], sum10: &mut [O10], sum11: &mut [O11],
        ) {
            let unit = UNIT_POS as usize;
            let block = CUBE_POS as usize;
            let global = block * BLOCK_SIZE as usize + unit;
            let logical_len = len[0] as usize;
            let safe_global = if global < logical_len { global } else { 0usize };
            scan_value_padded12::<Item, O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, Leaves, Layout, Op>(
                Expr::$method($( $slot, )+ read_offsets, safe_global),
                if global < logical_len { 1u32 } else { 0u32 },
                exclusive,
                logical_len, global, unit, block, zero_offsets, output_offsets,
                out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11,
                sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sum8, sum9, sum10, sum11,
            );
        }
    };
}

define_padded_scan_kernel!(padded_scan_a13,Eval13,eval13; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9,L10:slot10,L11:slot11,L12:slot12]);

#[cubecl::cube(launch_unchecked, explicit_define)]
#[allow(clippy::too_many_arguments)]
fn add_block_prefix_padded12<
    Item: CubeType + Send + Sync + 'static,
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
    Leaves: LoadPadded12<
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
        > + LoadMutPadded12<
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
        > + MutableLeaves
        + Send
        + Sync
        + 'static,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
>(
    prefix0: &[O0],
    prefix1: &[O1],
    prefix2: &[O2],
    prefix3: &[O3],
    prefix4: &[O4],
    prefix5: &[O5],
    prefix6: &[O6],
    prefix7: &[O7],
    prefix8: &[O8],
    prefix9: &[O9],
    prefix10: &[O10],
    prefix11: &[O11],
    len: &[u32],
    #[comptime] exclusive: bool,
    prefix_offsets: &[u32],
    output_offsets: &[u32],
    output0: &mut [O0],
    output1: &mut [O1],
    output2: &mut [O2],
    output3: &mut [O3],
    output4: &mut [O4],
    output5: &mut [O5],
    output6: &mut [O6],
    output7: &mut [O7],
    output8: &mut [O8],
    output9: &mut [O9],
    output10: &mut [O10],
    output11: &mut [O11],
) {
    let block = CUBE_POS as usize;
    let index = block * BLOCK_SIZE as usize + UNIT_POS as usize;
    if block > 0usize && index < len[0] as usize {
        let prefix_cells = Leaves::into_cells(Leaves::load_padded(
            prefix0,
            prefix1,
            prefix2,
            prefix3,
            prefix4,
            prefix5,
            prefix6,
            prefix7,
            prefix8,
            prefix9,
            prefix10,
            prefix11,
            prefix_offsets,
            block - 1usize,
        ));
        if exclusive {
            let initial = Layout::recompose(Leaves::load_mut_padded(
                output0,
                output1,
                output2,
                output3,
                output4,
                output5,
                output6,
                output7,
                output8,
                output9,
                output10,
                output11,
                output_offsets,
                0usize,
            ));
            let with_initial = Layout::decompose(Op::apply(
                initial,
                Layout::recompose(Leaves::read(&prefix_cells)),
            ));
            Leaves::store(&prefix_cells, with_initial);
        }
        if exclusive && UNIT_POS == 0u32 {
            Leaves::read(&prefix_cells).store_padded(
                output0,
                output1,
                output2,
                output3,
                output4,
                output5,
                output6,
                output7,
                output8,
                output9,
                output10,
                output11,
                output_offsets,
                index,
            );
        } else {
            let value = Layout::recompose(Leaves::load_mut_padded(
                output0,
                output1,
                output2,
                output3,
                output4,
                output5,
                output6,
                output7,
                output8,
                output9,
                output10,
                output11,
                output_offsets,
                index,
            ));
            Layout::decompose(Op::apply(
                Layout::recompose(Leaves::read(&prefix_cells)),
                value,
            ))
            .store_padded(
                output0,
                output1,
                output2,
                output3,
                output4,
                output5,
                output6,
                output7,
                output8,
                output9,
                output10,
                output11,
                output_offsets,
                index,
            );
        }
    }
}

#[doc(hidden)]
pub trait InclusiveScanDispatch<R, Input, Output, Item, ReadSlots, WriteSlots, Op>
where
    R: Runtime,
{
    fn run(
        exec: &Executor<R>,
        input: &Input,
        op: Op,
        output: &Output,
        exclusive: bool,
    ) -> Result<(), Error>;
}

#[doc(hidden)]
pub trait InclusiveScanPassDispatch<R, Input, Output, Partials, Item, ReadSlots, WriteSlots, Op>
where
    R: Runtime,
{
    fn run_pass(
        exec: &Executor<R>,
        input: &Input,
        output: &Output,
        partials: &Partials,
        exclusive: bool,
    ) -> Result<(), Error>;
}

macro_rules! impl_padded_scan_dispatch {
    (
        $arity:ty, $eval:ident, $kernel:ident, $env:ty;
        [$( $leaf:ident:$index:literal ),+]
    ) => {
        impl<
            R, Input, Output, Partials, Item, Op,
            O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11,
            $( $leaf ),+
        > InclusiveScanPassDispatch<
            R,
            Input,
            Output,
            Partials,
            Item,
            $env,
            Env12<O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11>,
            Op,
        >
            for Dispatch<$arity, S12>
        where
            R: Runtime,
            Item: StorageLayout + Send + Sync + 'static,
            Item::DeviceLayout: Recompose<Item, Leaves = Item::StorageLeaves>,
            Op: ReductionOp<Item>,
            $( $leaf: MStorageElement, )+
            O0: MStorageElement,
            O1: MStorageElement,
            O2: MStorageElement,
            O3: MStorageElement,
            O4: MStorageElement,
            O5: MStorageElement,
            O6: MStorageElement,
            O7: MStorageElement,
            O8: MStorageElement,
            O9: MStorageElement,
            O10: MStorageElement,
            O11: MStorageElement,
            Input: ReadExpression<Item = Item> + LowerReadExpression + StageRead<R, Env0>,
            Input::Slots: PaddedReadSlots<
                L0 = L0, L1 = L1, L2 = L2, L3 = L3, L4 = L4, L5 = L5, L6 = L6,
                L7 = L7, L8 = L8, L9 = L9, L10 = L10, L11 = L11, L12 = L12,
            >,
            Input::DeviceExpr: $eval<Item, $( $leaf ),+>,
            Output: OutputExpression<Item = Item>
                + LowerOutputExpression
                + StageOutput<R, Env0>,
            Output::Slots: PaddedOutputSlots<Leaves = Item::StorageLeaves>,
            Partials: OutputExpression<Item = Item>
                + LowerOutputExpression
                + StageOutput<R, Env0>,
            Partials::Slots: PaddedOutputSlots<Leaves = Item::StorageLeaves>,
            Item::StorageLeaves: SharedLeaves
                + MutableLeaves
                + PlaneShuffleLeaves
                + LoadMutPadded12<
                    O0 = O0, O1 = O1, O2 = O2, O3 = O3, O4 = O4, O5 = O5,
                    O6 = O6, O7 = O7, O8 = O8, O9 = O9, O10 = O10, O11 = O11,
                >
                + StorePadded12<
                    O0 = O0, O1 = O1, O2 = O2, O3 = O3, O4 = O4, O5 = O5,
                    O6 = O6, O7 = O7, O8 = O8, O9 = O9, O10 = O10, O11 = O11,
                >
                + Send + Sync + 'static,
        {
            fn run_pass(
                exec: &Executor<R>,
                input: &Input,
                output: &Output,
                partials: &Partials,
                exclusive: bool,
            ) -> Result<(), Error> {
                let len = input.logical_len()?;
                let output_len = output.logical_len()?;
                if output_len != len {
                    return Err(Error::LengthMismatch { left: len, right: output_len });
                }
                if len == 0 {
                    return Ok(());
                }
                let blocks = len.div_ceil(BLOCK_SIZE as usize);
                let partial_len = partials.logical_len()?;
                if partial_len != blocks {
                    return Err(Error::LengthMismatch { left: blocks, right: partial_len });
                }
                let mut reads = StagedBindings::new();
                input.stage_at(exec.client(), exec.id(), &mut reads)?;
                reads.pad_to_thirteen(exec.client());
                let mut writes = OutputBindings::new();
                output.stage_output(exec.id(), &mut writes)?;
                writes.pad_to_twelve(exec.client());
                let mut partial_bindings = OutputBindings::new();
                partials.stage_output(exec.id(), &mut partial_bindings)?;
                partial_bindings.pad_to_twelve(exec.client());
                let read_offsets = exec.client().create_from_slice(u32::as_bytes(&reads.offsets));
                let write_offsets = exec.client().create_from_slice(u32::as_bytes(&writes.offsets));
                let zero_values = [0u32; 12];
                let zero_offsets = exec.client().create_from_slice(u32::as_bytes(&zero_values));
                let len_handle = input.logical_extent()?.materialize(exec)?;
                unsafe {
                    $kernel::launch_unchecked::<
                        Item,
                        $( $leaf, )+
                        O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11,
                        Item::StorageLeaves,
                        Item::DeviceLayout,
                        Input::DeviceExpr,
                        Op,
                        R,
                    >(
                        exec.client(),
                        cube_count_1d(blocks)?,
                        CubeDim::new_1d(BLOCK_SIZE),
                        $( BufferArg::from_raw_parts(reads.slots[$index].0.clone(), reads.slots[$index].1), )+
                        BufferArg::from_raw_parts(read_offsets, reads.offsets.len()),
                        BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
                        exclusive,
                        BufferArg::from_raw_parts(zero_offsets.clone(), 12),
                        BufferArg::from_raw_parts(write_offsets.clone(), writes.offsets.len()),
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
                        BufferArg::from_raw_parts(partial_bindings.slots[0].0.clone(), partial_bindings.slots[0].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[1].0.clone(), partial_bindings.slots[1].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[2].0.clone(), partial_bindings.slots[2].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[3].0.clone(), partial_bindings.slots[3].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[4].0.clone(), partial_bindings.slots[4].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[5].0.clone(), partial_bindings.slots[5].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[6].0.clone(), partial_bindings.slots[6].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[7].0.clone(), partial_bindings.slots[7].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[8].0.clone(), partial_bindings.slots[8].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[9].0.clone(), partial_bindings.slots[9].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[10].0.clone(), partial_bindings.slots[10].1),
                        BufferArg::from_raw_parts(partial_bindings.slots[11].0.clone(), partial_bindings.slots[11].1),
                    );
                }
                Ok(())
            }
        }
    };
}

impl_padded_scan_dispatch!(A13,Eval13,padded_scan_a13,Env13<L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12>; [L0:0,L1:1,L2:2,L3:3,L4:4,L5:5,L6:6,L7:7,L8:8,L9:9,L10:10,L11:11,L12:12]);

fn scan_pass<R, Input, Output, Partials, Item, Op>(
    exec: &Executor<R>,
    input: &Input,
    output: &Output,
    partials: &Partials,
    exclusive: bool,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression<Item = Item>
        + LowerReadExpression<Slots: PaddedReadSlots>
        + StageRead<R, Env0>,
    Output: OutputExpression<Item = Item>
        + LowerOutputExpression<Slots: PaddedOutputSlots>
        + StageOutput<R, Env0>,
    Partials: OutputExpression<Item = Item>
        + LowerOutputExpression<Slots: PaddedOutputSlots>
        + StageOutput<R, Env0>,
    Item: StorageLayout,
    Op: ReductionOp<Item>,
    Dispatch<A13, S12>: InclusiveScanPassDispatch<
            R,
            Input,
            Output,
            Partials,
            Item,
            KernelReadSlots<Input::Slots>,
            crate::output::KernelOutputSlots<Output::Slots>,
            Op,
        >,
{
    <Dispatch<A13, S12> as InclusiveScanPassDispatch<
        R,
        Input,
        Output,
        Partials,
        Item,
        KernelReadSlots<Input::Slots>,
        crate::output::KernelOutputSlots<Output::Slots>,
        Op,
    >>::run_pass(exec, input, output, partials, exclusive)
}

fn add_fixed_prefixes<R, Output, Item, Op>(
    exec: &Executor<R>,
    prefixes: &FixedScanStorage<R, Item>,
    output: &Output,
    len: usize,
    extent: &crate::extent::LogicalExtent,
    exclusive: bool,
) -> Result<(), Error>
where
    R: Runtime,
    Item: crate::allocation::ScratchStorage<R>,
    Op: ReductionOp<Item>,
    Output: OutputExpression<Item = Item>
        + LowerOutputExpression<Slots: PaddedOutputSlots<Leaves = Item::StorageLeaves>>
        + StageOutput<R, Env0>,
{
    if len == 0 {
        return Ok(());
    }
    let blocks = len.div_ceil(BLOCK_SIZE as usize);
    let prefix_len = RowStorage::len(prefixes)?;
    if prefix_len != blocks {
        return Err(Error::LengthMismatch {
            left: blocks,
            right: prefix_len,
        });
    }

    let prefix_read = prefixes.read();
    let mut prefix_bindings = StagedBindings::new();
    prefix_read.stage_at(exec.client(), exec.id(), &mut prefix_bindings)?;
    prefix_bindings.pad_to_thirteen(exec.client());
    let mut output_bindings = OutputBindings::new();
    output.stage_output(exec.id(), &mut output_bindings)?;
    output_bindings.pad_to_twelve(exec.client());

    let prefix_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&prefix_bindings.offsets));
    let output_offsets = exec
        .client()
        .create_from_slice(u32::as_bytes(&output_bindings.offsets));
    let len_handle = extent.materialize(exec)?;

    unsafe {
        add_block_prefix_padded12::launch_unchecked::<
            Item,
            <Item::StorageLeaves as StorePadded12>::O0,
            <Item::StorageLeaves as StorePadded12>::O1,
            <Item::StorageLeaves as StorePadded12>::O2,
            <Item::StorageLeaves as StorePadded12>::O3,
            <Item::StorageLeaves as StorePadded12>::O4,
            <Item::StorageLeaves as StorePadded12>::O5,
            <Item::StorageLeaves as StorePadded12>::O6,
            <Item::StorageLeaves as StorePadded12>::O7,
            <Item::StorageLeaves as StorePadded12>::O8,
            <Item::StorageLeaves as StorePadded12>::O9,
            <Item::StorageLeaves as StorePadded12>::O10,
            <Item::StorageLeaves as StorePadded12>::O11,
            Item::StorageLeaves,
            Item::DeviceLayout,
            Op,
            R,
        >(
            exec.client(),
            cube_count_1d(blocks)?,
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[0].0.clone(),
                prefix_bindings.slots[0].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[1].0.clone(),
                prefix_bindings.slots[1].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[2].0.clone(),
                prefix_bindings.slots[2].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[3].0.clone(),
                prefix_bindings.slots[3].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[4].0.clone(),
                prefix_bindings.slots[4].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[5].0.clone(),
                prefix_bindings.slots[5].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[6].0.clone(),
                prefix_bindings.slots[6].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[7].0.clone(),
                prefix_bindings.slots[7].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[8].0.clone(),
                prefix_bindings.slots[8].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[9].0.clone(),
                prefix_bindings.slots[9].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[10].0.clone(),
                prefix_bindings.slots[10].1,
            ),
            BufferArg::from_raw_parts(
                prefix_bindings.slots[11].0.clone(),
                prefix_bindings.slots[11].1,
            ),
            BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
            exclusive,
            BufferArg::from_raw_parts(prefix_offsets, prefix_bindings.offsets.len()),
            BufferArg::from_raw_parts(output_offsets, output_bindings.offsets.len()),
            BufferArg::from_raw_parts(
                output_bindings.slots[0].0.clone(),
                output_bindings.slots[0].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[1].0.clone(),
                output_bindings.slots[1].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[2].0.clone(),
                output_bindings.slots[2].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[3].0.clone(),
                output_bindings.slots[3].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[4].0.clone(),
                output_bindings.slots[4].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[5].0.clone(),
                output_bindings.slots[5].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[6].0.clone(),
                output_bindings.slots[6].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[7].0.clone(),
                output_bindings.slots[7].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[8].0.clone(),
                output_bindings.slots[8].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[9].0.clone(),
                output_bindings.slots[9].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[10].0.clone(),
                output_bindings.slots[10].1,
            ),
            BufferArg::from_raw_parts(
                output_bindings.slots[11].0.clone(),
                output_bindings.slots[11].1,
            ),
        );
    }
    Ok(())
}

fn scan_fixed_storage<R, Item, Op>(
    exec: &Executor<R>,
    input: &FixedScanStorage<R, Item>,
    output: &mut FixedScanStorage<R, Item>,
) -> Result<(), Error>
where
    R: Runtime,
    Item: crate::allocation::ScratchStorage<R>,
    Op: ReductionOp<Item>,
    Dispatch<A13, S12>: InclusiveScanPassDispatch<
            R,
            FixedScanRead<R, Item>,
            FixedScanOutput<R, Item>,
            FixedScanOutput<R, Item>,
            Item,
            KernelReadSlots<<FixedScanRead<R, Item> as LowerReadExpression>::Slots>,
            crate::output::KernelOutputSlots<
                <FixedScanOutput<R, Item> as LowerOutputExpression>::Slots,
            >,
            Op,
        >,
{
    let len = RowStorage::len(input)?;
    let output_len = RowStorage::len(output)?;
    if output_len != len {
        return Err(Error::LengthMismatch {
            left: len,
            right: output_len,
        });
    }
    if len == 0 {
        return Ok(());
    }

    let extent = RowStorage::logical_extent(input);
    RowStorage::set_logical_extent(output, extent.clone());
    let blocks = len.div_ceil(BLOCK_SIZE as usize);
    let mut partials = Item::alloc_scratch(exec, blocks);
    RowStorage::set_logical_extent(
        &mut partials,
        extent.ceil_div(exec, BLOCK_SIZE as usize, blocks)?,
    );
    let input_read = FixedScanRead::<R, Item>::new(input.read());
    let output_write = output.write();
    let partial_write = partials.write();
    scan_pass::<R, _, _, _, Item, Op>(exec, &input_read, &output_write, &partial_write, false)?;

    if blocks > 1 {
        let mut prefixes = Item::alloc_scratch(exec, blocks);
        scan_fixed_storage::<R, Item, Op>(exec, &partials, &mut prefixes)?;
        add_fixed_prefixes::<R, _, Item, Op>(exec, &prefixes, &output_write, len, &extent, false)?;
    }
    Ok(())
}

impl<R, Input, Output, Item, Op, ReadSlots, WriteSlots>
    InclusiveScanDispatch<R, Input, Output, Item, ReadSlots, WriteSlots, Op> for Dispatch<A13, S12>
where
    R: Runtime,
    Input: ReadExpression<Item = Item> + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression<Item = Item>
        + LowerOutputExpression<Slots: PaddedOutputSlots<Leaves = Item::StorageLeaves>>
        + StageOutput<R, Env0>,
    Item: crate::allocation::ScratchStorage<R>,
    Op: ReductionOp<Item>,
    Dispatch<A13, S12>: InclusiveScanPassDispatch<
            R,
            Input,
            Output,
            FixedScanOutput<R, Item>,
            Item,
            ReadSlots,
            WriteSlots,
            Op,
        > + InclusiveScanPassDispatch<
            R,
            FixedScanRead<R, Item>,
            FixedScanOutput<R, Item>,
            FixedScanOutput<R, Item>,
            Item,
            KernelReadSlots<<FixedScanRead<R, Item> as LowerReadExpression>::Slots>,
            crate::output::KernelOutputSlots<
                <FixedScanOutput<R, Item> as LowerOutputExpression>::Slots,
            >,
            Op,
        >,
{
    fn run(
        exec: &Executor<R>,
        input: &Input,
        _op: Op,
        output: &Output,
        exclusive: bool,
    ) -> Result<(), Error> {
        let len = input.logical_len()?;
        let output_len = output.logical_len()?;
        if output_len != len {
            return Err(Error::LengthMismatch {
                left: len,
                right: output_len,
            });
        }
        if len == 0 {
            return Ok(());
        }

        let extent = input.logical_extent()?;
        let blocks = len.div_ceil(BLOCK_SIZE as usize);
        let mut partials = Item::alloc_scratch(exec, blocks);
        RowStorage::set_logical_extent(
            &mut partials,
            extent.ceil_div(exec, BLOCK_SIZE as usize, blocks)?,
        );
        let partial_write = partials.write();
        <Dispatch<A13, S12> as InclusiveScanPassDispatch<
            R,
            Input,
            Output,
            FixedScanOutput<R, Item>,
            Item,
            ReadSlots,
            WriteSlots,
            Op,
        >>::run_pass(exec, input, output, &partial_write, exclusive)?;

        if blocks > 1 {
            let mut prefixes = Item::alloc_scratch(exec, blocks);
            scan_fixed_storage::<R, Item, Op>(exec, &partials, &mut prefixes)?;
            add_fixed_prefixes::<R, _, Item, Op>(exec, &prefixes, output, len, &extent, exclusive)?;
        }
        Ok(())
    }
}

/// Computes an inclusive scan into preallocated output storage.
pub(crate) fn inclusive_scan<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression<Item = Output::Item> + LowerReadExpression + StageRead<R, Env0>,
    Op: ReductionOp<Input::Item>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots: PaddedOutputSlots,
    Dispatch<A13, S12>: InclusiveScanDispatch<
            R,
            Input,
            Output,
            Input::Item,
            KernelReadSlots<Input::Slots>,
            crate::output::KernelOutputSlots<Output::Slots>,
            Op,
        >,
{
    <Dispatch<A13, S12> as InclusiveScanDispatch<
        R,
        Input,
        Output,
        Input::Item,
        KernelReadSlots<Input::Slots>,
        crate::output::KernelOutputSlots<Output::Slots>,
        Op,
    >>::run(exec, &input, op, &output, false)
}

/// Computes adjacent reductions while preserving the first input item.
pub(crate) fn adjacent_difference<R, Input, Output, Op>(
    exec: &Executor<R>,
    input: Input,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression<Item = Output::Item>,
    Op: ReductionOp<Output::Item>,
    Adjacent<Input, Op>:
        ReadExpression<Item = Output::Item> + LowerReadExpression + StageRead<R, Env0>,
    Output: OutputExpression + LowerOutputExpression + StageOutput<R, Env0>,
    Output::Slots: PaddedOutputSlots<Leaves = <Output::Item as StorageLayout>::StorageLeaves>,
    <Output::Item as StorageLayout>::StorageLeaves: StorePadded12,
    <<Output::Item as StorageLayout>::StorageLeaves as CubeType>::ExpandType: StorePadded12Expand,
{
    materialize(exec, Adjacent::new(input, op), output)
}

/// Computes an exclusive scan into preallocated output storage.
pub(crate) fn exclusive_scan<R, Input, Output, Item, Op>(
    exec: &Executor<R>,
    input: Input,
    init: FixedScanStorage<R, Item>,
    op: Op,
    output: Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression<Item = Item> + LowerReadExpression + StageRead<R, Env0>,
    Item: crate::allocation::ScratchStorage<R>,
    FixedScanRead<R, Item>: crate::indexed::GatherInput<R, crate::Constant<u32>, Output>,
    Op: ReductionOp<Item>,
    Output: OutputExpression<Item = Item>
        + LowerOutputExpression
        + StageOutput<R, Env0>
        + SliceOutput
        + FillOutput<R>,
    Output::Slots: PaddedOutputSlots,
    Dispatch<A13, S12>: InclusiveScanDispatch<
            R,
            Input,
            Output,
            Item,
            KernelReadSlots<Input::Slots>,
            crate::output::KernelOutputSlots<Output::Slots>,
            Op,
        >,
{
    let len = input.logical_len()?;
    let output_len = output.logical_len()?;
    if output_len != len {
        return Err(Error::LengthMismatch {
            left: len,
            right: output_len,
        });
    }
    if len > 0 {
        crate::indexed::GatherInput::gather(
            crate::read::FixedRead::new(init.read()),
            exec,
            crate::Constant::new(0, 1),
            output.slice_output(..1),
        )?;
    }
    <Dispatch<A13, S12> as InclusiveScanDispatch<
        R,
        Input,
        Output,
        Item,
        KernelReadSlots<Input::Slots>,
        crate::output::KernelOutputSlots<Output::Slots>,
        Op,
    >>::run(exec, &input, op, &output, true)
}

pub(crate) fn inclusive_scan_u32<R: Runtime>(
    exec: &Executor<R>,
    input: &DeviceVec<R, u32>,
) -> Result<DeviceVec<R, u32>, Error> {
    if input.capacity() == 0 {
        return Ok(exec.alloc_row::<u32>(0));
    }
    let len = input.capacity();
    let blocks = len.div_ceil(BLOCK_SIZE as usize);
    let extent = input.logical_extent();
    let mut output = exec.alloc_row::<u32>(len);
    output.set_logical_extent(extent.clone());
    let mut block_sums = exec.alloc_row::<u32>(blocks);
    block_sums.set_logical_extent(extent.ceil_div(exec, BLOCK_SIZE as usize, blocks)?);
    let len_handle = extent.materialize(exec)?;
    let count = cube_count_1d(blocks)?;
    unsafe {
        u32_block_inclusive_scan_kernel::launch_unchecked::<R>(
            exec.client(),
            count.clone(),
            CubeDim::new_1d(BLOCK_SIZE),
            BufferArg::from_raw_parts(input.handle.clone(), len),
            BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
            BufferArg::from_raw_parts(output.handle.clone(), len),
            BufferArg::from_raw_parts(block_sums.handle.clone(), blocks),
        );
    }
    if blocks > 1 {
        let prefixes = inclusive_scan_u32(exec, &block_sums)?;
        unsafe {
            u32_add_block_prefix_kernel::launch_unchecked::<R>(
                exec.client(),
                count,
                CubeDim::new_1d(BLOCK_SIZE),
                BufferArg::from_raw_parts(prefixes.handle.clone(), blocks),
                BufferArg::from_raw_parts(len_handle.handle.clone(), 1),
                BufferArg::from_raw_parts(output.handle.clone(), len),
            );
        }
    }
    Ok(output)
}

pub(crate) fn last_u32<R: Runtime>(
    exec: &Executor<R>,
    input: &DeviceVec<R, u32>,
) -> Result<DeviceVec<R, u32>, Error> {
    let output = exec.alloc_row::<u32>(1);
    let len = input.logical_extent().materialize(exec)?;
    unsafe {
        copy_last_kernel::launch_unchecked::<R>(
            exec.client(),
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            BufferArg::from_raw_parts(input.handle.clone(), input.capacity()),
            BufferArg::from_raw_parts(len.handle.clone(), 1),
            BufferArg::from_raw_parts(output.handle.clone(), 1),
        );
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Counting, Permute, RowStorage, Transform, Zip, op::UnaryOp};
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    #[test]
    fn inclusive_u32_scan_crosses_block_and_recursive_boundaries() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&vec![1_u32; 70_001]);
        let output = inclusive_scan_u32(&exec, &input).unwrap();
        let actual = exec.to_host(&output).unwrap();
        assert_eq!(actual[0], 1);
        assert_eq!(actual[255], 256);
        assert_eq!(actual[256], 257);
        assert_eq!(actual[65_535], 65_536);
        assert_eq!(actual[70_000], 70_001);
        assert_eq!(
            exec.to_host(&last_u32(&exec, &output).unwrap()).unwrap(),
            vec![70_001]
        );
    }

    struct Sum;

    #[cubecl::cube]
    impl ReductionOp<u32> for Sum {
        fn apply(lhs: u32, rhs: u32) -> u32 {
            lhs + rhs
        }
    }

    struct TakeLeft;

    #[cubecl::cube]
    impl ReductionOp<u32> for TakeLeft {
        fn apply(lhs: u32, _rhs: u32) -> u32 {
            lhs
        }
    }

    struct SumPair;

    #[cubecl::cube]
    impl ReductionOp<(u32, u32)> for SumPair {
        fn apply(lhs: (u32, u32), rhs: (u32, u32)) -> (u32, u32) {
            (lhs.0 + rhs.0, lhs.1 + rhs.1)
        }
    }

    #[test]
    fn inclusive_pair_scan_crosses_recursive_block_boundary() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let len = 70_001;
        let left = exec.to_device(&vec![1_u32; len]);
        let right = exec.to_device(&vec![2_u32; len]);
        let output = exec.alloc_row::<(u32, u32)>(len);

        inclusive_scan(
            &exec,
            Zip::new(left.column(), right.column()),
            SumPair,
            output.write(),
        )
        .unwrap();

        let actual_left = exec.to_host(&output.0).unwrap();
        let actual_right = exec.to_host(&output.1).unwrap();
        for &index in &[0, 255, 256, 65_535, 70_000] {
            assert_eq!(actual_left[index], index as u32 + 1);
            assert_eq!(actual_right[index], 2 * (index as u32 + 1));
        }
    }

    type Seven = (u32, u32, u32, u32, u32, u32, u32);
    struct SumSeven;

    #[cubecl::cube]
    impl UnaryOp<Seven> for SumSeven {
        type Output = u32;
        fn apply(input: Seven) -> u32 {
            input.0 + input.1 + input.2 + input.3 + input.4 + input.5 + input.6
        }
    }

    struct SumSevenItems;

    #[cubecl::cube]
    impl ReductionOp<Seven> for SumSevenItems {
        fn apply(lhs: Seven, rhs: Seven) -> Seven {
            (
                lhs.0 + rhs.0,
                lhs.1 + rhs.1,
                lhs.2 + rhs.2,
                lhs.3 + rhs.3,
                lhs.4 + rhs.4,
                lhs.5 + rhs.5,
                lhs.6 + rhs.6,
            )
        }
    }

    #[test]
    fn inclusive_s7_scan_dispatches_eval8_and_normalizes_output_shape() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let columns: Vec<_> = (1_u32..=7)
            .map(|value| exec.to_device(&vec![value; 600]))
            .collect();
        let seven = Zip::new(
            columns[0].column(),
            Zip::new(
                columns[1].column(),
                Zip::new(
                    columns[2].column(),
                    Zip::new(
                        columns[3].column(),
                        Zip::new(
                            columns[4].column(),
                            Zip::new(columns[5].column(), columns[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let input = Permute::new(seven, Counting::new(0, 600));
        let output = exec.alloc_row::<Seven>(600);

        inclusive_scan(&exec, input, SumSevenItems, output.write()).unwrap();

        let (a, b, c, d, e, f, g) = crate::MStorage::into_columns(output);
        assert_eq!(exec.to_host(&a).unwrap()[599], 600);
        assert_eq!(exec.to_host(&b).unwrap()[599], 1_200);
        assert_eq!(exec.to_host(&c).unwrap()[599], 1_800);
        assert_eq!(exec.to_host(&d).unwrap()[599], 2_400);
        assert_eq!(exec.to_host(&e).unwrap()[599], 3_000);
        assert_eq!(exec.to_host(&f).unwrap()[599], 3_600);
        assert_eq!(exec.to_host(&g).unwrap()[599], 4_200);
    }

    #[test]
    fn inclusive_scalar_scan_dispatches_eval8() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let columns: Vec<_> = (0..7).map(|_| exec.to_device(&[1_u32; 600])).collect();
        let seven = Zip::new(
            columns[0].column(),
            Zip::new(
                columns[1].column(),
                Zip::new(
                    columns[2].column(),
                    Zip::new(
                        columns[3].column(),
                        Zip::new(
                            columns[4].column(),
                            Zip::new(columns[5].column(), columns[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let input = Transform::new(Permute::new(seven, Counting::new(0, 600)), SumSeven);
        let output = exec.to_device(&[0_u32; 600]);
        inclusive_scan(&exec, input, Sum, output.slice_mut(..)).unwrap();
        let actual = exec.to_host(&output).unwrap();
        assert_eq!(actual[0], 7);
        assert_eq!(actual[255], 7 * 256);
        assert_eq!(actual[256], 7 * 257);
        assert_eq!(actual[599], 7 * 600);
    }

    #[test]
    fn exclusive_scalar_scan_applies_init_once() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[1_u32, 2, 3, 4]);
        let output = exec.to_device(&[99_u32; 6]);
        let init = exec.value(10_u32).unwrap();
        exclusive_scan(
            &exec,
            input.column(),
            init.into_scratch_storage(),
            Sum,
            output.slice_mut(1..5),
        )
        .unwrap();
        assert_eq!(exec.to_host(&output).unwrap(), vec![99, 10, 11, 13, 16, 99]);
    }

    #[test]
    fn exclusive_scan_crosses_recursive_boundaries_in_operand_order() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let len = 70_001usize;
        let input = exec.to_device(&vec![1_u32; len]);
        let output = exec.to_device(&vec![0_u32; len]);
        let init = exec.value(10_u32).unwrap();
        exclusive_scan(
            &exec,
            input.column(),
            init.into_scratch_storage(),
            Sum,
            output.slice_mut(..),
        )
        .unwrap();
        let actual = exec.to_host(&output).unwrap();
        for &index in &[0, 255, 256, 65_535, 70_000] {
            assert_eq!(actual[index], 10 + index as u32);
        }

        let ordered = exec.to_device(&vec![0_u32; 600]);
        let init = exec.value(42_u32).unwrap();
        exclusive_scan(
            &exec,
            input.slice(..600),
            init.into_scratch_storage(),
            TakeLeft,
            ordered.slice_mut(..),
        )
        .unwrap();
        assert_eq!(exec.to_host(&ordered).unwrap(), vec![42; 600]);
    }

    #[test]
    fn adjacent_difference_is_a_regular_fused_read_expression() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let input = exec.to_device(&[1_u32, 3, 6, 10]);
        let output = exec.to_device(&[0_u32; 4]);
        adjacent_difference(&exec, input.column(), Sum, output.slice_mut(..)).unwrap();
        assert_eq!(exec.to_host(&output).unwrap(), vec![1, 4, 9, 16]);
    }

    #[test]
    fn exclusive_storage7_accepts_eval8_and_preserves_semantic_init() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let columns: Vec<_> = (1_u32..=7)
            .map(|value| exec.to_device(&[value; 4]))
            .collect();
        let seven = Zip::new(
            columns[0].column(),
            Zip::new(
                columns[1].column(),
                Zip::new(
                    columns[2].column(),
                    Zip::new(
                        columns[3].column(),
                        Zip::new(
                            columns[4].column(),
                            Zip::new(columns[5].column(), columns[6].column()),
                        ),
                    ),
                ),
            ),
        );
        let input = Permute::new(seven, Counting::new(0, 4));
        let output = exec.alloc_row::<Seven>(4);
        let init: Seven = (10, 20, 30, 40, 50, 60, 70);
        let init = exec.value(init).unwrap();
        exclusive_scan(
            &exec,
            input,
            init.into_scratch_storage(),
            SumSevenItems,
            output.write(),
        )
        .unwrap();

        let (first, _, _, _, _, _, last) = crate::MStorage::into_columns(output);
        assert_eq!(exec.to_host(&first).unwrap(), vec![10, 11, 12, 13]);
        assert_eq!(exec.to_host(&last).unwrap(), vec![70, 77, 84, 91]);
    }

    #[test]
    fn scan_rejects_mismatched_output_tree_and_foreign_storage() {
        let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let left = exec.to_device(&[1_u32, 2, 3]);
        let right = exec.to_device(&[4_u32, 5, 6]);
        let out_left = exec.to_device(&[0_u32; 3]);
        let out_right = exec.to_device(&[0_u32; 2]);
        assert_eq!(
            inclusive_scan(
                &exec,
                Zip::new(left.column(), right.column()),
                SumPair,
                Zip::new(out_left.slice_mut(..), out_right.slice_mut(..)),
            ),
            Err(Error::LengthMismatch { left: 3, right: 2 })
        );

        let other = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
        let foreign_output = other.to_device(&[0_u32; 3]);
        assert_eq!(
            inclusive_scan(&exec, left.column(), Sum, foreign_output.slice_mut(..)),
            Err(Error::ForeignExecutor)
        );
    }
}
