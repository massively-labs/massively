//! Read-arity and storage-arity indexed reductions.

#![allow(private_interfaces)]

use core::marker::PhantomData;
use cubecl::prelude::*;

use crate::allocation::CopyStorage;
use crate::{
    A13, Column, Constant, Counting, DivModCounting, Error, Executor, MStorageElement, Permute,
    ReadExpression, ReverseCounting, RowStorage, S12, StorageLayout, Stride, Taken, Transform, Zip,
    eval::Eval13,
    launch::cube_count_1d,
    op::UnaryOp,
    output::OutputBindings,
    read::{
        BindSlots, Env0, Env1, Env2, Env3, Env4, Env5, Env6, Env7, Env8, Env9, Env10, Env11, Env12,
        Env13, LowerReadExpression, PaddedReadSlots, TakenSource,
    },
    storage::{
        Decompose, MutableLeaves, MutableLeavesExpand, PlaneShuffleLeaves, Recompose, SharedLeaves,
        SharedLeavesExpand, StorePadded12, StorePadded12Expand,
    },
};

type FixedReduceStorage<R, Item> = <Item as crate::core::allocation::ScratchStorage<R>>::Storage;
type FixedReduceRead<R, Item> =
    crate::read::FixedRead<<FixedReduceStorage<R, Item> as crate::RowStorage<R>>::Read>;
type FixedReduceOutput<R, Item> = <FixedReduceStorage<R, Item> as crate::RowStorage<R>>::Write;

const BLOCK_SIZE: u32 = 256;
const ITEMS_PER_UNIT: usize = 256;
const TILE_SIZE: usize = BLOCK_SIZE as usize * ITEMS_PER_UNIT;

/// Associative binary operation used by scans and reductions.
///
/// Scans preserve operand order. Reductions may regroup and reorder operands
/// across GPU units, so operations passed to `reduce` must also be commutative.
///
/// # Examples
///
/// ```
/// use cubecl::prelude::*;
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Executor, MVal, op, vector::reduce};
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
/// let input = exec.to_device(&[1_u32, 2, 3]);
///
/// let init = 0_u32;
/// let sum = reduce(&exec, input.slice(..), init, Add).unwrap();
/// assert_eq!(sum.read(&exec).unwrap(), 6);
/// ```
#[cubecl::cube]
pub trait ReductionOp<Item: CubeType>: 'static + Send + Sync {
    fn apply(lhs: Item, rhs: Item) -> Item;
}

/// Semantic items supported by the first single-leaf reduction slice.

/// Dispatch key combining read arity and reduction storage arity.
#[derive(Clone, Copy, Debug, Default)]
pub struct Dispatch<Read, Storage>(PhantomData<fn() -> (Read, Storage)>);

#[cubecl::cube]
fn accumulate_register<Item, Leaves, Layout, Op>(cells: &Leaves::Cells, value: Item)
where
    Item: CubeType,
    Leaves: MutableLeaves,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
{
    let accumulated = Op::apply(Layout::recompose(Leaves::read(cells)), value);
    Leaves::store(cells, Layout::decompose(accumulated));
}

#[cubecl::cube]
fn reduce_plane_full<Item, Leaves, Layout, Op>(value: Item) -> Leaves
where
    Item: CubeType,
    Leaves: MutableLeaves + PlaneShuffleLeaves,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
{
    let cells = Layout::decompose(value).into_cells();
    let offset = RuntimeCell::<u32>::new(PLANE_DIM / 2u32);
    while offset.read() > 0u32 {
        let right = Leaves::shuffle_leaves_down(Leaves::read(&cells), offset.read());
        if UNIT_POS_PLANE < offset.read() {
            let combined = Op::apply(
                Layout::recompose(Leaves::read(&cells)),
                Layout::recompose(right),
            );
            Leaves::store(&cells, Layout::decompose(combined));
        }
        offset.store(offset.read() / 2u32);
    }
    Leaves::read(&cells)
}

#[cubecl::cube]
fn reduce_plane_valid<Item, Leaves, Layout, Op>(value: Item, valid: u32) -> (Leaves, u32)
where
    Item: CubeType,
    Leaves: MutableLeaves + PlaneShuffleLeaves,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
{
    let cells = Layout::decompose(value).into_cells();
    let cell_valid = RuntimeCell::<u32>::new(valid);
    let offset = RuntimeCell::<u32>::new(PLANE_DIM / 2u32);
    while offset.read() > 0u32 {
        let right = Leaves::shuffle_leaves_down(Leaves::read(&cells), offset.read());
        let right_cells = right.into_cells();
        let right_valid = plane_shuffle_down(cell_valid.read(), offset.read());
        if UNIT_POS_PLANE < offset.read() && right_valid != 0u32 {
            if cell_valid.read() != 0u32 {
                let combined = Op::apply(
                    Layout::recompose(Leaves::read(&cells)),
                    Layout::recompose(Leaves::read(&right_cells)),
                );
                Leaves::store(&cells, Layout::decompose(combined));
            } else {
                Leaves::store(&cells, Leaves::read(&right_cells));
                cell_valid.store(1u32);
            }
        }
        offset.store(offset.read() / 2u32);
    }
    (Leaves::read(&cells), cell_valid.read())
}

#[doc(hidden)]
pub struct StagedBindings {
    pub(crate) slots: Vec<(cubecl::server::Handle, usize)>,
    pub(crate) offsets: Vec<u32>,
}

impl StagedBindings {
    pub(crate) fn new() -> Self {
        Self {
            slots: Vec::new(),
            offsets: Vec::new(),
        }
    }

    fn push(&mut self, handle: cubecl::server::Handle, len: usize, offset: u32) {
        self.slots.push((handle, len));
        self.offsets.push(offset);
    }

    /// Pads the staged read ABI to thirteen buffers. The shared dummy buffer
    /// is never indexed by expressions whose corresponding slots are unused.
    pub(crate) fn pad_to_thirteen<R: Runtime>(&mut self, client: &ComputeClient<R>) {
        debug_assert!(self.slots.len() <= 13);
        if self.slots.len() == 13 {
            return;
        }
        let dummy = client.empty(core::mem::size_of::<u32>());
        while self.slots.len() < 13 {
            self.slots.push((dummy.clone(), 1));
            self.offsets.push(0);
        }
    }
}

/// Host-side staging following the same left-first recursion as [`BindSlots`].
#[doc(hidden)]
pub trait StageRead<R: Runtime, Env>: BindSlots<Env> {
    fn logical_len(&self) -> Result<usize, Error>;
    fn logical_extent(&self) -> Result<crate::extent::LogicalExtent, Error> {
        Ok(crate::extent::LogicalExtent::fixed(self.logical_len()?))
    }
    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error>;
}

macro_rules! impl_leaf_staging {
    (impl <$( $env_ty:ident ),*> $env:ty) => {
        impl<R, T, $( $env_ty ),*> StageRead<R, $env> for Column<T>
        where
            R: Runtime,
            T: MStorageElement,
            Column<T>: BindSlots<$env>,
        {
            fn logical_len(&self) -> Result<usize, Error> {
                Ok(self.len)
            }

            fn logical_extent(&self) -> Result<crate::extent::LogicalExtent, Error> {
                Ok(self.extent.clone())
            }

            fn stage_at(
                &self,
                _client: &ComputeClient<R>,
                owner: u64,
                bindings: &mut StagedBindings,
            ) -> Result<(), Error> {
                if self.owner != Some(owner) {
                    return Err(Error::ForeignExecutor);
                }
                let handle = self.handle.clone().ok_or(Error::UnboundColumn)?;
                bindings.push(handle, self.buffer_len, self.offset);
                Ok(())
            }
        }

        impl<R, T, $( $env_ty ),*> StageRead<R, $env> for Constant<T>
        where
            R: Runtime,
            T: MStorageElement,
            Constant<T>: BindSlots<$env>,
        {
            fn logical_len(&self) -> Result<usize, Error> {
                Ok(self.len)
            }

            fn stage_at(
                &self,
                client: &ComputeClient<R>,
                _owner: u64,
                bindings: &mut StagedBindings,
            ) -> Result<(), Error> {
                let handle = client.create_from_slice(T::as_bytes(&[self.value]));
                bindings.push(handle, 1, 0);
                Ok(())
            }
        }

        impl<R, T, $( $env_ty ),*> StageRead<R, $env> for crate::read::Value<T>
        where
            R: Runtime,
            T: MStorageElement,
            crate::read::Value<T>: BindSlots<$env>,
        {
            fn logical_len(&self) -> Result<usize, Error> {
                Ok(1)
            }

            fn stage_at(
                &self,
                client: &ComputeClient<R>,
                owner: u64,
                bindings: &mut StagedBindings,
            ) -> Result<(), Error> {
                let handle = match self {
                    crate::read::Value::Host(value) => {
                        client.create_from_slice(T::as_bytes(&[*value]))
                    }
                    crate::read::Value::Device {
                        handle,
                        owner: value_owner,
                    } => {
                        if *value_owner != owner {
                            return Err(Error::ForeignExecutor);
                        }
                        handle.clone()
                    }
                };
                bindings.push(handle, 1, 0);
                Ok(())
            }
        }

        impl<R, $( $env_ty ),*> StageRead<R, $env> for Counting
        where
            R: Runtime,
            Counting: BindSlots<$env>,
        {
            fn logical_len(&self) -> Result<usize, Error> {
                Ok(self.len)
            }

            fn stage_at(
                &self,
                client: &ComputeClient<R>,
                _owner: u64,
                bindings: &mut StagedBindings,
            ) -> Result<(), Error> {
                let handle = client.create_from_slice(u32::as_bytes(&[self.start]));
                bindings.push(handle, 1, 0);
                Ok(())
            }
        }

        impl<R, $( $env_ty ),*> StageRead<R, $env> for Stride
        where
            R: Runtime,
            Stride: BindSlots<$env>,
        {
            fn logical_len(&self) -> Result<usize, Error> {
                Ok(self.len)
            }

            fn stage_at(
                &self,
                client: &ComputeClient<R>,
                _owner: u64,
                bindings: &mut StagedBindings,
            ) -> Result<(), Error> {
                let handle = client.create_from_slice(u32::as_bytes(&[self.start, self.step]));
                bindings.push(handle, 2, 0);
                Ok(())
            }
        }

        impl<R, $( $env_ty ),*> StageRead<R, $env> for DivModCounting
        where
            R: Runtime,
            DivModCounting: BindSlots<$env>,
        {
            fn logical_len(&self) -> Result<usize, Error> {
                Ok(self.len)
            }

            fn stage_at(
                &self,
                client: &ComputeClient<R>,
                _owner: u64,
                bindings: &mut StagedBindings,
            ) -> Result<(), Error> {
                let handle = client.create_from_slice(u32::as_bytes(&[
                    self.start,
                    self.divisor,
                    self.modulus,
                ]));
                bindings.push(handle, 3, 0);
                Ok(())
            }
        }

        impl<R, $( $env_ty ),*> StageRead<R, $env> for ReverseCounting
        where
            R: Runtime,
            ReverseCounting: BindSlots<$env>,
        {
            fn logical_len(&self) -> Result<usize, Error> {
                Ok(self.len)
            }

            fn stage_at(
                &self,
                client: &ComputeClient<R>,
                _owner: u64,
                bindings: &mut StagedBindings,
            ) -> Result<(), Error> {
                let start = u32::try_from(self.start).map_err(|_| Error::LengthTooLarge {
                    len: self.start.saturating_add(1),
                })?;
                let handle = client.create_from_slice(u32::as_bytes(&[start]));
                bindings.push(handle, 1, 0);
                Ok(())
            }
        }
    };
}

impl_leaf_staging!(impl <> Env0);
impl_leaf_staging!(impl <L0> Env1<L0>);
impl_leaf_staging!(impl <L0, L1> Env2<L0, L1>);
impl_leaf_staging!(impl <L0, L1, L2> Env3<L0, L1, L2>);
impl_leaf_staging!(impl <L0, L1, L2, L3> Env4<L0, L1, L2, L3>);
impl_leaf_staging!(impl <L0, L1, L2, L3, L4> Env5<L0, L1, L2, L3, L4>);
impl_leaf_staging!(impl <L0, L1, L2, L3, L4, L5> Env6<L0, L1, L2, L3, L4, L5>);
impl_leaf_staging!(impl <L0, L1, L2, L3, L4, L5, L6> Env7<L0, L1, L2, L3, L4, L5, L6>);
impl_leaf_staging!(impl <L0, L1, L2, L3, L4, L5, L6, L7> Env8<L0, L1, L2, L3, L4, L5, L6, L7>);
impl_leaf_staging!(impl <L0, L1, L2, L3, L4, L5, L6, L7, L8> Env9<L0, L1, L2, L3, L4, L5, L6, L7, L8>);
impl_leaf_staging!(impl <L0, L1, L2, L3, L4, L5, L6, L7, L8, L9> Env10<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9>);
impl_leaf_staging!(impl <L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10> Env11<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10>);
impl_leaf_staging!(impl <L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11> Env12<L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11>);

impl<R, Source, Env> StageRead<R, Env> for Taken<Source>
where
    R: Runtime,
    Source: TakenSource,
    Source::Read: StageRead<R, Env>,
    Taken<Source>: BindSlots<Env>,
{
    fn logical_len(&self) -> Result<usize, Error> {
        Ok(self.len as usize)
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error> {
        self.lower().stage_at(client, owner, bindings)
    }
}

impl<R, Left, Right, Env> StageRead<R, Env> for Zip<Left, Right>
where
    R: Runtime,
    Left: StageRead<R, Env>,
    Right: StageRead<R, Left::NextEnv>,
    Zip<Left, Right>: BindSlots<Env>,
{
    fn logical_len(&self) -> Result<usize, Error> {
        let left = self.0.logical_len()?;
        let right = self.1.logical_len()?;
        if left != right {
            return Err(Error::LengthMismatch { left, right });
        }
        Ok(left)
    }

    fn logical_extent(&self) -> Result<crate::extent::LogicalExtent, Error> {
        self.0.logical_extent()?.zipped(&self.1.logical_extent()?)
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error> {
        self.0.stage_at(client, owner, bindings)?;
        self.1.stage_at(client, owner, bindings)
    }
}

impl<R, Values, Offsets, Env> StageRead<R, Env> for crate::seg::SegmentRead<Values, Offsets>
where
    R: Runtime,
    Values: StageRead<R, Env>,
    Offsets: StageRead<R, Values::NextEnv>,
    crate::seg::SegmentRead<Values, Offsets>: BindSlots<Env>,
{
    fn logical_len(&self) -> Result<usize, Error> {
        crate::seg::segment_count(self.offsets().logical_len()?)
    }

    fn logical_extent(&self) -> Result<crate::extent::LogicalExtent, Error> {
        Ok(self
            .offsets()
            .logical_extent()?
            .slice(1, self.logical_len()?))
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error> {
        self.values().stage_at(client, owner, bindings)?;
        self.offsets().stage_at(client, owner, bindings)
    }
}

impl<R, Input, Op, Env> StageRead<R, Env> for Transform<Input, Op>
where
    R: Runtime,
    Input: ReadExpression + StageRead<R, Env>,
    Op: UnaryOp<Input::Item>,
    Transform<Input, Op>: BindSlots<Env>,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.input.logical_len()
    }

    fn logical_extent(&self) -> Result<crate::extent::LogicalExtent, Error> {
        self.input.logical_extent()
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error> {
        self.input.stage_at(client, owner, bindings)
    }
}

impl<R, Input, Op, Env> StageRead<R, Env> for crate::read::IndexedTransform<Input, Op>
where
    R: Runtime,
    Input: ReadExpression + StageRead<R, Env>,
    Op: crate::op::IndexedUnaryOp<Input::Item>,
    crate::read::IndexedTransform<Input, Op>: BindSlots<Env>,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.input.logical_len()
    }

    fn logical_extent(&self) -> Result<crate::extent::LogicalExtent, Error> {
        self.input.logical_extent()
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error> {
        self.input.stage_at(client, owner, bindings)
    }
}

impl<R, Input, Op, Env> StageRead<R, Env> for crate::read::AdjacentIndexedTransform<Input, Op>
where
    R: Runtime,
    Input: ReadExpression + StageRead<R, Env>,
    Op: crate::op::IndexedBinaryOp<Input::Item>,
    crate::read::AdjacentIndexedTransform<Input, Op>: BindSlots<Env>,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.input.logical_len()
    }

    fn logical_extent(&self) -> Result<crate::extent::LogicalExtent, Error> {
        self.input.logical_extent()
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error> {
        self.input.stage_at(client, owner, bindings)
    }
}

impl<R, Input, Op, Env> StageRead<R, Env> for crate::read::Adjacent<Input, Op>
where
    R: Runtime,
    Input: ReadExpression + StageRead<R, Env>,
    Op: ReductionOp<Input::Item>,
    crate::read::Adjacent<Input, Op>: BindSlots<Env>,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.input.logical_len()
    }

    fn logical_extent(&self) -> Result<crate::extent::LogicalExtent, Error> {
        self.input.logical_extent()
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error> {
        self.input.stage_at(client, owner, bindings)
    }
}

impl<R, Input, Env> StageRead<R, Env> for crate::read::Slice<R, Input>
where
    R: Runtime,
    Input: StageRead<R, Env>,
    crate::read::Slice<R, Input>: BindSlots<Env>,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.input.logical_len()
    }

    fn logical_extent(&self) -> Result<crate::extent::LogicalExtent, Error> {
        self.input.logical_extent()
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error> {
        self.input.stage_at(client, owner, bindings)
    }
}

impl<R, Values, Indices, Env> StageRead<R, Env> for Permute<Values, Indices>
where
    R: Runtime,
    Values: StageRead<R, Env>,
    Indices: StageRead<R, Values::NextEnv>,
    Permute<Values, Indices>: BindSlots<Env>,
{
    fn logical_len(&self) -> Result<usize, Error> {
        self.indices.logical_len()
    }

    fn logical_extent(&self) -> Result<crate::extent::LogicalExtent, Error> {
        self.indices.logical_extent()
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error> {
        self.values.stage_at(client, owner, bindings)?;
        self.indices.stage_at(client, owner, bindings)
    }
}

impl<R, Values, Env> StageRead<R, Env> for crate::read::Reverse<Values>
where
    R: Runtime,
    Values: StageRead<R, Env>,
    crate::read::Reverse<Values>: BindSlots<Env>,
{
    fn logical_len(&self) -> Result<usize, Error> {
        match self.len {
            Some(len) => Ok(len),
            None => self.values.logical_len(),
        }
    }

    fn logical_extent(&self) -> Result<crate::extent::LogicalExtent, Error> {
        let capacity = self.logical_len()?;
        Ok(self.values.logical_extent()?.slice(self.offset, capacity))
    }

    fn stage_at(
        &self,
        client: &ComputeClient<R>,
        owner: u64,
        bindings: &mut StagedBindings,
    ) -> Result<(), Error> {
        let exec = crate::Executor::from_client(client, owner);
        let start = self
            .values
            .logical_extent()?
            .reverse_start(&exec, self.offset)?;
        self.values.stage_at(client, owner, bindings)?;
        bindings.push(start.handle.clone(), 1, 0);
        Ok(())
    }
}

#[cubecl::cube]
fn combine_plane_results<Item, Leaves, Layout, Op>(value: Leaves, value_valid: u32) -> (Leaves, u32)
where
    Item: CubeType + Send + Sync + 'static,
    Leaves: SharedLeaves + MutableLeaves + PlaneShuffleLeaves + Send + Sync + 'static,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
{
    let cube_dim = BLOCK_SIZE as usize;
    let mut shared = Leaves::new_shared(cube_dim);
    let mut plane_valid = Shared::<[u32]>::new_slice(cube_dim);
    let result = value.into_cells();
    let result_valid = RuntimeCell::<u32>::new(0u32);
    if UNIT_POS_PLANE == 0u32 {
        Leaves::read(&result).store_shared(&mut shared, PLANE_POS as usize);
        plane_valid[PLANE_POS as usize] = value_valid;
    }
    sync_cube();
    if PLANE_POS == 0u32 {
        let plane_count = (CUBE_DIM + PLANE_DIM - 1u32) / PLANE_DIM;
        let source = if UNIT_POS_PLANE < plane_count {
            UNIT_POS_PLANE as usize
        } else {
            0usize
        };
        let source_valid = if UNIT_POS_PLANE < plane_count {
            plane_valid[source]
        } else {
            0u32
        };
        let accumulator =
            Layout::decompose(Layout::recompose(Leaves::load_shared(&shared, source))).into_cells();
        let accumulator_valid = RuntimeCell::<u32>::new(source_valid);
        let cursor = RuntimeCell::<u32>::new(UNIT_POS_PLANE + PLANE_DIM);
        while cursor.read() < plane_count {
            let index = cursor.read() as usize;
            if plane_valid[index] != 0u32 {
                let next = Leaves::load_shared(&shared, index).into_cells();
                if accumulator_valid.read() != 0u32 {
                    accumulate_register::<Item, Leaves, Layout, Op>(
                        &accumulator,
                        Layout::recompose(Leaves::read(&next)),
                    );
                } else {
                    Leaves::store(&accumulator, Leaves::read(&next));
                    accumulator_valid.store(1u32);
                }
            }
            cursor.store(cursor.read() + PLANE_DIM);
        }
        let block_result = reduce_plane_valid::<Item, Leaves, Layout, Op>(
            Layout::recompose(Leaves::read(&accumulator)),
            accumulator_valid.read(),
        );
        if UNIT_POS_PLANE == 0u32 && block_result.1 != 0u32 {
            Leaves::store(&result, block_result.0);
            result_valid.store(1u32);
        }
    }
    (Leaves::read(&result), result_valid.read())
}

#[cubecl::cube]
#[allow(clippy::too_many_arguments)]
fn finish_reduce_value_padded12<
    Item,
    O0,
    O1,
    O2,
    O3,
    O4,
    O5,
    O6,
    O7,
    O8,
    O9,
    O10,
    O11,
    Leaves,
    Layout,
    Op,
>(
    value: Leaves,
    value_valid: u32,
    zero_offsets: &[u32],
    partial0: &mut [O0],
    partial1: &mut [O1],
    partial2: &mut [O2],
    partial3: &mut [O3],
    partial4: &mut [O4],
    partial5: &mut [O5],
    partial6: &mut [O6],
    partial7: &mut [O7],
    partial8: &mut [O8],
    partial9: &mut [O9],
    partial10: &mut [O10],
    partial11: &mut [O11],
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
        > + Send
        + Sync
        + 'static,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
    Op: ReductionOp<Item>,
{
    let block_result = combine_plane_results::<Item, Leaves, Layout, Op>(value, value_valid);
    if block_result.1 != 0u32 {
        block_result.0.store_padded(
            partial0,
            partial1,
            partial2,
            partial3,
            partial4,
            partial5,
            partial6,
            partial7,
            partial8,
            partial9,
            partial10,
            partial11,
            zero_offsets,
            CUBE_POS as usize,
        );
    }
}

macro_rules! define_padded_reduce_eval_kernel {
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
            zero_offsets: &[u32],
            partial0: &mut [O0], partial1: &mut [O1], partial2: &mut [O2],
            partial3: &mut [O3], partial4: &mut [O4], partial5: &mut [O5],
            partial6: &mut [O6], partial7: &mut [O7], partial8: &mut [O8],
            partial9: &mut [O9], partial10: &mut [O10], partial11: &mut [O11],
        ) {
            let unit = UNIT_POS as usize;
            let cube_dim = BLOCK_SIZE as usize;
            let logical_len = len[0] as usize;
            let tile_start = (CUBE_POS as usize) * TILE_SIZE;
            let first_index = tile_start + unit;
            let safe_index = if logical_len == 0usize {
                0usize
            } else if first_index < logical_len {
                first_index
            } else {
                logical_len - 1usize
            };
            let accumulator = Layout::decompose(
                Expr::$method($( $slot, )+ read_offsets, safe_index),
            ).into_cells();

            if tile_start + TILE_SIZE <= logical_len {
                for item in 1usize..ITEMS_PER_UNIT {
                    let value = Expr::$method(
                        $( $slot, )+ read_offsets, first_index + item * cube_dim,
                    );
                    accumulate_register::<Item, Leaves, Layout, Op>(&accumulator, value);
                }
                let result = reduce_plane_full::<Item, Leaves, Layout, Op>(
                    Layout::recompose(Leaves::read(&accumulator)),
                );
                finish_reduce_value_padded12::<Item, O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, Leaves, Layout, Op>(
                    result, 1u32, zero_offsets,
                    partial0, partial1, partial2, partial3, partial4, partial5,
                    partial6, partial7, partial8, partial9, partial10, partial11,
                );
            } else {
                for item in 1usize..ITEMS_PER_UNIT {
                    let index = first_index + item * cube_dim;
                    if index < logical_len {
                        let value = Expr::$method($( $slot, )+ read_offsets, index);
                        accumulate_register::<Item, Leaves, Layout, Op>(&accumulator, value);
                    }
                }
                let result = reduce_plane_valid::<Item, Leaves, Layout, Op>(
                    Layout::recompose(Leaves::read(&accumulator)),
                    if first_index < logical_len { 1u32 } else { 0u32 },
                );
                finish_reduce_value_padded12::<Item, O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, Leaves, Layout, Op>(
                    result.0, result.1, zero_offsets,
                    partial0, partial1, partial2, partial3, partial4, partial5,
                    partial6, partial7, partial8, partial9, partial10, partial11,
                );
            }
        }
    };
}

define_padded_reduce_eval_kernel!(padded_reduce_a13,Eval13,eval13; [L0:slot0,L1:slot1,L2:slot2,L3:slot3,L4:slot4,L5:slot5,L6:slot6,L7:slot7,L8:slot8,L9:slot9,L10:slot10,L11:slot11,L12:slot12]);

fn pass_block_count(len: usize) -> usize {
    len.div_ceil(TILE_SIZE).max(1)
}

/// Consumer-specific reduction dispatch.
#[doc(hidden)]
pub(crate) trait ReduceDispatch<R: Runtime, Input, Item, Op, Slots> {
    type Storage;

    fn execute(
        exec: &Executor<R>,
        input: &Input,
        init: Self::Storage,
    ) -> Result<Self::Storage, Error>;
}

/// One fixed-ABI reduction pass. The output contains one partial per tile.
#[doc(hidden)]
pub trait ReducePassDispatch<R, Input, Output, Item, Op, ReadSlots, WriteSlots>
where
    R: Runtime,
{
    fn execute_pass(exec: &Executor<R>, input: &Input, output: &Output) -> Result<(), Error>;
}

#[path = "reduce_fixed.rs"]
mod reduce_fixed;

fn reduce_pass<R, Input, Output, Item, Op>(
    exec: &Executor<R>,
    input: &Input,
    output: &Output,
) -> Result<(), Error>
where
    R: Runtime,
    Input: ReadExpression<Item = Item>
        + LowerReadExpression<Slots: crate::read::PaddedReadSlots>
        + StageRead<R, Env0>,
    Output: crate::output::OutputExpression<Item = Item>
        + crate::output::LowerOutputExpression<Slots: crate::output::PaddedOutputSlots>
        + crate::output::StageOutput<R, Env0>,
    Item: StorageLayout,
    Op: ReductionOp<Item>,
    Dispatch<A13, S12>: ReducePassDispatch<
            R,
            Input,
            Output,
            Item,
            Op,
            crate::read::KernelReadSlots<Input::Slots>,
            crate::output::KernelOutputSlots<Output::Slots>,
        >,
{
    <Dispatch<A13, S12> as ReducePassDispatch<
        R,
        Input,
        Output,
        Item,
        Op,
        crate::read::KernelReadSlots<Input::Slots>,
        crate::output::KernelOutputSlots<Output::Slots>,
    >>::execute_pass(exec, input, output)
}

fn finish_fixed_reduce<R, Item, Op>(
    exec: &Executor<R>,
    mut current: FixedReduceStorage<R, Item>,
    mut current_len: usize,
    init: FixedReduceStorage<R, Item>,
) -> Result<FixedReduceStorage<R, Item>, Error>
where
    R: Runtime,
    Item: crate::core::allocation::ScratchStorage<R>,
    Op: ReductionOp<Item>,
    Dispatch<A13, S12>: ReducePassDispatch<
            R,
            FixedReduceRead<R, Item>,
            FixedReduceOutput<R, Item>,
            Item,
            Op,
            crate::read::KernelReadSlots<<FixedReduceRead<R, Item> as LowerReadExpression>::Slots>,
            crate::output::KernelOutputSlots<
                <FixedReduceOutput<R, Item> as crate::output::LowerOutputExpression>::Slots,
            >,
        >,
{
    while current_len > 1 {
        let next_len = pass_block_count(current_len);
        let current_extent = RowStorage::logical_extent(&current);
        let mut next = Item::alloc_scratch(exec, next_len);
        RowStorage::set_logical_extent(
            &mut next,
            current_extent.ceil_div(exec, TILE_SIZE, next_len)?,
        );
        let input = FixedReduceRead::<R, Item>::new(current.read());
        let output = next.write();
        reduce_pass::<R, _, _, Item, Op>(exec, &input, &output)?;
        current = next;
        current_len = next_len;
    }

    let current_extent = RowStorage::logical_extent(&current);
    let mut combined = Item::alloc_scratch(exec, 2);
    init.copy_storage(exec, combined.slice_mut(..1))?;
    current.copy_storage(exec, combined.slice_mut(1..))?;
    RowStorage::set_logical_extent(
        &mut combined,
        crate::extent::LogicalExtent::add(
            exec,
            &crate::extent::LogicalExtent::fixed(1),
            &current_extent,
            2,
        )?,
    );

    let result = Item::alloc_scratch(exec, 1);
    let input = FixedReduceRead::<R, Item>::new(combined.read());
    let output = result.write();
    reduce_pass::<R, _, _, Item, Op>(exec, &input, &output)?;

    Ok(result)
}

impl<R, Input, Item, Op, Slots> ReduceDispatch<R, Input, Item, Op, Slots> for Dispatch<A13, S12>
where
    R: Runtime,
    Input: ReadExpression<Item = Item> + LowerReadExpression + StageRead<R, Env0>,
    Item: crate::core::allocation::ScratchStorage<R>,
    Op: ReductionOp<Item>,
    Dispatch<A13, S12>: ReducePassDispatch<
            R,
            Input,
            FixedReduceOutput<R, Item>,
            Item,
            Op,
            Slots,
            crate::output::KernelOutputSlots<
                <FixedReduceOutput<R, Item> as crate::output::LowerOutputExpression>::Slots,
            >,
        > + ReducePassDispatch<
            R,
            FixedReduceRead<R, Item>,
            FixedReduceOutput<R, Item>,
            Item,
            Op,
            crate::read::KernelReadSlots<<FixedReduceRead<R, Item> as LowerReadExpression>::Slots>,
            crate::output::KernelOutputSlots<
                <FixedReduceOutput<R, Item> as crate::output::LowerOutputExpression>::Slots,
            >,
        >,
{
    type Storage = FixedReduceStorage<R, Item>;

    fn execute(
        exec: &Executor<R>,
        input: &Input,
        init: Self::Storage,
    ) -> Result<Self::Storage, Error> {
        let len = input.logical_len()?;
        if len == 0 {
            return Ok(init);
        }
        let extent = input.logical_extent()?;
        let blocks = pass_block_count(len);
        let mut partials = Item::alloc_scratch(exec, blocks);
        RowStorage::set_logical_extent(&mut partials, extent.ceil_div(exec, TILE_SIZE, blocks)?);
        let output = partials.write();
        <Dispatch<A13, S12> as ReducePassDispatch<
            R,
            Input,
            FixedReduceOutput<R, Item>,
            Item,
            Op,
            Slots,
            crate::output::KernelOutputSlots<
                <FixedReduceOutput<R, Item> as crate::output::LowerOutputExpression>::Slots,
            >,
        >>::execute_pass(exec, input, &output)?;
        finish_fixed_reduce::<R, Item, Op>(exec, partials, blocks, init)
    }
}

/// Reduces all input items, starting from `init`.
pub(crate) fn reduce<R, Input, Op>(
    exec: &Executor<R>,
    input: Input,
    init: <Dispatch<A13, S12> as ReduceDispatch<
        R,
        Input,
        Input::Item,
        Op,
        crate::read::KernelReadSlots<Input::Slots>,
    >>::Storage,
    _op: Op,
) -> Result<
    <Dispatch<A13, S12> as ReduceDispatch<
        R,
        Input,
        Input::Item,
        Op,
        crate::read::KernelReadSlots<Input::Slots>,
    >>::Storage,
    Error,
>
where
    R: Runtime,
    Input: ReadExpression + LowerReadExpression + StageRead<R, Env0>,
    Input::Item: StorageLayout,
    Op: ReductionOp<Input::Item>,
    Dispatch<A13, S12>:
        ReduceDispatch<R, Input, Input::Item, Op, crate::read::KernelReadSlots<Input::Slots>>,
{
    <Dispatch<A13, S12> as ReduceDispatch<
        R,
        Input,
        Input::Item,
        Op,
        crate::read::KernelReadSlots<Input::Slots>,
    >>::execute(exec, &input, init)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::Identity;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    struct Sum;

    #[cubecl::cube]
    impl ReductionOp<u32> for Sum {
        fn apply(lhs: u32, rhs: u32) -> u32 {
            lhs + rhs
        }
    }

    struct Double;

    #[cubecl::cube]
    impl UnaryOp<u32> for Double {
        type Output = u32;

        fn apply(input: u32) -> u32 {
            input * 2
        }
    }

    struct AddPair;

    #[cubecl::cube]
    impl UnaryOp<(u32, u32)> for AddPair {
        type Output = u32;

        fn apply(input: (u32, u32)) -> u32 {
            input.0 + input.1
        }
    }

    struct AddTriple;

    #[cubecl::cube]
    impl UnaryOp<(u32, u32, u32)> for AddTriple {
        type Output = u32;

        fn apply(input: (u32, u32, u32)) -> u32 {
            input.0 + input.1 + input.2
        }
    }

    struct AddFour;

    #[cubecl::cube]
    impl UnaryOp<(u32, u32, u32, u32)> for AddFour {
        type Output = u32;

        fn apply(input: (u32, u32, u32, u32)) -> u32 {
            input.0 + input.1 + input.2 + input.3
        }
    }

    type Seven = (u32, u32, u32, u32, u32, u32, u32);
    struct AddSeven;

    #[cubecl::cube]
    impl ReductionOp<Seven> for AddSeven {
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

    fn executor() -> Executor<WgpuRuntime> {
        Executor::new(WgpuDevice::DefaultDevice)
    }

    #[test]
    fn dispatch_a1_storage1_fuses_column_transform_reduce() {
        let exec = executor();
        let len = 4097;
        let values = exec.to_device(&vec![1_u32; len]);
        let input = Transform::new(values.column(), Double);

        let output = reduce(&exec, input, exec.to_device(&[7]), Sum).unwrap();
        assert_eq!(exec.to_host(&output).unwrap()[0], 7 + 2 * len as u32);
    }

    #[test]
    fn dispatch_a2_storage1_fuses_binary_zip_transform_reduce() {
        let exec = executor();
        let len = 4097;
        let left = exec.to_device(&vec![1_u32; len]);
        let right = exec.to_device(&vec![2_u32; len]);
        let input = Transform::new(Zip::new(left.column(), right.column()), AddPair);

        let output = reduce(&exec, input, exec.to_device(&[0]), Sum).unwrap();
        assert_eq!(exec.to_host(&output).unwrap()[0], 3 * len as u32);
    }

    #[test]
    fn dispatch_a3_storage1_uses_flat_zip_semantics() {
        let exec = executor();
        let len = 4097;
        let first = exec.to_device(&vec![1_u32; len]);
        let second = exec.to_device(&vec![2_u32; len]);
        let third = exec.to_device(&vec![3_u32; len]);
        let input = Transform::new(
            Zip::new(Zip::new(first.column(), second.column()), third.column()),
            AddTriple,
        );

        let output = reduce(&exec, input, exec.to_device(&[0]), Sum).unwrap();
        assert_eq!(exec.to_host(&output).unwrap()[0], 6 * len as u32);
    }

    #[test]
    fn empty_reduce_returns_init_without_launching_or_staging() {
        let exec = executor();
        let input = Transform::new(Column::<u32>::new(), Identity);
        let output = reduce(&exec, input, exec.to_device(&[42]), Sum).unwrap();
        assert_eq!(exec.to_host(&output).unwrap(), vec![42]);
    }

    #[test]
    fn zip_length_mismatch_is_rejected_before_launch() {
        let exec = executor();
        let left = exec.to_device(&[1_u32, 2]);
        let right = exec.to_device(&[3_u32]);
        let input = Transform::new(Zip::new(left.column(), right.column()), AddPair);
        assert!(matches!(
            reduce(&exec, input, exec.to_device(&[0]), Sum),
            Err(Error::LengthMismatch { left: 2, right: 1 })
        ));
    }

    type FourColumns = Zip<Zip<Zip<Column<u32>, Column<u32>>, Column<u32>>, Column<u32>>;
    type FourInput = Transform<FourColumns, AddFour>;
    type FourExpr = <FourInput as LowerReadExpression>::DeviceExpr;

    #[test]
    fn dispatch_a4_storage1_is_a_regular_evaluator_path() {
        let exec = executor();
        let columns: Vec<_> = (1_u32..=4)
            .map(|value| exec.to_device(&vec![value; 513]))
            .collect();
        let input = Transform::new(
            Zip::new(
                Zip::new(
                    Zip::new(columns[0].column(), columns[1].column()),
                    columns[2].column(),
                ),
                columns[3].column(),
            ),
            AddFour,
        );
        let output = reduce(&exec, input, exec.to_device(&[5]), Sum).unwrap();
        assert_eq!(exec.to_host(&output).unwrap()[0], 5 + 10 * 513);
    }

    #[test]
    fn dispatch_a8_storage7_reduces_semantic_item_with_physical_leaf_partials() {
        let exec = executor();
        let len = TILE_SIZE * 2 + 17;
        let columns: Vec<_> = (1_u32..=7)
            .map(|value| exec.to_device(&vec![value; len]))
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
        let input = Permute::new(seven, Counting::new(0, len));
        let init: Seven = (10, 20, 30, 40, 50, 60, 70);

        let output = reduce(
            &exec,
            input,
            exec.scalar(init).unwrap().into_scratch_storage(),
            AddSeven,
        )
        .unwrap();
        let output = crate::Scalar::<WgpuRuntime, Seven>::from_storage(output)
            .unwrap()
            .read(&exec)
            .unwrap();
        assert_eq!(
            output,
            (
                10 + len as u32,
                20 + 2 * len as u32,
                30 + 3 * len as u32,
                40 + 4 * len as u32,
                50 + 5 * len as u32,
                60 + 6 * len as u32,
                70 + 7 * len as u32,
            )
        );
    }

    #[allow(dead_code)]
    fn a4_expression_still_has_a_valid_evaluator()
    where
        FourExpr: crate::eval::Eval4<u32, u32, u32, u32, u32>,
    {
    }
}
