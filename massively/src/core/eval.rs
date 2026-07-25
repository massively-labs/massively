//! Arity-indexed device evaluators.

use core::marker::PhantomData;
use cubecl::prelude::*;
use std::rc::Rc;

use crate::{
    op::{IndexedBinaryOp, IndexedUnaryOp, UnaryOp},
    reduce::ReductionOp,
    seg::{Segment, SegmentExpand, SegmentReader},
    storage::{
        Concat, ConcatExpand, Decompose, JoinedReadRow, ReadFlatLeaves, ReadLayout, ReadRow,
        Recompose, SelectLeaves,
    },
};

/// A type-level device expression producing `Item`.
#[doc(hidden)]
pub trait DeviceExpr<Item: CubeType>: 'static + Send + Sync {}

/// How a staged leaf slot is interpreted.
#[doc(hidden)]
#[cubecl::cube]
pub trait ReadMode<T: CubePrimitive>: 'static + Send + Sync {
    fn read(slot: &[T], offset: u32, index: usize) -> T;
}

/// Reads one element at `offset + index`.
#[doc(hidden)]
pub struct Direct;

/// Reads the single staged value in a constant slot.
#[doc(hidden)]
pub struct Broadcast;

/// Reads a staged start and adds the logical index.
#[doc(hidden)]
pub struct Count;

/// Reads a staged start and advances by a staged stride.
#[doc(hidden)]
pub struct StrideCount;

/// Reads `(start + index) / divisor % modulus`.
#[doc(hidden)]
pub struct DivModCount;

/// Reads a staged final index and subtracts the logical index.
#[doc(hidden)]
pub struct ReverseCount;

/// Device expression for one row of an offset-delimited value stream.
#[doc(hidden)]
pub struct SegmentIteratorExpr<ValuesExpr, OffsetsExpr> {
    _marker: PhantomData<fn() -> (ValuesExpr, OffsetsExpr)>,
}

impl<ValuesExpr, OffsetsExpr, Item> DeviceExpr<Segment<Item>>
    for SegmentIteratorExpr<ValuesExpr, OffsetsExpr>
where
    Item: CubeType + 'static,
    ValuesExpr: DeviceExpr<Item>,
    OffsetsExpr: DeviceExpr<u32>,
{
}

#[cubecl::cube]
impl<T: CubePrimitive> ReadMode<T> for Direct {
    fn read(slot: &[T], offset: u32, index: usize) -> T {
        slot[offset as usize + index]
    }
}

#[cubecl::cube]
impl<T: CubePrimitive> ReadMode<T> for Broadcast {
    fn read(slot: &[T], _offset: u32, _index: usize) -> T {
        slot[0]
    }
}

#[cubecl::cube]
impl ReadMode<u32> for Count {
    fn read(slot: &[u32], offset: u32, index: usize) -> u32 {
        slot[0] + offset + index as u32
    }
}

#[cubecl::cube]
impl ReadMode<u32> for StrideCount {
    fn read(slot: &[u32], offset: u32, index: usize) -> u32 {
        slot[0] + (offset + index as u32) * slot[1]
    }
}

#[cubecl::cube]
impl ReadMode<u32> for DivModCount {
    fn read(slot: &[u32], offset: u32, index: usize) -> u32 {
        (slot[0] + offset + index as u32) / slot[1] % slot[2]
    }
}

#[cubecl::cube]
impl ReadMode<u32> for ReverseCount {
    fn read(slot: &[u32], offset: u32, index: usize) -> u32 {
        slot[0] - offset - index as u32
    }
}

macro_rules! define_slots {
    ($($slot:ident),+ $(,)?) => {
        $(
            #[doc(hidden)]
            pub struct $slot<T, Mode> {
                _marker: PhantomData<fn() -> (T, Mode)>,
            }

            impl<T, Mode> DeviceExpr<T> for $slot<T, Mode>
            where
                T: CubePrimitive + 'static,
                Mode: ReadMode<T>,
            {
            }
        )+
    };
}

define_slots!(
    Slot0, Slot1, Slot2, Slot3, Slot4, Slot5, Slot6, Slot7, Slot8, Slot9, Slot10, Slot11, Slot12
);

/// Device expression for a binary zip.
#[doc(hidden)]
pub struct ZipExpr<Left, Right, LeftItem, RightItem> {
    _marker: PhantomData<fn() -> (Left, Right, LeftItem, RightItem)>,
}

impl<Left, Right, LeftItem, RightItem> DeviceExpr<JoinedReadRow<LeftItem, RightItem>>
    for ZipExpr<Left, Right, LeftItem, RightItem>
where
    LeftItem: ReadRow + 'static,
    RightItem: ReadRow + 'static,
    LeftItem::ReadLeaves: ReadFlatLeaves<Item = LeftItem> + Concat<RightItem::ReadLeaves>,
    RightItem::ReadLeaves: ReadFlatLeaves<Item = RightItem>,
    <LeftItem::ReadLeaves as Concat<RightItem::ReadLeaves>>::Output: ReadFlatLeaves,
    Left: DeviceExpr<LeftItem>,
    Right: DeviceExpr<RightItem>,
{
}

/// Device expression for a unary transform.
#[doc(hidden)]
pub struct TransformExpr<InputExpr, InputItem, Op> {
    _marker: PhantomData<fn() -> (InputExpr, InputItem, Op)>,
}

/// Device expression for an index-aware unary transform.
#[doc(hidden)]
pub struct IndexedTransformExpr<InputExpr, InputItem, Op> {
    _marker: PhantomData<fn() -> (InputExpr, InputItem, Op)>,
}

/// Device expression for an index-aware adjacent transform.
#[doc(hidden)]
pub struct AdjacentIndexedTransformExpr<InputExpr, InputItem, Op> {
    _marker: PhantomData<fn() -> (InputExpr, InputItem, Op)>,
}

impl<InputExpr, InputItem, Op> DeviceExpr<Op::Output>
    for AdjacentIndexedTransformExpr<InputExpr, InputItem, Op>
where
    InputItem: CubeType + 'static,
    InputExpr: DeviceExpr<InputItem>,
    Op: IndexedBinaryOp<InputItem>,
{
}

impl<InputExpr, InputItem, Op> DeviceExpr<Op::Output>
    for IndexedTransformExpr<InputExpr, InputItem, Op>
where
    InputItem: CubeType + 'static,
    InputExpr: DeviceExpr<InputItem>,
    Op: IndexedUnaryOp<InputItem>,
{
}

/// Device expression for adjacent reduction, preserving the first item.
#[doc(hidden)]
pub struct AdjacentExpr<InputExpr, Item, Op, Layout, Leaves> {
    _marker: PhantomData<fn() -> (InputExpr, Item, Op, Layout, Leaves)>,
}

impl<InputExpr, Item, Op, Layout, Leaves> DeviceExpr<Item>
    for AdjacentExpr<InputExpr, Item, Op, Layout, Leaves>
where
    Item: CubeType + 'static,
    InputExpr: DeviceExpr<Item>,
    Op: ReductionOp<Item>,
    Leaves: CubeType + SelectLeaves + 'static,
    Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
{
}

impl<InputExpr, InputItem, Op> DeviceExpr<Op::Output> for TransformExpr<InputExpr, InputItem, Op>
where
    InputItem: CubeType + 'static,
    InputExpr: DeviceExpr<InputItem>,
    Op: UnaryOp<InputItem>,
{
}

/// Device expression for `values[indices[index]]`.
#[doc(hidden)]
pub struct PermuteExpr<ValuesExpr, IndicesExpr> {
    _marker: PhantomData<fn() -> (ValuesExpr, IndicesExpr)>,
}

impl<ValuesExpr, IndicesExpr, Item> DeviceExpr<Item> for PermuteExpr<ValuesExpr, IndicesExpr>
where
    Item: CubeType + 'static,
    ValuesExpr: DeviceExpr<Item>,
    IndicesExpr: DeviceExpr<crate::MIndex>,
{
}

macro_rules! define_eval {
    ($trait_name:ident, $method:ident; $( $leaf:ident : $slot:ident ),+ $(,)?) => {
        #[doc = concat!("Evaluates a device expression using `", stringify!($trait_name), "` staged leaves.")]
        #[cubecl::cube]
        pub trait $trait_name<Item: CubeType, $( $leaf: CubePrimitive ),+>: DeviceExpr<Item> {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> Item;
        }

        #[cubecl::cube]
        impl<LeftItem, RightItem, LeftExpr, RightExpr, $( $leaf ),+>
            $trait_name<JoinedReadRow<LeftItem, RightItem>, $( $leaf ),+>
            for ZipExpr<LeftExpr, RightExpr, LeftItem, RightItem>
        where
            LeftItem: ReadRow + 'static,
            RightItem: ReadRow + 'static,
            LeftItem::ReadLeaves:
                ReadFlatLeaves<Item = LeftItem> + Concat<RightItem::ReadLeaves>,
            RightItem::ReadLeaves: ReadFlatLeaves<Item = RightItem>,
            <LeftItem::ReadLeaves as Concat<RightItem::ReadLeaves>>::Output: ReadFlatLeaves,
            $( $leaf: CubePrimitive, )+
            LeftExpr: $trait_name<LeftItem, $( $leaf ),+>,
            RightExpr: $trait_name<RightItem, $( $leaf ),+>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> JoinedReadRow<LeftItem, RightItem> {
                let left = LeftItem::ReadDeviceLayout::decompose(
                    LeftExpr::$method($( $slot, )+ slot_offsets, index),
                );
                let right = RightItem::ReadDeviceLayout::decompose(
                    RightExpr::$method($( $slot, )+ slot_offsets, index),
                );
                <<JoinedReadRow<LeftItem, RightItem> as ReadLayout>::ReadDeviceLayout as Recompose<
                    JoinedReadRow<LeftItem, RightItem>,
                >>::recompose(left.concat(right))
            }
        }

        #[cubecl::cube]
        impl<InputItem, OutputItem, InputExpr, Op, $( $leaf ),+>
            $trait_name<OutputItem, $( $leaf ),+> for TransformExpr<InputExpr, InputItem, Op>
        where
            InputItem: CubeType + 'static,
            OutputItem: CubeType + 'static,
            $( $leaf: CubePrimitive, )+
            InputExpr: $trait_name<InputItem, $( $leaf ),+>,
            Op: UnaryOp<InputItem, Output = OutputItem>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> OutputItem {
                let input = InputExpr::$method($( $slot, )+ slot_offsets, index);
                Op::apply(input)
            }
        }

        #[cubecl::cube]
        impl<InputItem, OutputItem, InputExpr, Op, $( $leaf ),+>
            $trait_name<OutputItem, $( $leaf ),+>
            for AdjacentIndexedTransformExpr<InputExpr, InputItem, Op>
        where
            InputItem: CubeType + 'static,
            OutputItem: CubeType + 'static,
            $( $leaf: CubePrimitive, )+
            InputExpr: $trait_name<InputItem, $( $leaf ),+>,
            Op: IndexedBinaryOp<InputItem, Output = OutputItem>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> OutputItem {
                let previous_index = if index == 0usize { 0usize } else { index - 1usize };
                let previous = InputExpr::$method(
                    $( $slot, )+
                    slot_offsets,
                    previous_index,
                );
                let current = InputExpr::$method($( $slot, )+ slot_offsets, index);
                Op::apply(previous, current, index as u32)
            }
        }

        #[cubecl::cube]
        impl<InputItem, OutputItem, InputExpr, Op, $( $leaf ),+>
            $trait_name<OutputItem, $( $leaf ),+>
            for IndexedTransformExpr<InputExpr, InputItem, Op>
        where
            InputItem: CubeType + 'static,
            OutputItem: CubeType + 'static,
            $( $leaf: CubePrimitive, )+
            InputExpr: $trait_name<InputItem, $( $leaf ),+>,
            Op: IndexedUnaryOp<InputItem, Output = OutputItem>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> OutputItem {
                let input = InputExpr::$method($( $slot, )+ slot_offsets, index);
                Op::apply(input, index as u32)
            }
        }

        #[cubecl::cube]
        impl<Item, InputExpr, Op, Layout, Leaves, $( $leaf ),+>
            $trait_name<Item, $( $leaf ),+>
            for AdjacentExpr<InputExpr, Item, Op, Layout, Leaves>
        where
            Item: CubeType + 'static,
            $( $leaf: CubePrimitive, )+
            InputExpr: $trait_name<Item, $( $leaf ),+>,
            Op: ReductionOp<Item>,
            Leaves: CubeType + SelectLeaves + 'static,
            Layout: Decompose<Item, Leaves = Leaves> + Recompose<Item, Leaves = Leaves>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> Item {
                let previous_index = if index == 0usize { 0usize } else { index - 1usize };
                let first = Layout::decompose(
                    InputExpr::$method($( $slot, )+ slot_offsets, index),
                );
                let adjacent = Layout::decompose(Op::apply(
                    InputExpr::$method($( $slot, )+ slot_offsets, previous_index),
                    InputExpr::$method($( $slot, )+ slot_offsets, index),
                ));
                Layout::recompose(Leaves::select(index == 0usize, first, adjacent))
            }
        }

        #[cubecl::cube]
        impl<Item, ValuesExpr, IndicesExpr, $( $leaf ),+>
            $trait_name<Item, $( $leaf ),+> for PermuteExpr<ValuesExpr, IndicesExpr>
        where
            Item: CubeType + 'static,
            $( $leaf: CubePrimitive, )+
            ValuesExpr: $trait_name<Item, $( $leaf ),+>,
            IndicesExpr: $trait_name<crate::MIndex, $( $leaf ),+>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> Item {
                let gathered = IndicesExpr::$method($( $slot, )+ slot_offsets, index);
                ValuesExpr::$method($( $slot, )+ slot_offsets, gathered as usize)
            }
        }

    };
}

define_eval!(Eval1, eval1; L0: slot0);
define_eval!(Eval2, eval2; L0: slot0, L1: slot1);
define_eval!(Eval3, eval3; L0: slot0, L1: slot1, L2: slot2);
define_eval!(Eval4, eval4; L0: slot0, L1: slot1, L2: slot2, L3: slot3);
define_eval!(Eval5, eval5; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4);
define_eval!(Eval6, eval6; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5);
define_eval!(Eval7, eval7; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5, L6: slot6);
define_eval!(Eval8, eval8; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5, L6: slot6, L7: slot7);
define_eval!(Eval9, eval9; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5, L6: slot6, L7: slot7, L8: slot8);
define_eval!(Eval10, eval10; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5, L6: slot6, L7: slot7, L8: slot8, L9: slot9);
define_eval!(Eval11, eval11; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5, L6: slot6, L7: slot7, L8: slot8, L9: slot9, L10: slot10);
define_eval!(Eval12, eval12; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5, L6: slot6, L7: slot7, L8: slot8, L9: slot9, L10: slot10, L11: slot11);
define_eval!(Eval13, eval13; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5, L6: slot6, L7: slot7, L8: slot8, L9: slot9, L10: slot10, L11: slot11, L12: slot12);

/// Binds a segmented row to the same staged leaves used by its backing value
/// expression.  There is one implementation per total read arity, not per
/// values-arity x offsets-arity combination.
macro_rules! impl_segment_iterator_eval {
    ($trait_name:ident, $method:ident, $expand_method:ident; $( $leaf:ident : $slot:ident ),+ $(,)?) => {
        impl<Item, ValuesExpr, OffsetsExpr, $( $leaf ),+>
            $trait_name<Segment<Item>, $( $leaf ),+>
            for SegmentIteratorExpr<ValuesExpr, OffsetsExpr>
        where
            Item: CubeType + Send + Sync + 'static,
            $( $leaf: CubePrimitive + 'static, )+
            ValuesExpr: $trait_name<Item, $( $leaf ),+>,
            OffsetsExpr: $trait_name<u32, $( $leaf ),+>,
        {
            fn $method(
                $( $slot: &[$leaf], )+
                _slot_offsets: &[u32],
                _index: usize,
            ) -> Segment<Item> {
                let _ = ($( $slot, )+);
                unreachable!("segments are constructed while CubeCL expands a kernel")
            }

            fn $expand_method(
                scope: &Scope,
                $( $slot: &<[$leaf] as CubeType>::ExpandType, )+
                slot_offsets: &<[u32] as CubeType>::ExpandType,
                index: <usize as CubeType>::ExpandType,
            ) -> <Segment<Item> as CubeType>::ExpandType {
                let next_index = ExpandTypeClone::clone_unchecked(&index).__expand_add_method(
                    scope,
                    NativeExpand::from_lit(scope, 1usize),
                );
                let start = OffsetsExpr::$expand_method(
                    scope,
                    $( $slot, )+
                    slot_offsets,
                    ExpandTypeClone::clone_unchecked(&index),
                );
                let end = OffsetsExpr::$expand_method(
                    scope,
                    $( $slot, )+
                    slot_offsets,
                    next_index,
                );

                $( let $slot = ExpandTypeClone::clone_unchecked($slot); )+
                let slot_offsets = ExpandTypeClone::clone_unchecked(slot_offsets);
                let reader: SegmentReader<Item> = Rc::new(move |scope, absolute| {
                    let absolute = <usize as Cast>::__expand_cast_from(scope, absolute);
                    ValuesExpr::$expand_method(
                        scope,
                        $( &$slot, )+
                        &slot_offsets,
                        absolute,
                    )
                });

                SegmentExpand::from_bounds(scope, reader, start, end)
            }
        }
    };
}

impl_segment_iterator_eval!(Eval1, eval1, __expand_eval1; L0: slot0);
impl_segment_iterator_eval!(Eval2, eval2, __expand_eval2; L0: slot0, L1: slot1);
impl_segment_iterator_eval!(Eval3, eval3, __expand_eval3; L0: slot0, L1: slot1, L2: slot2);
impl_segment_iterator_eval!(Eval4, eval4, __expand_eval4; L0: slot0, L1: slot1, L2: slot2, L3: slot3);
impl_segment_iterator_eval!(Eval5, eval5, __expand_eval5; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4);
impl_segment_iterator_eval!(Eval6, eval6, __expand_eval6; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5);
impl_segment_iterator_eval!(Eval7, eval7, __expand_eval7; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5, L6: slot6);
impl_segment_iterator_eval!(Eval8, eval8, __expand_eval8; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5, L6: slot6, L7: slot7);
impl_segment_iterator_eval!(Eval9, eval9, __expand_eval9; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5, L6: slot6, L7: slot7, L8: slot8);
impl_segment_iterator_eval!(Eval10, eval10, __expand_eval10; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5, L6: slot6, L7: slot7, L8: slot8, L9: slot9);
impl_segment_iterator_eval!(Eval11, eval11, __expand_eval11; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5, L6: slot6, L7: slot7, L8: slot8, L9: slot9, L10: slot10);
impl_segment_iterator_eval!(Eval12, eval12, __expand_eval12; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5, L6: slot6, L7: slot7, L8: slot8, L9: slot9, L10: slot10, L11: slot11);
impl_segment_iterator_eval!(Eval13, eval13, __expand_eval13; L0: slot0, L1: slot1, L2: slot2, L3: slot3, L4: slot4, L5: slot5, L6: slot6, L7: slot7, L8: slot8, L9: slot9, L10: slot10, L11: slot11, L12: slot12);

macro_rules! impl_slot_eval {
    (
        $trait_name:ident, $method:ident, $slot_expr:ident, $offset_index:literal;
        <$( $generic:ident ),+>;
        [$( $leaf_ty:ty ),+];
        [$( $slot:ident ),+];
        $selected:ident
    ) => {
        #[cubecl::cube]
        impl<Mode, $( $generic ),+> $trait_name<T, $( $leaf_ty ),+> for $slot_expr<T, Mode>
        where
            $( $generic: CubePrimitive, )+
            Mode: ReadMode<T>,
        {
            fn $method(
                $( $slot: &[$leaf_ty], )+
                slot_offsets: &[u32],
                index: usize,
            ) -> T {
                let _ = ($( $slot, )+);
                Mode::read($selected, slot_offsets[$offset_index], index)
            }
        }
    };
}

impl_slot_eval!(Eval1, eval1, Slot0, 0; <T>; [T]; [slot0]; slot0);

impl_slot_eval!(Eval2, eval2, Slot0, 0; <T, L1>; [T, L1]; [slot0, slot1]; slot0);
impl_slot_eval!(Eval2, eval2, Slot1, 1; <L0, T>; [L0, T]; [slot0, slot1]; slot1);

impl_slot_eval!(Eval3, eval3, Slot0, 0; <T, L1, L2>; [T, L1, L2]; [slot0, slot1, slot2]; slot0);
impl_slot_eval!(Eval3, eval3, Slot1, 1; <L0, T, L2>; [L0, T, L2]; [slot0, slot1, slot2]; slot1);
impl_slot_eval!(Eval3, eval3, Slot2, 2; <L0, L1, T>; [L0, L1, T]; [slot0, slot1, slot2]; slot2);

impl_slot_eval!(Eval4, eval4, Slot0, 0; <T, L1, L2, L3>; [T, L1, L2, L3]; [slot0, slot1, slot2, slot3]; slot0);
impl_slot_eval!(Eval4, eval4, Slot1, 1; <L0, T, L2, L3>; [L0, T, L2, L3]; [slot0, slot1, slot2, slot3]; slot1);
impl_slot_eval!(Eval4, eval4, Slot2, 2; <L0, L1, T, L3>; [L0, L1, T, L3]; [slot0, slot1, slot2, slot3]; slot2);
impl_slot_eval!(Eval4, eval4, Slot3, 3; <L0, L1, L2, T>; [L0, L1, L2, T]; [slot0, slot1, slot2, slot3]; slot3);

impl_slot_eval!(Eval5, eval5, Slot0, 0; <T, L1, L2, L3, L4>; [T, L1, L2, L3, L4]; [slot0, slot1, slot2, slot3, slot4]; slot0);
impl_slot_eval!(Eval5, eval5, Slot1, 1; <L0, T, L2, L3, L4>; [L0, T, L2, L3, L4]; [slot0, slot1, slot2, slot3, slot4]; slot1);
impl_slot_eval!(Eval5, eval5, Slot2, 2; <L0, L1, T, L3, L4>; [L0, L1, T, L3, L4]; [slot0, slot1, slot2, slot3, slot4]; slot2);
impl_slot_eval!(Eval5, eval5, Slot3, 3; <L0, L1, L2, T, L4>; [L0, L1, L2, T, L4]; [slot0, slot1, slot2, slot3, slot4]; slot3);
impl_slot_eval!(Eval5, eval5, Slot4, 4; <L0, L1, L2, L3, T>; [L0, L1, L2, L3, T]; [slot0, slot1, slot2, slot3, slot4]; slot4);

impl_slot_eval!(Eval6, eval6, Slot0, 0; <T, L1, L2, L3, L4, L5>; [T, L1, L2, L3, L4, L5]; [slot0, slot1, slot2, slot3, slot4, slot5]; slot0);
impl_slot_eval!(Eval6, eval6, Slot1, 1; <L0, T, L2, L3, L4, L5>; [L0, T, L2, L3, L4, L5]; [slot0, slot1, slot2, slot3, slot4, slot5]; slot1);
impl_slot_eval!(Eval6, eval6, Slot2, 2; <L0, L1, T, L3, L4, L5>; [L0, L1, T, L3, L4, L5]; [slot0, slot1, slot2, slot3, slot4, slot5]; slot2);
impl_slot_eval!(Eval6, eval6, Slot3, 3; <L0, L1, L2, T, L4, L5>; [L0, L1, L2, T, L4, L5]; [slot0, slot1, slot2, slot3, slot4, slot5]; slot3);
impl_slot_eval!(Eval6, eval6, Slot4, 4; <L0, L1, L2, L3, T, L5>; [L0, L1, L2, L3, T, L5]; [slot0, slot1, slot2, slot3, slot4, slot5]; slot4);
impl_slot_eval!(Eval6, eval6, Slot5, 5; <L0, L1, L2, L3, L4, T>; [L0, L1, L2, L3, L4, T]; [slot0, slot1, slot2, slot3, slot4, slot5]; slot5);

impl_slot_eval!(Eval7, eval7, Slot0, 0; <T, L1, L2, L3, L4, L5, L6>; [T, L1, L2, L3, L4, L5, L6]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6]; slot0);
impl_slot_eval!(Eval7, eval7, Slot1, 1; <L0, T, L2, L3, L4, L5, L6>; [L0, T, L2, L3, L4, L5, L6]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6]; slot1);
impl_slot_eval!(Eval7, eval7, Slot2, 2; <L0, L1, T, L3, L4, L5, L6>; [L0, L1, T, L3, L4, L5, L6]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6]; slot2);
impl_slot_eval!(Eval7, eval7, Slot3, 3; <L0, L1, L2, T, L4, L5, L6>; [L0, L1, L2, T, L4, L5, L6]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6]; slot3);
impl_slot_eval!(Eval7, eval7, Slot4, 4; <L0, L1, L2, L3, T, L5, L6>; [L0, L1, L2, L3, T, L5, L6]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6]; slot4);
impl_slot_eval!(Eval7, eval7, Slot5, 5; <L0, L1, L2, L3, L4, T, L6>; [L0, L1, L2, L3, L4, T, L6]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6]; slot5);
impl_slot_eval!(Eval7, eval7, Slot6, 6; <L0, L1, L2, L3, L4, L5, T>; [L0, L1, L2, L3, L4, L5, T]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6]; slot6);

impl_slot_eval!(Eval8, eval8, Slot0, 0; <T, L1, L2, L3, L4, L5, L6, L7>; [T, L1, L2, L3, L4, L5, L6, L7]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7]; slot0);
impl_slot_eval!(Eval8, eval8, Slot1, 1; <L0, T, L2, L3, L4, L5, L6, L7>; [L0, T, L2, L3, L4, L5, L6, L7]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7]; slot1);
impl_slot_eval!(Eval8, eval8, Slot2, 2; <L0, L1, T, L3, L4, L5, L6, L7>; [L0, L1, T, L3, L4, L5, L6, L7]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7]; slot2);
impl_slot_eval!(Eval8, eval8, Slot3, 3; <L0, L1, L2, T, L4, L5, L6, L7>; [L0, L1, L2, T, L4, L5, L6, L7]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7]; slot3);
impl_slot_eval!(Eval8, eval8, Slot4, 4; <L0, L1, L2, L3, T, L5, L6, L7>; [L0, L1, L2, L3, T, L5, L6, L7]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7]; slot4);
impl_slot_eval!(Eval8, eval8, Slot5, 5; <L0, L1, L2, L3, L4, T, L6, L7>; [L0, L1, L2, L3, L4, T, L6, L7]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7]; slot5);
impl_slot_eval!(Eval8, eval8, Slot6, 6; <L0, L1, L2, L3, L4, L5, T, L7>; [L0, L1, L2, L3, L4, L5, T, L7]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7]; slot6);
impl_slot_eval!(Eval8, eval8, Slot7, 7; <L0, L1, L2, L3, L4, L5, L6, T>; [L0, L1, L2, L3, L4, L5, L6, T]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7]; slot7);
impl_slot_eval!(Eval9, eval9, Slot0, 0; <T, L1, L2, L3, L4, L5, L6, L7, L8>; [T, L1, L2, L3, L4, L5, L6, L7, L8]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8]; slot0);
impl_slot_eval!(Eval9, eval9, Slot1, 1; <L0, T, L2, L3, L4, L5, L6, L7, L8>; [L0, T, L2, L3, L4, L5, L6, L7, L8]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8]; slot1);
impl_slot_eval!(Eval9, eval9, Slot2, 2; <L0, L1, T, L3, L4, L5, L6, L7, L8>; [L0, L1, T, L3, L4, L5, L6, L7, L8]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8]; slot2);
impl_slot_eval!(Eval9, eval9, Slot3, 3; <L0, L1, L2, T, L4, L5, L6, L7, L8>; [L0, L1, L2, T, L4, L5, L6, L7, L8]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8]; slot3);
impl_slot_eval!(Eval9, eval9, Slot4, 4; <L0, L1, L2, L3, T, L5, L6, L7, L8>; [L0, L1, L2, L3, T, L5, L6, L7, L8]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8]; slot4);
impl_slot_eval!(Eval9, eval9, Slot5, 5; <L0, L1, L2, L3, L4, T, L6, L7, L8>; [L0, L1, L2, L3, L4, T, L6, L7, L8]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8]; slot5);
impl_slot_eval!(Eval9, eval9, Slot6, 6; <L0, L1, L2, L3, L4, L5, T, L7, L8>; [L0, L1, L2, L3, L4, L5, T, L7, L8]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8]; slot6);
impl_slot_eval!(Eval9, eval9, Slot7, 7; <L0, L1, L2, L3, L4, L5, L6, T, L8>; [L0, L1, L2, L3, L4, L5, L6, T, L8]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8]; slot7);
impl_slot_eval!(Eval9, eval9, Slot8, 8; <L0, L1, L2, L3, L4, L5, L6, L7, T>; [L0, L1, L2, L3, L4, L5, L6, L7, T]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8]; slot8);
impl_slot_eval!(Eval10, eval10, Slot0, 0; <T, L1, L2, L3, L4, L5, L6, L7, L8, L9>; [T, L1, L2, L3, L4, L5, L6, L7, L8, L9]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9]; slot0);
impl_slot_eval!(Eval10, eval10, Slot1, 1; <L0, T, L2, L3, L4, L5, L6, L7, L8, L9>; [L0, T, L2, L3, L4, L5, L6, L7, L8, L9]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9]; slot1);
impl_slot_eval!(Eval10, eval10, Slot2, 2; <L0, L1, T, L3, L4, L5, L6, L7, L8, L9>; [L0, L1, T, L3, L4, L5, L6, L7, L8, L9]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9]; slot2);
impl_slot_eval!(Eval10, eval10, Slot3, 3; <L0, L1, L2, T, L4, L5, L6, L7, L8, L9>; [L0, L1, L2, T, L4, L5, L6, L7, L8, L9]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9]; slot3);
impl_slot_eval!(Eval10, eval10, Slot4, 4; <L0, L1, L2, L3, T, L5, L6, L7, L8, L9>; [L0, L1, L2, L3, T, L5, L6, L7, L8, L9]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9]; slot4);
impl_slot_eval!(Eval10, eval10, Slot5, 5; <L0, L1, L2, L3, L4, T, L6, L7, L8, L9>; [L0, L1, L2, L3, L4, T, L6, L7, L8, L9]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9]; slot5);
impl_slot_eval!(Eval10, eval10, Slot6, 6; <L0, L1, L2, L3, L4, L5, T, L7, L8, L9>; [L0, L1, L2, L3, L4, L5, T, L7, L8, L9]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9]; slot6);
impl_slot_eval!(Eval10, eval10, Slot7, 7; <L0, L1, L2, L3, L4, L5, L6, T, L8, L9>; [L0, L1, L2, L3, L4, L5, L6, T, L8, L9]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9]; slot7);
impl_slot_eval!(Eval10, eval10, Slot8, 8; <L0, L1, L2, L3, L4, L5, L6, L7, T, L9>; [L0, L1, L2, L3, L4, L5, L6, L7, T, L9]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9]; slot8);
impl_slot_eval!(Eval10, eval10, Slot9, 9; <L0, L1, L2, L3, L4, L5, L6, L7, L8, T>; [L0, L1, L2, L3, L4, L5, L6, L7, L8, T]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9]; slot9);
impl_slot_eval!(Eval11, eval11, Slot0, 0; <T, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10>; [T, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10]; slot0);
impl_slot_eval!(Eval11, eval11, Slot1, 1; <L0, T, L2, L3, L4, L5, L6, L7, L8, L9, L10>; [L0, T, L2, L3, L4, L5, L6, L7, L8, L9, L10]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10]; slot1);
impl_slot_eval!(Eval11, eval11, Slot2, 2; <L0, L1, T, L3, L4, L5, L6, L7, L8, L9, L10>; [L0, L1, T, L3, L4, L5, L6, L7, L8, L9, L10]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10]; slot2);
impl_slot_eval!(Eval11, eval11, Slot3, 3; <L0, L1, L2, T, L4, L5, L6, L7, L8, L9, L10>; [L0, L1, L2, T, L4, L5, L6, L7, L8, L9, L10]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10]; slot3);
impl_slot_eval!(Eval11, eval11, Slot4, 4; <L0, L1, L2, L3, T, L5, L6, L7, L8, L9, L10>; [L0, L1, L2, L3, T, L5, L6, L7, L8, L9, L10]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10]; slot4);
impl_slot_eval!(Eval11, eval11, Slot5, 5; <L0, L1, L2, L3, L4, T, L6, L7, L8, L9, L10>; [L0, L1, L2, L3, L4, T, L6, L7, L8, L9, L10]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10]; slot5);
impl_slot_eval!(Eval11, eval11, Slot6, 6; <L0, L1, L2, L3, L4, L5, T, L7, L8, L9, L10>; [L0, L1, L2, L3, L4, L5, T, L7, L8, L9, L10]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10]; slot6);
impl_slot_eval!(Eval11, eval11, Slot7, 7; <L0, L1, L2, L3, L4, L5, L6, T, L8, L9, L10>; [L0, L1, L2, L3, L4, L5, L6, T, L8, L9, L10]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10]; slot7);
impl_slot_eval!(Eval11, eval11, Slot8, 8; <L0, L1, L2, L3, L4, L5, L6, L7, T, L9, L10>; [L0, L1, L2, L3, L4, L5, L6, L7, T, L9, L10]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10]; slot8);
impl_slot_eval!(Eval11, eval11, Slot9, 9; <L0, L1, L2, L3, L4, L5, L6, L7, L8, T, L10>; [L0, L1, L2, L3, L4, L5, L6, L7, L8, T, L10]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10]; slot9);
impl_slot_eval!(Eval11, eval11, Slot10, 10; <L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, T>; [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, T]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10]; slot10);
impl_slot_eval!(Eval12, eval12, Slot0, 0; <T, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11>; [T, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11]; slot0);
impl_slot_eval!(Eval12, eval12, Slot1, 1; <L0, T, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11>; [L0, T, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11]; slot1);
impl_slot_eval!(Eval12, eval12, Slot2, 2; <L0, L1, T, L3, L4, L5, L6, L7, L8, L9, L10, L11>; [L0, L1, T, L3, L4, L5, L6, L7, L8, L9, L10, L11]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11]; slot2);
impl_slot_eval!(Eval12, eval12, Slot3, 3; <L0, L1, L2, T, L4, L5, L6, L7, L8, L9, L10, L11>; [L0, L1, L2, T, L4, L5, L6, L7, L8, L9, L10, L11]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11]; slot3);
impl_slot_eval!(Eval12, eval12, Slot4, 4; <L0, L1, L2, L3, T, L5, L6, L7, L8, L9, L10, L11>; [L0, L1, L2, L3, T, L5, L6, L7, L8, L9, L10, L11]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11]; slot4);
impl_slot_eval!(Eval12, eval12, Slot5, 5; <L0, L1, L2, L3, L4, T, L6, L7, L8, L9, L10, L11>; [L0, L1, L2, L3, L4, T, L6, L7, L8, L9, L10, L11]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11]; slot5);
impl_slot_eval!(Eval12, eval12, Slot6, 6; <L0, L1, L2, L3, L4, L5, T, L7, L8, L9, L10, L11>; [L0, L1, L2, L3, L4, L5, T, L7, L8, L9, L10, L11]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11]; slot6);
impl_slot_eval!(Eval12, eval12, Slot7, 7; <L0, L1, L2, L3, L4, L5, L6, T, L8, L9, L10, L11>; [L0, L1, L2, L3, L4, L5, L6, T, L8, L9, L10, L11]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11]; slot7);
impl_slot_eval!(Eval12, eval12, Slot8, 8; <L0, L1, L2, L3, L4, L5, L6, L7, T, L9, L10, L11>; [L0, L1, L2, L3, L4, L5, L6, L7, T, L9, L10, L11]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11]; slot8);
impl_slot_eval!(Eval12, eval12, Slot9, 9; <L0, L1, L2, L3, L4, L5, L6, L7, L8, T, L10, L11>; [L0, L1, L2, L3, L4, L5, L6, L7, L8, T, L10, L11]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11]; slot9);
impl_slot_eval!(Eval12, eval12, Slot10, 10; <L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, T, L11>; [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, T, L11]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11]; slot10);
impl_slot_eval!(Eval12, eval12, Slot11, 11; <L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, T>; [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, T]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11]; slot11);
impl_slot_eval!(Eval13, eval13, Slot0, 0; <T, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>; [T, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot0);
impl_slot_eval!(Eval13, eval13, Slot1, 1; <L0, T, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>; [L0, T, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot1);
impl_slot_eval!(Eval13, eval13, Slot2, 2; <L0, L1, T, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12>; [L0, L1, T, L3, L4, L5, L6, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot2);
impl_slot_eval!(Eval13, eval13, Slot3, 3; <L0, L1, L2, T, L4, L5, L6, L7, L8, L9, L10, L11, L12>; [L0, L1, L2, T, L4, L5, L6, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot3);
impl_slot_eval!(Eval13, eval13, Slot4, 4; <L0, L1, L2, L3, T, L5, L6, L7, L8, L9, L10, L11, L12>; [L0, L1, L2, L3, T, L5, L6, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot4);
impl_slot_eval!(Eval13, eval13, Slot5, 5; <L0, L1, L2, L3, L4, T, L6, L7, L8, L9, L10, L11, L12>; [L0, L1, L2, L3, L4, T, L6, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot5);
impl_slot_eval!(Eval13, eval13, Slot6, 6; <L0, L1, L2, L3, L4, L5, T, L7, L8, L9, L10, L11, L12>; [L0, L1, L2, L3, L4, L5, T, L7, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot6);
impl_slot_eval!(Eval13, eval13, Slot7, 7; <L0, L1, L2, L3, L4, L5, L6, T, L8, L9, L10, L11, L12>; [L0, L1, L2, L3, L4, L5, L6, T, L8, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot7);
impl_slot_eval!(Eval13, eval13, Slot8, 8; <L0, L1, L2, L3, L4, L5, L6, L7, T, L9, L10, L11, L12>; [L0, L1, L2, L3, L4, L5, L6, L7, T, L9, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot8);
impl_slot_eval!(Eval13, eval13, Slot9, 9; <L0, L1, L2, L3, L4, L5, L6, L7, L8, T, L10, L11, L12>; [L0, L1, L2, L3, L4, L5, L6, L7, L8, T, L10, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot9);
impl_slot_eval!(Eval13, eval13, Slot10, 10; <L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, T, L11, L12>; [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, T, L11, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot10);
impl_slot_eval!(Eval13, eval13, Slot11, 11; <L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, T, L12>; [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, T, L12]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot11);
impl_slot_eval!(Eval13, eval13, Slot12, 12; <L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, T>; [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11, T]; [slot0, slot1, slot2, slot3, slot4, slot5, slot6, slot7, slot8, slot9, slot10, slot11, slot12]; slot12);
