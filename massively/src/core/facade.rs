#![allow(private_interfaces)]

use cubecl::prelude::Runtime;

use crate::Error;

/// Exact-arity form of a logical iterator read.
///
/// The expression keeps its recursively computed [`crate::read::ReadExpression::ReadArity`]
/// until a consumer chooses a kernel ABI.  [`KernelInput::into_fixed`] is the
/// current launch adapter; exact-arity consumers can use this type directly.
pub trait KernelInput<R: Runtime>:
    Clone
    + crate::read::ReadExpression
    + crate::read::LowerReadExpression
    + crate::reduce::StageRead<R, crate::read::Env0>
{
    /// Selects the fixed launch ABI without changing the underlying read tree.
    fn into_fixed(self) -> crate::read::FixedRead<Self> {
        crate::read::FixedRead::new(self)
    }
}

impl<R, Input> KernelInput<R> for Input
where
    R: Runtime,
    Input: Clone
        + crate::read::ReadExpression
        + crate::read::LowerReadExpression
        + crate::reduce::StageRead<R, crate::read::Env0>,
{
}

/// Physical output leaves supported by the current fixed write ABI.
pub trait KernelOutputLeaves:
    crate::storage::StorePadded12
    + cubecl::prelude::CubeType<ExpandType: crate::storage::StorePadded12Expand>
    + Send
    + Sync
    + 'static
{
}

impl<Leaves> KernelOutputLeaves for Leaves
where
    Leaves: crate::storage::StorePadded12 + Send + Sync + 'static,
    <Leaves as cubecl::prelude::CubeType>::ExpandType: crate::storage::StorePadded12Expand,
{
}

/// A preallocated output tree that can be staged through the current
/// twelve-slot write ABI.
///
/// This is purely a property of the destination buffers.  It does not imply
/// that the source value has a storage layout or that new storage can be
/// allocated for either value type.
pub trait KernelOutput<R: Runtime>:
    crate::output::OutputExpression<Item: crate::StorageLayout<StorageLeaves: KernelOutputLeaves>>
    + crate::output::LowerOutputExpression<
        Slots: crate::output::PaddedOutputSlots<
            Leaves = <Self::Item as crate::StorageLayout>::StorageLeaves,
        >,
    > + crate::output::StageOutput<R, crate::read::Env0>
{
}

impl<R, Output> KernelOutput<R> for Output
where
    R: Runtime,
    Output: crate::output::OutputExpression
        + crate::output::LowerOutputExpression
        + crate::output::StageOutput<R, crate::read::Env0>,
    Output::Slots: crate::output::PaddedOutputSlots<
        Leaves = <Output::Item as crate::StorageLayout>::StorageLeaves,
    >,
    <Output::Item as crate::StorageLayout>::StorageLeaves: crate::storage::StorePadded12,
    <<Output::Item as crate::StorageLayout>::StorageLeaves as cubecl::prelude::CubeType>::ExpandType:
        crate::storage::StorePadded12Expand,
{
}

/// Device-side operations that follow directly from an item's physical leaf
/// layout. This trait has no algorithm dispatch methods.
pub trait KernelValue:
    Sized
    + Send
    + Sync
    + 'static
    + crate::storage::SelectLeaves
    + crate::storage::SharedLeaves
    + crate::storage::MutableLeaves
    + crate::storage::PlaneShuffleLeaves
    + crate::storage::LoadPadded12
    + crate::storage::LoadMutPadded12
    + crate::output::OutputSlotLayout<
        Slots: crate::output::OutputSlotEnvironment<StorageArity = Self::StorageArity>,
    >
{
    type StorageArity: crate::storage::StorageArity;
}

impl<Leaves> KernelValue for Leaves
where
    Leaves: Sized
        + Send
        + Sync
        + 'static
        + crate::storage::SelectLeaves
        + crate::storage::SharedLeaves
        + crate::storage::MutableLeaves
        + crate::storage::PlaneShuffleLeaves
        + crate::storage::LoadPadded12
        + crate::storage::LoadMutPadded12
        + crate::output::OutputSlotLayout,
    <Leaves as crate::output::OutputSlotLayout>::Slots: crate::output::OutputSlotEnvironment,
{
    type StorageArity = <<Leaves as crate::output::OutputSlotLayout>::Slots as crate::output::OutputSlotEnvironment>::StorageArity;
}

pub(crate) fn logical_len<R, Input>(input: &Input) -> Result<usize, Error>
where
    R: Runtime,
    Input: KernelInput<R>,
{
    <Input as crate::reduce::StageRead<R, crate::read::Env0>>::logical_len(input)
}

pub(crate) fn logical_extent<R, Input>(input: &Input) -> Result<crate::extent::LogicalExtent, Error>
where
    R: Runtime,
    Input: KernelInput<R>,
{
    <Input as crate::reduce::StageRead<R, crate::read::Env0>>::logical_extent(input)
}
