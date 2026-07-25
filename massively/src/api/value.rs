use core::marker::PhantomData;

use cubecl::prelude::*;

use crate::{
    Error, Executor, MAlloc, MIndex, MIter, MStorage, MStorageElement, MVec,
    api::iter::MStorageExtent, lazy, op,
};

/// A logical value that can be consumed on the device or resolved on the host.
///
/// [`as_iter`](Self::as_iter) never performs a readback. It returns a
/// one-item iterator that can be fused into a consuming GPU operation.
/// [`read`](Self::read) is the explicit host-resolution boundary.
pub trait MVal<R, Item>: Sized
where
    R: Runtime,
    Item: CubeType + Send + Sync + 'static,
{
    /// The one-item iterator representation of this value.
    type Iter<'a>: MIter<R, Item = Item>
    where
        Self: 'a;

    /// Borrows this value as a one-item device-readable iterator.
    fn as_iter(&self) -> Self::Iter<'_>;

    /// Resolves this value on the host.
    fn read(&self, exec: &Executor<R>) -> Result<Item, Error>;
}

/// Internal ability to express one host value as constant read leaves.
///
/// This is intentionally separate from [`MAlloc`]: producing a read
/// expression does not imply that the semantic item has an owned storage
/// representation.
#[doc(hidden)]
pub trait OneValueRead<R>: CubeType + Send + Sync + 'static
where
    R: Runtime,
{
    type Iter<'a>: MIter<R, Item = Self>
    where
        Self: 'a;

    fn one_value_iter(&self) -> Self::Iter<'_>;
}

type One<T> = lazy::Taken<lazy::Constant<T>>;

impl<R, T> OneValueRead<R> for T
where
    R: Runtime,
    T: MStorageElement,
{
    type Iter<'a>
        = One<T>
    where
        Self: 'a;

    fn one_value_iter(&self) -> Self::Iter<'_> {
        lazy::constant(*self).take(1)
    }
}

macro_rules! nested_zip_type {
    ($first:ident, $second:ident $(, $rest:ident)*) => {
        nested_zip_type!(@fold crate::Zipped<One<$first>, One<$second>> $(, $rest)*)
    };
    (@fold $current:ty) => { $current };
    (@fold $current:ty, $next:ident $(, $rest:ident)*) => {
        nested_zip_type!(@fold crate::Zipped<$current, One<$next>> $(, $rest)*)
    };
}

macro_rules! impl_one_value_tuple {
    ($(($($ty:ident:$index:tt),+)),+ $(,)?) => {
        $(
            impl<R, $($ty),+> OneValueRead<R> for ($($ty,)+)
            where
                R: Runtime,
                $($ty: MStorageElement,)+
            {
                type Iter<'a>
                    = nested_zip_type!($($ty),+)
                where
                    Self: 'a;

                fn one_value_iter(&self) -> Self::Iter<'_> {
                    let iter = crate::zip2(
                        lazy::constant(self.0).take(1),
                        lazy::constant(self.1).take(1),
                    );
                    $(
                        impl_one_value_tuple!(@append iter, self, $index);
                    )+
                    iter
                }
            }
        )+
    };
    (@append $iter:ident, $self:ident, 0) => {};
    (@append $iter:ident, $self:ident, 1) => {};
    (@append $iter:ident, $self:ident, $index:tt) => {
        let $iter = crate::Zipped::new(
            $iter,
            lazy::constant($self.$index).take(1),
        );
    };
}

impl_one_value_tuple!(
    (A:0, B:1),
    (A:0, B:1, C:2),
    (A:0, B:1, C:2, D:3),
    (A:0, B:1, C:2, D:3, E:4),
    (A:0, B:1, C:2, D:3, E:4, F:5),
    (A:0, B:1, C:2, D:3, E:4, F:5, G:6),
    (A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7),
    (A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8),
    (A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9),
    (A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10),
    (A:0, B:1, C:2, D:3, E:4, F:5, G:6, H:7, I:8, J:9, K:10, L:11),
);

impl<R, T> MVal<R, T> for T
where
    R: Runtime,
    T: OneValueRead<R> + Clone,
{
    type Iter<'a>
        = T::Iter<'a>
    where
        Self: 'a;

    fn as_iter(&self) -> Self::Iter<'_> {
        self.one_value_iter()
    }

    fn read(&self, _exec: &Executor<R>) -> Result<T, Error> {
        Ok(self.clone())
    }
}

/// One directly representable logical value stored on the device.
pub(crate) struct Scalar<R, T>
where
    R: Runtime,
    T: MAlloc<R>,
{
    storage: MVec<R, T>,
}

impl<R, T> Clone for Scalar<R, T>
where
    R: Runtime,
    T: MAlloc<R>,
{
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
        }
    }
}

impl<R, T> Scalar<R, T>
where
    R: Runtime,
    T: MAlloc<R>,
{
    pub(crate) fn from_storage(storage: MVec<R, T>) -> Result<Self, Error> {
        let len = storage.capacity()?;
        if len != 1 {
            return Err(Error::LengthMismatch {
                left: len as usize,
                right: 1,
            });
        }
        Ok(Self { storage })
    }

    pub(crate) fn into_storage(self) -> MVec<R, T> {
        self.storage
    }

    pub(crate) fn into_scratch_storage(self) -> <T as crate::allocation::ScratchStorage<R>>::Storage
    where
        T: crate::allocation::ScratchStorage<R>,
    {
        <T::Dispatch as crate::api::iter::ItemDispatch<R>>::into_scratch(self.into_storage())
    }

    pub(crate) fn as_iter(&self) -> <MVec<R, T> as MStorage<R>>::Slice<'_> {
        self.storage.slice(..)
    }

    pub(crate) fn read(&self, exec: &Executor<R>) -> Result<T, Error> {
        <T::Dispatch as crate::api::iter::ItemDispatch<R>>::read_value(exec, &self.storage)
    }
}

impl<R, T> MVal<R, T> for Scalar<R, T>
where
    R: Runtime,
    T: MAlloc<R>,
{
    type Iter<'a>
        = <MVec<R, T> as MStorage<R>>::Slice<'a>
    where
        Self: 'a;

    fn as_iter(&self) -> Self::Iter<'_> {
        Scalar::as_iter(self)
    }

    fn read(&self, exec: &Executor<R>) -> Result<T, Error> {
        Scalar::read(self, exec)
    }
}

impl<R, T> MVal<R, T> for &Scalar<R, T>
where
    R: Runtime,
    T: MAlloc<R>,
{
    type Iter<'a>
        = <MVec<R, T> as MStorage<R>>::Slice<'a>
    where
        Self: 'a;

    fn as_iter(&self) -> Self::Iter<'_> {
        Scalar::as_iter(self)
    }

    fn read(&self, exec: &Executor<R>) -> Result<T, Error> {
        Scalar::read(self, exec)
    }
}

/// A device/host pair of equivalent decode operations for one logical value.
pub trait ValueMap<Input>: op::UnaryOp<Input>
where
    Input: CubeType,
{
    /// Applies the same decode operation on the host.
    fn apply_host(value: Input) -> Self::Output;
}

/// A value whose public representation is lazily decoded from another value.
pub(crate) struct MappedValue<Value, Op, Input> {
    value: Value,
    op: Op,
    _input: PhantomData<fn(Input)>,
}

impl<Value, Op, Input> MappedValue<Value, Op, Input> {
    pub(crate) const fn new(value: Value, op: Op) -> Self {
        Self {
            value,
            op,
            _input: PhantomData,
        }
    }
}

impl<R, Input, Output, Value, Op> MVal<R, Output> for MappedValue<Value, Op, Input>
where
    R: Runtime,
    Input: CubeType + Send + Sync + 'static,
    Output: CubeType + Send + Sync + 'static,
    Value: MVal<R, Input>,
    Op: ValueMap<Input, Output = Output> + Clone,
    for<'a> lazy::Map<Value::Iter<'a>, Op>: MIter<R, Item = Output>,
{
    type Iter<'a>
        = lazy::Map<Value::Iter<'a>, Op>
    where
        Self: 'a;

    fn as_iter(&self) -> Self::Iter<'_> {
        lazy::map(self.value.as_iter(), self.op.clone())
    }

    fn read(&self, exec: &Executor<R>) -> Result<Output, Error> {
        Ok(Op::apply_host(self.value.read(exec)?))
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DecodeOptionalIndex;

#[cubecl::cube]
impl op::UnaryOp<MIndex> for DecodeOptionalIndex {
    type Output = Option<MIndex>;

    fn apply(value: MIndex) -> Self::Output {
        let mut result = Option::Some(value);
        if value == MIndex::MAX {
            result = Option::None;
        }
        result
    }
}

impl ValueMap<MIndex> for DecodeOptionalIndex {
    fn apply_host(value: MIndex) -> Self::Output {
        (value != MIndex::MAX).then_some(value)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DecodeOptionalIndexPair;

#[cubecl::cube]
impl op::UnaryOp<(MIndex, MIndex)> for DecodeOptionalIndexPair {
    type Output = Option<(MIndex, MIndex)>;

    fn apply(value: (MIndex, MIndex)) -> Self::Output {
        let mut result = Option::Some(value);
        if value.0 == MIndex::MAX {
            result = Option::None;
        }
        result
    }
}

impl ValueMap<(MIndex, MIndex)> for DecodeOptionalIndexPair {
    fn apply_host(value: (MIndex, MIndex)) -> Self::Output {
        (value.0 != MIndex::MAX).then_some(value)
    }
}

impl<R> Scalar<R, MIndex>
where
    R: Runtime,
    MIndex: MAlloc<R, Owned = crate::DeviceVec<R, MIndex>>,
{
    pub(crate) fn logical_extent(&self, upper_bound: MIndex) -> crate::extent::LogicalExtent {
        crate::extent::LogicalExtent::from_device(&self.storage, upper_bound as usize)
    }

    pub(crate) fn into_optional_index(self) -> MappedValue<Self, DecodeOptionalIndex, MIndex> {
        MappedValue::new(self, DecodeOptionalIndex)
    }
}

impl<R> Scalar<R, (MIndex, MIndex)>
where
    R: Runtime,
    (MIndex, MIndex): MAlloc<R>,
{
    pub(crate) fn into_optional_index_pair(
        self,
    ) -> MappedValue<Self, DecodeOptionalIndexPair, (MIndex, MIndex)> {
        MappedValue::new(self, DecodeOptionalIndexPair)
    }
}

impl<R: Runtime> Executor<R> {
    pub(crate) fn scalar<T>(&self, value: T) -> Result<Scalar<R, T>, Error>
    where
        T: MAlloc<R>,
    {
        Scalar::from_storage(
            <T::Dispatch as crate::api::iter::ItemDispatch<R>>::store_value(self, value)?,
        )
    }
}

pub(crate) fn materialize_value<R, T, Value>(
    exec: &Executor<R>,
    value: &Value,
) -> Result<Scalar<R, T>, Error>
where
    R: Runtime,
    T: MAlloc<R>,
    Value: MVal<R, T>,
{
    Scalar::from_storage(crate::vector::map(exec, value.as_iter(), op::Identity)?)
}
