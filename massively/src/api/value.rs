use std::any::Any;

use cubecl::prelude::Runtime;

use crate::{Error, Executor, MAlloc, MIndex, MStorage, MVec};

trait StoredValue<R: Runtime, T>: Send + Sync {
    fn clone_box(&self) -> Box<dyn StoredValue<R, T>>;

    fn read(&self, exec: &Executor<R>) -> Result<T, Error>;

    fn storage(&self) -> &(dyn Any + Send + Sync);

    fn into_storage(self: Box<Self>) -> Box<dyn Any + Send + Sync>;
}

struct EncodedValue<R, D, T>
where
    R: Runtime,
    D: MAlloc<R>,
{
    storage: MVec<R, D>,
    decode: fn(D) -> T,
}

impl<R, D, T> StoredValue<R, T> for EncodedValue<R, D, T>
where
    R: Runtime,
    D: MAlloc<R>,
    T: 'static,
{
    fn clone_box(&self) -> Box<dyn StoredValue<R, T>> {
        Box::new(Self {
            storage: self.storage.clone(),
            decode: self.decode,
        })
    }

    fn read(&self, exec: &Executor<R>) -> Result<T, Error> {
        let value =
            <D::Dispatch as crate::api::iter::ItemDispatch<R>>::read_value(exec, &self.storage)?;
        Ok((self.decode)(value))
    }

    fn storage(&self) -> &(dyn Any + Send + Sync) {
        &self.storage
    }

    fn into_storage(self: Box<Self>) -> Box<dyn Any + Send + Sync> {
        Box::new(self.storage)
    }
}

/// A logical value that can be resolved on either the host or the device.
///
/// Ordinary host values and [`Scalar`] implement this trait. APIs use
/// [`into_host`](Self::into_host) when their control flow needs a host value
/// and [`into_device`](Self::into_device) when a kernel needs device-resident
/// storage.
///
/// ```
/// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
/// use massively::{Scalar, Executor, MVal};
///
/// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
/// let device: Scalar<WgpuRuntime, u32> = 7_u32.into_device(&exec).unwrap();
///
/// assert_eq!((&device).into_host(&exec).unwrap(), 7);
/// assert_eq!(device.read(&exec).unwrap(), 7);
/// ```
pub trait MVal<R: Runtime, T>: Sized {
    /// Resolves this value on the host.
    ///
    /// A [`Scalar`] is read back and synchronized; a host value is returned
    /// without a transfer.
    fn into_host(self, exec: &Executor<R>) -> Result<T, Error>;

    /// Resolves this value on the device.
    ///
    /// A host value is uploaded; a [`Scalar`] is reused without a readback.
    fn into_device(self, exec: &Executor<R>) -> Result<Scalar<R, T>, Error>
    where
        T: MAlloc<R>;
}

/// One logical value stored on the device.
///
/// `T` is the logical host value returned by [`read`](Self::read). Its
/// device-resident representation is private to Massively, so logical values
/// such as `Option<MIndex>` do not need to implement a device storage trait.
///
/// GPU algorithms can pass a `Scalar` directly to later algorithms without
/// synchronizing. [`read`](Self::read) is the explicit device-to-host
/// synchronization and decoding boundary.
pub struct Scalar<R: Runtime, T> {
    value: Box<dyn StoredValue<R, T>>,
}

impl<R: Runtime, T> Clone for Scalar<R, T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone_box(),
        }
    }
}

impl<R: Runtime, T> Scalar<R, T> {
    fn from_encoded_storage<D>(storage: MVec<R, D>, decode: fn(D) -> T) -> Result<Self, Error>
    where
        D: MAlloc<R>,
        T: 'static,
    {
        let len = storage.capacity()?;
        if len != 1 {
            return Err(Error::LengthMismatch {
                left: len as usize,
                right: 1,
            });
        }
        Ok(Self {
            value: Box::new(EncodedValue { storage, decode }),
        })
    }

    /// Explicitly copies this value to the host and decodes it.
    pub fn read(&self, exec: &Executor<R>) -> Result<T, Error> {
        self.value.read(exec)
    }
}

impl<R, T> MVal<R, T> for T
where
    R: Runtime,
    T: MAlloc<R>,
{
    fn into_host(self, _exec: &Executor<R>) -> Result<T, Error> {
        Ok(self)
    }

    fn into_device(self, exec: &Executor<R>) -> Result<Scalar<R, T>, Error> {
        exec.value(self)
    }
}

impl<R, T> MVal<R, T> for Scalar<R, T>
where
    R: Runtime,
{
    fn into_host(self, exec: &Executor<R>) -> Result<T, Error> {
        self.read(exec)
    }

    fn into_device(self, _exec: &Executor<R>) -> Result<Scalar<R, T>, Error>
    where
        T: MAlloc<R>,
    {
        Ok(self)
    }
}

impl<R, T> MVal<R, T> for &Scalar<R, T>
where
    R: Runtime,
{
    fn into_host(self, exec: &Executor<R>) -> Result<T, Error> {
        self.read(exec)
    }

    fn into_device(self, _exec: &Executor<R>) -> Result<Scalar<R, T>, Error>
    where
        T: MAlloc<R>,
    {
        Ok(self.clone())
    }
}

impl<R, T> Scalar<R, T>
where
    R: Runtime,
    T: MAlloc<R>,
{
    pub(crate) fn from_storage(storage: MVec<R, T>) -> Result<Self, Error> {
        Scalar::from_encoded_storage(storage, core::convert::identity)
    }

    fn storage(&self) -> &MVec<R, T> {
        self.value
            .storage()
            .downcast_ref()
            .expect("Scalar device representation does not match its internal use")
    }

    pub(crate) fn into_storage(self) -> MVec<R, T> {
        match self.value.into_storage().downcast() {
            Ok(storage) => *storage,
            Err(_) => panic!("Scalar device representation does not match its internal use"),
        }
    }

    pub(crate) fn into_scratch_storage(self) -> <T as crate::allocation::ScratchStorage<R>>::Storage
    where
        T: crate::allocation::ScratchStorage<R>,
    {
        <T::Dispatch as crate::api::iter::ItemDispatch<R>>::into_scratch(self.into_storage())
    }

    /// Borrows this value as a one-item device iterator.
    ///
    /// This allows a device-resident scalar to participate in ordinary
    /// iterator compositions without a host readback.
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::{Executor, lazy, vector};
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let value = exec.value(7_u32).unwrap();
    /// let copies =
    ///     vector::gather(&exec, value.as_iter(), lazy::constant(0_u32).take(3)).unwrap();
    ///
    /// assert_eq!(exec.to_host(&copies).unwrap(), vec![7, 7, 7]);
    /// ```
    pub fn as_iter(&self) -> <MVec<R, T> as MStorage<R>>::Slice<'_> {
        self.storage().slice(..)
    }
}

impl<R> Scalar<R, MIndex>
where
    R: Runtime,
    MIndex: MAlloc<R>,
{
    pub(crate) fn into_optional_index(self) -> Result<Scalar<R, Option<MIndex>>, Error> {
        Scalar::from_encoded_storage(self.into_storage(), |value| {
            (value != MIndex::MAX).then_some(value)
        })
    }
}

impl<R> Scalar<R, (MIndex, MIndex)>
where
    R: Runtime,
    (MIndex, MIndex): MAlloc<R>,
{
    pub(crate) fn into_optional_index_pair(
        self,
    ) -> Result<Scalar<R, Option<(MIndex, MIndex)>>, Error> {
        Scalar::from_encoded_storage(self.into_storage(), |value: (MIndex, MIndex)| {
            (value.0 != MIndex::MAX).then_some(value)
        })
    }
}

impl<R: Runtime> Executor<R> {
    /// Uploads one device-representable host value into a [`Scalar`].
    ///
    /// ```
    /// use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    /// use massively::Executor;
    ///
    /// let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
    /// let value = exec.value(7_u32).unwrap();
    ///
    /// assert_eq!(value.read(&exec).unwrap(), 7);
    /// ```
    pub fn value<T>(&self, value: T) -> Result<Scalar<R, T>, Error>
    where
        T: MAlloc<R>,
    {
        Scalar::from_storage(
            <T::Dispatch as crate::api::iter::ItemDispatch<R>>::store_value(self, value)?,
        )
    }
}
