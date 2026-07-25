use cubecl::prelude::Runtime;

use crate::{Error, Executor, MAlloc, MStorage, MVec};

/// One device-resident logical value.
///
/// An `MVal` owns one row of ordinary Massively storage. Tuple values use the
/// same structure-of-arrays layout as tuple vectors. Public algorithms create
/// these values internally and read them only at synchronous return boundaries.
#[allow(private_bounds)]
pub(crate) struct MVal<R, T>
where
    R: Runtime,
    T: MAlloc<R>,
{
    pub(crate) storage: MVec<R, T>,
}

impl<R, T> Clone for MVal<R, T>
where
    R: Runtime,
    T: MAlloc<R>,
    MVec<R, T>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
        }
    }
}

impl<R, T> MVal<R, T>
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
        <<T as MAlloc<R>>::Dispatch as crate::api::iter::ItemDispatch<R>>::into_scratch(
            self.storage,
        )
    }

    /// Borrows this value as a one-item device iterator.
    pub(crate) fn as_iter(&self) -> <MVec<R, T> as MStorage<R>>::Slice<'_> {
        self.storage.slice(..)
    }

    /// Explicitly copies this value to the host and waits for its producers.
    pub(crate) fn read(&self, exec: &Executor<R>) -> Result<T, Error> {
        <<T as MAlloc<R>>::Dispatch as crate::api::iter::ItemDispatch<R>>::read_value(
            exec,
            &self.storage,
        )
    }
}

impl<R: Runtime> Executor<R> {
    /// Uploads one host value into a device-resident [`MVal`].
    pub(crate) fn value<T>(&self, value: T) -> Result<MVal<R, T>, Error>
    where
        T: MAlloc<R>,
    {
        MVal::from_storage(
            <<T as MAlloc<R>>::Dispatch as crate::api::iter::ItemDispatch<R>>::store_value(
                self, value,
            )?,
        )
    }
}
