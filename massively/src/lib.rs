//! Multi-platform GPU parallel algorithms for Rust.
//!
//! Massively provides Thrust-style algorithms on top of CubeCL. Host/device transfers are
//! explicit, algorithms return owned device storage when they naturally produce a new sequence,
//! and lazy expressions can be fused into the consuming GPU kernel. Operations whose semantics
//! require an existing destination, such as scatter, take that destination as an argument.
//!
//! # Quick start
//!
//! ```
//! use cubecl::prelude::*;
//! use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
//! use massively::{Executor, op, vector::map};
//!
//! struct Double;
//!
//! #[cubecl::cube]
//! impl op::UnaryOp<u32> for Double {
//!     type Output = u32;
//!
//!     fn apply(value: u32) -> u32 {
//!         value * 2
//!     }
//! }
//!
//! let exec = Executor::<WgpuRuntime>::new(WgpuDevice::DefaultDevice);
//! let input = exec.to_device(&[1_u32, 2, 3, 4]);
//! let output = map(&exec, input.slice(..), Double).unwrap();
//!
//! assert_eq!(exec.to_host(&output).unwrap(), vec![2, 4, 6, 8]);
//! ```
//!
//! # Core concepts
//!
//! - [`Executor`] owns the GPU runtime client and provides allocation and transfer methods.
//! - [`DeviceVec`], [`DeviceSlice`], and [`DeviceSliceMut`] are the owning and borrowed device
//!   containers used at API boundaries.
//! - [`MIter`] and [`MIterMut`] are the public iterator capabilities accepted by algorithms.
//! - [`zip2`] through [`zip12`] combine columns into native flat row tuples.
//! - [`lazy`] provides allocation-free sources and adapters.
//! - [`op`] contains reusable GPU operations such as [`op::Identity`].
//! - [`flag`] converts between CubeCL conditions and [`MFlag`].
//!
//! # Synchronization model
//!
//! Single-value-returning algorithms synchronize and return ordinary host values. Device-resident
//! intermediate scalar storage remains an internal implementation detail.
//! Logical iterator extents may remain device-resident. They are propagated
//! internally for allocation and dispatch rather than exposed as public
//! length or capacity queries.

mod api;
mod core;
pub mod flag;
pub mod seg;
pub mod vector;

// Compatibility aliases keep the kernel core independent from the public
// module layout.
pub(crate) use core::allocation::{RowAlloc, RowStorage};
pub(crate) use core::arity::{A1, A2, A3, A4, A5, A6, A7, A8, A9, A10, A11, A12, A13};
pub(crate) use core::iter::Zip;
#[doc(hidden)]
pub use core::read::Value as ValueRead;
pub(crate) use core::read::{
    Column, Constant, Counting, DivModCounting, Permute, ReadExpression, ReverseCounting, Stride,
    Taken, Transform,
};
pub(crate) use core::reduce::Dispatch;
pub(crate) use core::runtime::ColumnMut;
pub(crate) use core::storage::{S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, StorageLayout};
pub(crate) use core::value::MStorageElement;
pub(crate) use core::{
    allocation, arg_reduce, arity, eval, expansion, extent, indexed, launch, merge, ordering,
    output, predicate, radix, read, reduce, scan, search, segmented, selection, storage, transform,
    value,
};

/// Index and logical-length value used by Massively device APIs.
///
/// This is an alias of `u32`.
pub type MIndex = u32;

/// Truth flag used by Massively algorithms and device storage.
///
/// Consumers interpret zero as false and every non-zero value as true. Use
/// [`flag::from_bool()`] to construct a canonical flag and [`flag::is_set()`]
/// to test one.
pub type MFlag = u32;

pub use api::iter::{
    MAlloc, MIter, MIterMut, MRadix, MStorage, MVec, zip2, zip3, zip4, zip5, zip6, zip7, zip8,
    zip9, zip10, zip11, zip12,
};
#[doc(hidden)]
pub use api::iter::{MItem, Zipped};
#[doc(hidden)]
pub use api::runtime::{DeviceSlice, DeviceSliceMut, DeviceVec, Executor};
pub use api::{Error, lazy, op, util};
/// Common public data and operation types.
pub mod prelude {
    pub use crate::{
        DeviceSlice, DeviceSliceMut, DeviceVec, Executor, MAlloc, MFlag, MIndex, MIter, MIterMut,
        MRadix, MStorage, MVec, flag, op, zip2, zip3, zip4, zip5, zip6, zip7, zip8, zip9, zip10,
        zip11, zip12,
    };
}
