//! Truth-flag conversion helpers.

use cubecl::prelude::*;

use crate::MFlag;

/// Converts a CubeCL condition into a canonical [`MFlag`].
#[cubecl::cube]
pub fn from_bool(value: bool) -> MFlag {
    if value { 1u32 } else { 0u32 }
}

/// Tests whether an [`MFlag`] is set.
///
/// Zero is false; every non-zero value is true.
#[cubecl::cube]
pub fn is_set(value: MFlag) -> bool {
    value != 0u32
}

#[cfg(test)]
mod tests {
    use super::{from_bool, is_set};

    #[test]
    fn conversions_use_nonzero_truth_semantics() {
        assert_eq!(from_bool(false), 0);
        assert_eq!(from_bool(true), 1);
        assert!(!is_set(0));
        assert!(is_set(1));
        assert!(is_set(7));
    }
}
