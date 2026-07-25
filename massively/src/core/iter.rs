//! Iterator composition types.

/// Private binary node used to stage zipped columns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Zip<X, Y>(pub(crate) X, pub(crate) Y);

impl<X, Y> Zip<X, Y> {
    pub(crate) const fn new(left: X, right: Y) -> Self {
        Self(left, right)
    }
}
