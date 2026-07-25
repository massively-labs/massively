//! Error type for massively execution.

use core::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Error {
    LengthMismatch { left: usize, right: usize },
    OutputTooShort { input: usize, output: usize },
    LengthTooLarge { len: usize },
    InvalidSegmentation,
    UnboundColumn,
    ForeignExecutor,
    Launch { message: String },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { left, right } => {
                write!(f, "input lengths differ: {left} != {right}")
            }
            Self::OutputTooShort { input, output } => {
                write!(
                    f,
                    "output length {output} is shorter than input length {input}"
                )
            }
            Self::LengthTooLarge { len } => write!(f, "length does not fit in u32: {len}"),
            Self::InvalidSegmentation => {
                write!(f, "segmentation is not a valid ordered partition")
            }
            Self::UnboundColumn => write!(f, "column is not bound to device storage"),
            Self::ForeignExecutor => write!(f, "executor does not own this device data"),
            Self::Launch { message } => write!(f, "CubeCL launch failed: {message}"),
        }
    }
}

impl std::error::Error for Error {}
