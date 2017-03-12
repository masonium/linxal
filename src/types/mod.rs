//! Globally-used traits, structs, and enums
#![deny(missing_docs)]

pub use lapack::{c32, c64};
pub use std::{f32, f64};

/// Enum for symmetric matrix inputs.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Symmetric {
    /// Read elements from the upper-triangular portion of the matrix
    Upper = b'U',

    /// Read elements from the lower-triangular portion of the matrix
    Lower = b'L',
}

pub mod impl_scalar;
pub mod scalar;
pub mod error;
pub mod matrix;

pub use self::error::Error;
pub use self::impl_scalar::LinxalImplScalar;
pub use self::scalar::{LinxalScalar, LinxalReal, LinxalComplex};
pub use self::matrix::{LinxalMatrix};
