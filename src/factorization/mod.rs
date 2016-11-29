//! Traits and functions for computing matrix factoriations.
#![deny(missing_docs)]

pub mod qr;
pub mod lu;
pub mod cholesky;

pub use self::qr::{QR, QRFactors, QRError};
pub use self::lu::{LU, LUFactors, LUError};
pub use self::cholesky::{Cholesky, CholeskyError};
