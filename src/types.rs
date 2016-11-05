//! Globally-used traits, structs, and enums

use svd::types::SVDError;
use eigenvalues::types::EigenError;
use solve_linear::types::SolveError;
use least_squares::LeastSquaresError;
use factorization::qr::QRError;
use std::ops::Sub;
use std::fmt::Debug;
use num_traits::Float;
pub use lapack::{c32, c64};

/// Enum for symmetric matrix inputs.
#[repr(u8)]
pub enum Symmetric {
    /// Read elements from the upper-triangular portion of the matrix
    Upper = b'U',

    /// Read elements from the lower-triangular portion of the matrix
    Lower = b'L',
}

/// Universal `linxal` error enum
///
/// This enum can be used as a catch-all for errors from `linxal`
/// computations.
pub enum Error {
    SVD(SVDError),
    Eigen(EigenError),
    LeastSquares(LeastSquaresError),
    SolveLinear(SolveError),
    QR(QRError),
}

impl From<SVDError> for Error {
    fn from(e: SVDError) -> Error {
        Error::SVD(e)
    }
}

impl From<EigenError> for Error {
    fn from(e: EigenError) -> Error {
        Error::Eigen(e)
    }
}

impl From<LeastSquaresError> for Error {
    fn from(e: LeastSquaresError) -> Error {
        Error::LeastSquares(e)
    }
}

impl From<SolveError> for Error {
    fn from(e: SolveError) -> Error {
        Error::SolveLinear(e)
    }
}

impl From<QRError> for Error {
    fn from(e: QRError) -> Error {
        Error::QR(e)
    }
}

/// Represents quantities that have a magnitude.
pub trait Magnitude: Copy {
    fn mag(self) -> f64;
}

impl Magnitude for f32 {
    fn mag(self) -> f64 {
        self.abs() as f64
    }
}

impl Magnitude for f64 {
    fn mag(self) -> f64 {
        self.abs()
    }
}

impl Magnitude for c32 {
    fn mag(self) -> f64 {
        self.norm() as f64
    }
}

impl Magnitude for c64 {
    fn mag(self) -> f64 {
        self.norm()
    }
}

/// Common traits for all operations.
pub trait LinxalScalar: Sized + Clone + Magnitude + Debug + Sub<Output=Self> {}
impl<T: Sized + Clone + Magnitude + Debug + Sub<Output=T>> LinxalScalar for T {}

/// Scalars that are also (real) floats.
pub trait LinxalFloat: LinxalScalar + Float {}
impl<T: LinxalScalar + Float> LinxalFloat for T {}
