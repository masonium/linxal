//! Globally-used traits, structs, and enums

use svd::types::SVDError;
use eigenvalues::types::EigenError;
use solve_linear::types::SolveError;
use least_squares::LeastSquaresError;
use factorization::qr::QRError;
use factorization::lu::LUError;
use std::ops::Sub;
use std::fmt::Debug;
use num_traits::{Float, Zero, One, NumCast};
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
    LU(LUError),
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

impl From<LUError> for Error {
    fn from(e: LUError) -> Error {
        Error::LU(e)
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
pub trait LinxalScalar: Sized + Clone + Magnitude + Debug + Zero + One + Sub<Output=Self> {
    type RealPart: Float + NumCast;
}

impl LinxalScalar for f32 {
    type RealPart = f32;
}

impl LinxalScalar for f64 {
    type RealPart = f64;
}

impl LinxalScalar for c32 {
    type RealPart = f32;
}

impl LinxalScalar for c64 {
    type RealPart = f64;
}


/// Scalars that are also (real) floats.
pub trait LinxalFloat: LinxalScalar + Float {}
impl<T: LinxalScalar + Float> LinxalFloat for T {}

/// Scalars that are also (real) floats.
pub trait LinxalComplex: LinxalScalar {}
impl LinxalComplex for c32 {}
impl LinxalComplex for c64 {}
