//! Globally-used traits, structs, and enums
use svd::types::SVDError;
use ndarray::LinalgScalar;
use eigenvalues::types::EigenError;
use solve_linear::types::SolveError;
use least_squares::LeastSquaresError;
use factorization::qr::QRError;
use factorization::lu::LUError;
use std::ops::Sub;
use std::fmt::{Debug, Display};
use num_traits::{Float, Zero, One, NumCast};
use std::{f32, f64};
use rand::distributions::range::SampleRange;
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

/// Trait for matrix operations and utilities, including conjugation and magnitude.
///
/// This trait is unifies most required operations for real and
/// complex scalars.
pub trait LinxalScalar: Sized + Clone + Debug + Display + Zero + One + Sub<Output=Self> + LinalgScalar {
    type RealPart: LinxalScalar + Float + NumCast + From<f32> + SampleRange;

    /// Return the conjugate of the value.
    fn cj(self) -> Self;

    /// Returns the magnitude of the scalar.
    fn mag(self) -> Self::RealPart;

    /// Return the machine epsilon for the value.
    fn eps() -> Self::RealPart;

    /// Return the default tolerance used for comparisons.
    fn tol() -> Self::RealPart;
}

impl LinxalScalar for f32 {
    type RealPart = f32;

    fn cj(self) -> Self {
        self
    }
    fn eps() -> Self::RealPart {
        f32::EPSILON
    }
    fn mag(self) -> Self::RealPart {
        self.abs()
    }
    fn tol() -> Self::RealPart {
        1e-5
    }
}

impl LinxalScalar for f64 {
    type RealPart = f64;

    fn cj(self) -> Self {
        self
    }
    fn eps() -> Self::RealPart {
        f64::EPSILON
    }
    fn mag(self) -> Self::RealPart {
        self.abs()
    }
    fn tol() -> Self::RealPart {
        2e-14
    }
}

impl LinxalScalar for c32 {
    type RealPart = f32;

    fn cj(self) -> Self{
        self.conj()
    }
    fn eps() -> Self::RealPart {
        f32::EPSILON
    }
    fn mag(self) -> Self::RealPart {
        self.norm()
    }
    fn tol() -> Self::RealPart {
        2e-5
    }
}

impl LinxalScalar for c64 {
    type RealPart = f64;

    fn cj(self) -> Self{
        self.conj()
    }
    fn eps() -> Self::RealPart {
        f64::EPSILON
    }
    fn mag(self) -> Self::RealPart {
        self.norm()
    }
    fn tol() -> Self::RealPart {
        4e-14
    }
}


/// Scalars that are also (real) floats.
pub trait LinxalFloat: LinxalScalar + Float {}
impl<T: LinxalScalar + Float> LinxalFloat for T {}

/// Scalars that are also complex floats.
pub trait LinxalComplex: LinxalScalar {}
impl LinxalComplex for c32 {}
impl LinxalComplex for c64 {}
