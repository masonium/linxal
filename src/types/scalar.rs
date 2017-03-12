//! Define `LinxalScalar` and related types for abstracting over
//! scalars that can be used in the various linxal matrix operations.

use std::fmt::{Debug, Display};
use ndarray::LinalgScalar;
use num_traits::{Float, Zero, One, NumCast};
use std::ops::Sub;
use rand::distributions::range::SampleRange;
use std::{f32, f64};
use lapack::{c32, c64};

/// Trait for matrix operations and utilities, including conjugation and magnitude.
///
/// This trait is unifies most required operations for real and
/// complex scalars.
pub trait LinxalScalar: Sized + Default + Clone + Debug + Display + Zero + One + Sub<Output=Self> + LinalgScalar {

    /// Associated type defining the type of just the real portion of
    /// the scalar.
    ///
    /// For real-type scalars, this type is trivially the type itself.
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
