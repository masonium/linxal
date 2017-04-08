//! Define `LinxalImplScalar` and related types for abstracting over
//! scalars that can be used in the various linxal matrix operations.

use std::fmt::{Debug, Display};
use ndarray::LinalgScalar;
use num_traits::{Float, Zero, One, NumCast};
use std::ops::Sub;
use rand::distributions::range::SampleRange;
use std::{f32, f64};
use lapack::{c32, c64};

/// Aggregate trait for implementing matrix operations and utilities.
///
/// This trait is unifies most required operations for real and
/// complex scalars. It also defines common utility functions for
/// internal implementations.
pub trait LinxalImplScalar
    : Sized + Default + Clone + Debug + Display + Zero + One + Sub<Output = Self> + LinalgScalar
    {
    /// Associated type defining the type of just the real portion of
    /// the scalar.
    ///
    /// For real-type scalars, this type is trivially the type itself.
    type RealPart: LinxalImplScalar + Float + NumCast + From<f32> + SampleRange;

    /// Associated type defining the complex variant of this scalar.
    ///
    /// For real-type scalars, this type is trivially the type itself.
    type Complex: LinxalImplScalar;

    /// Return the conjugate of the value.
    fn cj(self) -> Self;

    /// Returns the magnitude of the scalar.
    fn mag(self) -> Self::RealPart;

    /// Return the machine epsilon for the value.
    fn eps() -> Self::RealPart;

    /// Return the default tolerance used for comparisons.
    fn tol() -> Self::RealPart;
}

impl LinxalImplScalar for f32 {
    type RealPart = f32;
    type Complex = c32;

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

impl LinxalImplScalar for f64 {
    type RealPart = f64;
    type Complex = c64;

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

impl LinxalImplScalar for c32 {
    type RealPart = f32;
    type Complex = c32;

    fn cj(self) -> Self {
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


impl LinxalImplScalar for c64 {
    type RealPart = f64;
    type Complex = c64;

    fn cj(self) -> Self {
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
pub trait LinxalImplFloat: LinxalImplScalar + Float {}
impl<T: LinxalImplScalar + Float> LinxalImplFloat for T {}

/// Scalars that are also complex floats.
pub trait LinxalImplComplex: LinxalImplScalar {}
impl LinxalImplComplex for c32 {}
impl LinxalImplComplex for c64 {}
