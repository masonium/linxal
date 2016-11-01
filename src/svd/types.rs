use ndarray::prelude::*;
use num_traits::{Float, ToPrimitive};
use std::fmt::Display;

/// Trait for singular values
///
/// A type implementing `SingularValue` can be returned as a singular
/// value by `SVD::compute*`.
pub trait SingularValue: Float + Display + ToPrimitive {
}

impl<T: Float + Display + ToPrimitive> SingularValue for T {
}


/// A solution to the singular value decomposition.
///
/// A singular value decomposition solution includes singular values
/// and, optionally, the left and right singular vectors, stored as a
/// mtrix.
pub struct SVDSolution<IV: Sized, SV: Sized> {
    /// Singular values of the matrix.
    ///
    /// Singular values, which are guaranteed to be non-negative
    /// reals, are returned in descending order.
    pub values: Array<SV, Ix>,

    /// The matrix `U` of left singular vectors.
    pub left_vectors: Option<Array<IV, (Ix, Ix)>>,

    /// The matrix V^t of singular vectors.
    ///
    /// The transpose of V is stored, not V itself.
    pub right_vectors: Option<Array<IV, (Ix, Ix)>>
}

/// An error resulting from a `SVD::compute*` method.
#[derive(Debug)]
pub enum SVDError {
    Unconverged,
    IllegalParameter(i32),
    BadLayout
}
