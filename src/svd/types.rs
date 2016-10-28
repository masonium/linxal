use ndarray::prelude::*;
use num_traits::{Float, ToPrimitive};
use std::fmt::Display;

/// Trait for singular values
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
    pub values: Array<SV, Ix>,
    pub left_vectors: Option<Array<IV, (Ix, Ix)>>,
    pub right_vectors: Option<Array<IV, (Ix, Ix)>>
}

impl<IV: Sized, SV: Sized> SVDSolution<IV, SV> {
    pub fn singular_values(&self) -> &Array<SV, Ix> {
        &self.values
    }
}

/// An Error resulting from SVD::compute.
#[derive(Debug)]
pub enum SVDError {
    Unconverged,
    IllegalParameter(i32),
    BadLayout
}
