use ndarray::prelude::*;
use ndarray::{Ix1, Ix2};

/// Errors from an eigenvalue problem.
#[derive(Debug)]
pub enum EigenError {
    NotSquare,
    BadLayout,
    BadParameter(i32),
    Failed,
}

/// Solution to an eigenvalue problem..
///
/// Contains thte eigenvalues and, optionally, the left and/or right
/// eigenvectors of the solution. For symmetric problems, the
/// eigenvectors are placed in `right_eigenvectors`.
pub struct Solution<IV, EV> {
    pub values: Array<EV, Ix1>,
    pub left_vectors: Option<Array<IV, Ix2>>,
    pub right_vectors: Option<Array<IV, Ix2>>,
}
