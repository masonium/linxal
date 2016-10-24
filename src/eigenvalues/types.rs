use ndarray::prelude::*;

#[derive(Debug)]
pub enum EigenError {
    Success,
    BadLayout,
    BadParameter(i32),
    Failed
}

pub enum ComputeVectors {
    Left,
    Right,
    Both
}

/// Solution to an eigenvalue problem..
///
/// Contains thte eigenvalues and, optionally, the left and/or right
/// eigenvectors of the solution. For symmetric problems, the
/// eigenvectors are placed in `right_eigenvectors`.
pub struct Solution<IV, EV> {
    pub values: Array<EV, Ix>,
    pub left_vectors: Option<Array<IV, (Ix, Ix)>>,
    pub right_vectors: Option<Array<IV, (Ix, Ix)>>
}

