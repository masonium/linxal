use ndarray::prelude::*;

/// A solution to the singular value decomposition.
///
/// A singular value decomposition solution includes singular values
/// and, optionally, the left and right singular vectors, stored as a
/// mtrix.
pub struct Solution<IV: Sized, SV: Sized> {
    pub values: Array<SV, Ix>,
    pub left_vectors: Option<Array<IV, (Ix, Ix)>>,
    pub right_vectors: Option<Array<IV, (Ix, Ix)>>
}

/// An Error resulting from SVD::compute.
#[derive(Debug)]
pub enum SVDError {
    Unconverged,
    IllegalParameter(i32),
    BadLayout
}
