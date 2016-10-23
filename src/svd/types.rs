use ndarray::prelude::*;

/// A solution to the singular value decomposition.
///
/// A singular value decomposition solution includes singular values
/// and, optionally, the left and right singular vectors, stored as a
/// mtrix.
pub struct Solution<T: Sized> {
    pub values: Array<T, Ix>,
    pub left_vectors: Option<Array<T, (Ix, Ix)>>,
    pub right_vectors: Option<Array<T, (Ix, Ix)>>
}

pub enum SVDError<T: Sized> {
    Unconverged((i32, Array<T, Ix>)),
    IllegalParameter(i32),
    BadLayout
}
