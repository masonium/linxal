use impl_prelude::*;

/// A solution to the singular value decomposition.
///
/// A singular value decomposition solution includes singular values
/// and, optionally, the left and right singular vectors, stored as a
/// mtrix.
pub struct SVDSolution<T: LinxalImplScalar> {
    /// Singular values of the matrix.
    ///
    /// Singular values, which are guaranteed to be non-negative
    /// reals, are returned in descending order.
    pub values: Array<T::RealPart, Ix1>,

    /// The matrix `U` of left singular vectors.
    pub left_vectors: Option<Array<T, Ix2>>,

    /// The matrix V^t of singular vectors.
    ///
    /// The transpose of V is stored, not V itself.
    pub right_vectors: Option<Array<T, Ix2>>,
}

/// An error resulting from a `SVD::compute*` method.
#[derive(Debug)]
pub enum SVDError {
    Unconverged,
    IllegalParameter(i32),
    BadLayout,
}
