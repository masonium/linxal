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

impl<T: LinxalImplScalar> SVDSolution<T> {
    // Return true if the full complement of vectors is available to
    // fully reconstruct the original matrix.
    pub fn is_reconstructable(&self) -> bool {
        self.left_vectors.is_some() && self.right_vectors.is_some()
    }

    /// Reconstruct the original matrix if both left and right
    /// singular vectors are present.
    pub fn reconstruct(&self) -> Option<Array<T, Ix2>> {
        match (&self.left_vectors, &self.right_vectors)  {
            (&Some(ref u), &Some(ref vt)) => {
                let mut d: Array<T, Ix2> = Array::zeros((u.dim().1, vt.dim().0));
                d.diag_mut().zip_mut_with(&self.values, |dx, vx| {
                    *dx = T::from_real(*vx)
                });

                Some(u.dot(&d.dot(vt)))
            },
            _ => None
        }
    }
}

/// An error resulting from a `SVD::compute*` method.
#[derive(Debug)]
pub enum SVDError {
    /// The decomposition algorithm failed to converge
    Unconverged,

    // Internal error from underlying LAPACK call
    IllegalParameter(i32),

    /// The input matrix does not follow C- or F- layout.
    BadLayout,
}
