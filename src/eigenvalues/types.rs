//! Error and solution types for eigenvalue problems.

use impl_prelude::*;

/// Errors from an eigenvalue problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EigenError {
    /// The input matrix is not square.
    NotSquare,

    /// The input matrix does not have a conforming layout.
    BadLayout,

    /// The parameter to the LAPACKE method was invalid.
    ///
    /// This error should not occur in user-facing code.
    IllegalParameter(i32),

    /// Eigenvalues could not be found.
    Failed,
}

/// Solution to an eigenvalue problem.
///
/// Contains thte eigenvalues and, optionally, the left and/or right
/// eigenvectors of the solution. For symmetric problems, the
/// eigenvectors are placed in `right_eigenvectors`.
pub struct Solution<IV, EV> {
    /// Eigenvalues of the matrix.
    pub values: Array<EV, Ix1>,

    /// Eigenvectors for the left-eigenvalue problem ($x^H A = \lambda x^H$)
    pub left_vectors: Option<Array<IV, Ix2>>,

    /// Eigenvectors for the right-eigenvalue problem ($Ax = \lambda x$)
    ///
    /// For symmetric eigenvalue problems, the eigenvectors will be stored
    /// in `right_vectors`.
    pub right_vectors: Option<Array<IV, Ix2>>,
}
