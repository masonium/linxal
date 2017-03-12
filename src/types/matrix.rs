//! Create a

use eigenvalues::{Eigen, SymEigen};
use solve_linear::SolveLinear;
use super::error::*;
use impl_prelude::*;

/// Catch-all aggregate trait for computational routines needed by
/// `LinxalMatrix`.
pub trait LinxalMatrixScalar: Eigen + SymEigen + SolveLinear {}
impl<T: Eigen + SymEigen + SolveLinear> LinxalMatrixScalar for T {}

/// All-encompassing matrix trait, supporting all of the linear
/// algebra operations defined for any `LinxalScalar`.
pub trait LinxalMatrix<F: LinxalMatrixScalar> {
    /// Compute the eigenvalues of a matrix.
    fn eigenvalues(&self,
                   compute_left: bool,
                   compute_right: bool)
                   -> Result<<F as Eigen>::Solution, EigenError>;

    /// Compute the eigenvalues of a symmetric matrix
    fn symmetric_eigenvalues(&self,
                             uplo: Symmetric,
                             with_vectors: bool)
                             -> Result<<F as SymEigen>::Solution, EigenError>;

    /// Solve a single system of linear equations.
    fn solve_linear<D1: Data<Elem = F>>(&self,
                                        b: &ArrayBase<D1, Ix1>)
                                        -> Result<Array<F, Ix1>, SolveError>;
}

impl<F: LinxalMatrixScalar, D: Data<Elem = F>> LinxalMatrix<F> for ArrayBase<D, Ix2> {
    fn eigenvalues(&self,
                   compute_left: bool,
                   compute_right: bool)
                   -> Result<<F as Eigen>::Solution, EigenError> {
        Eigen::compute(self, compute_left, compute_right)
    }

    fn symmetric_eigenvalues(&self,
                             uplo: Symmetric,
                             with_vectors: bool)
                             -> Result<<F as SymEigen>::Solution, EigenError> {
        SymEigen::compute(self, uplo, with_vectors)
    }

    fn solve_linear<D1: Data<Elem = F>>(&self,
                                        b: &ArrayBase<D1, Ix1>)
                                        -> Result<Array<F, Ix1>, SolveError> {
        SolveLinear::compute(self, b)
    }
}
