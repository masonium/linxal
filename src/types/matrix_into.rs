//! Define matrix traits for performing linear algebra operations.

use eigenvalues::{Eigen, SymEigen};
use solve_linear::{SolveLinear, SymmetricSolveLinear};
use super::error::*;
use super::scalar::LinxalScalar;
use impl_prelude::*;
use svd::{SVD, SVDSolution};

/// All-encompassing matrix trait, supporting all of the linear
/// algebra operations defined for any `LinxalScalar`.
pub trait LinxalMatrixInto<F: LinxalScalar> {
    /// Compute the eigenvalues of a matrix.
    fn eigenvalues_into(self,
                        compute_left: bool,
                        compute_right: bool)
                        -> Result<<F as Eigen>::Solution, EigenError>;

    /// Compute the eigenvalues of a symmetric matrix
    fn symmetric_eigenvalues_into(self,
                                  uplo: Symmetric)
                                  -> Result<Array<F::RealPart, Ix1>, EigenError>;

    /// Solve a single system of linear equations.
    fn solve_linear_into<D1: DataMut<Elem = F> + DataOwned<Elem = F>>
        (self, b: ArrayBase<D1, Ix1>)
         -> Result<ArrayBase<D1, Ix1>, SolveError>;

    /// Solve a single system of linear equations with a symmetrix coefficient matrix.
    fn solve_symmetric_linear_into<D1: DataMut<Elem = F> + DataOwned<Elem = F>>
        (self, b: ArrayBase<D1, Ix1>, uplo: Symmetric)
         -> Result<ArrayBase<D1, Ix1>, SolveError>;


    /// Solve a system of linear equations with multiple RHS vectors.
    fn solve_multi_linear_into<D1: DataMut<Elem = F> + DataOwned<Elem = F>>
        (self, b: ArrayBase<D1, Ix2>)
         -> Result<ArrayBase<D1, Ix2>, SolveError>;

    /// Solve a system of linear equations with a symmetrix
    /// coefficient matrix for multiple RHS vectors.
    ///
    /// Each column of `b` is a RHS vector to be solved for.
    fn solve_symmetric_multi_linear_into<D1: DataMut<Elem = F> + DataOwned<Elem = F>>
        (self, b: ArrayBase<D1, Ix2>, uplo: Symmetric)
         -> Result<ArrayBase<D1, Ix2>, SolveError>;

    /// Return the singular value decomposition of the matrix.
    fn svd_into(self, compute_u: bool, compute_vt: bool) -> Result<SVDSolution<F>, SVDError>;

    /// Return the conjugate of the matrix.
    fn conj_into(self) -> Self;
}

impl<F: LinxalScalar, D: DataMut<Elem = F> + DataOwned<Elem = F>> LinxalMatrixInto<F> for ArrayBase<D, Ix2> {
    fn eigenvalues_into(self,
                        compute_left: bool,
                        compute_right: bool)
                        -> Result<<F as Eigen>::Solution, EigenError> {
        Eigen::compute_into(self, compute_left, compute_right)
    }

    fn symmetric_eigenvalues_into(self,
                                  uplo: Symmetric) -> Result<Array<F::RealPart, Ix1>, EigenError> {
        SymEigen::compute_into(self, uplo)
    }

    fn solve_linear_into<D1: DataMut<Elem = F> + DataOwned<Elem = F>>
        (self, b: ArrayBase<D1, Ix1>)
         -> Result<ArrayBase<D1, Ix1>, SolveError> {
            SolveLinear::compute_into(self, b)
        }

    fn solve_symmetric_linear_into<D1: DataMut<Elem = F> + DataOwned<Elem = F>>
        (self, b: ArrayBase<D1, Ix1>, uplo: Symmetric)
         -> Result<ArrayBase<D1, Ix1>, SolveError> {
            SymmetricSolveLinear::compute_into(self, uplo, b)
        }


    fn solve_multi_linear_into<D1: DataMut<Elem = F> + DataOwned<Elem = F>>
        (self, b: ArrayBase<D1, Ix2>)
         -> Result<ArrayBase<D1, Ix2>, SolveError> {
            SolveLinear::compute_multi_into(self, b)
        }

    fn solve_symmetric_multi_linear_into<D1: DataMut<Elem = F> + DataOwned<Elem = F>>
        (self, b: ArrayBase<D1, Ix2>, uplo: Symmetric)
         -> Result<ArrayBase<D1, Ix2>, SolveError> {
            SymmetricSolveLinear::compute_multi_into(self, uplo, b)
        }

    fn svd_into(self, compute_u: bool, compute_vt: bool) -> Result<SVDSolution<F>, SVDError> {
        SVD::compute_into(self, compute_u, compute_vt)
    }

    fn conj_into(self) -> Self {
        self.mapv_into(|x| x.cj())
    }
}
