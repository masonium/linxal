//! Define matrix traits for performing linear algebra operations.

use eigenvalues::{self, Eigen, SymEigen};
use solve_linear::{SolveLinear, SymmetricSolveLinear};
use super::error::*;
use super::scalar::LinxalScalar;
use impl_prelude::*;
use svd::{SVD, SVDSolution, SVDComputeVectors};

/// All-encompassing matrix trait, supporting all of the linear
/// algebra operations defined for any `LinxalScalar`.
pub trait LinxalMatrixInto<F: LinxalScalar> {
    /// Compute the eigenvalues of a matrix.
    fn eigenvalues_into(self)
                        -> Result<Array<F::Complex, Ix1>, EigenError>;

    /// Compute the eigenvalues and the right and/or left eigenvectors
    /// of a generic matrix.
    fn eigenvalues_vectors_into(self,
                        compute_left: bool,
                        compute_right: bool)
                        -> Result<eigenvalues::types::Solution<F, F::Complex>, EigenError>;

    /// Compute the eigenvalues of a symmetric matrix.
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

    /// Return the full singular value decomposition of the matrix.
    ///
    /// The `SVDSolution` contains full size matrices `u` (m x m) and `vt` (n x n).
    fn svd_full_into(self) -> Result<SVDSolution<F>, SVDError>;

    /// Return the economic singular value decomposition of the matrix.
    ///
    /// The `SVDSolution` contains sufficient matrices `u` (m x p) and
    /// `vt` (p x n), where `p` is the minimum of `m` and `n`.
    fn svd_econ_into(self) -> Result<SVDSolution<F>, SVDError>;

    /// Return the full singular value decomposition of the matrix.
    ///
    /// The `SVDSolution` contains full size matrices `u` (m x m) and `vt` (n x n).
    fn singular_values_into(self) -> Result<Array<F::RealPart, Ix1>, SVDError>;

    /// Return the conjugate of the matrix.
    fn conj_into(self) -> Self;
}

impl<F: LinxalScalar, D: DataMut<Elem = F> + DataOwned<Elem = F>> LinxalMatrixInto<F> for ArrayBase<D, Ix2> {
    fn eigenvalues_into(self) -> Result<Array<F::Complex, Ix1>, EigenError> {
        Eigen::compute_into(self, false, false).map(|sol| sol.values)
    }

    fn eigenvalues_vectors_into(self,
                        compute_left: bool,
                        compute_right: bool)
                        -> Result<eigenvalues::types::Solution<F, F::Complex>, EigenError> {
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

    fn svd_full_into(self) -> Result<SVDSolution<F>, SVDError> {
        SVD::compute_into(self, SVDComputeVectors::Full)
    }

    fn svd_econ_into(self) -> Result<SVDSolution<F>, SVDError> {
        SVD::compute_into(self, SVDComputeVectors::Economic)
    }

    fn singular_values_into(self) -> Result<Array<F::RealPart, Ix1>, SVDError> {
        SVD::compute_into(self, SVDComputeVectors::None).map(|x| x.values)
    }

    fn conj_into(self) -> Self {
        self.mapv_into(|x| x.cj())
    }
}
