//! Define matrix traits for performing linear algebra operations.

use eigenvalues::{self, Eigen, SymEigen};
use solve_linear::{SolveLinear, SymmetricSolveLinear};
use least_squares::{LeastSquares, LeastSquaresType, LeastSquaresSolution};
use super::error::*;
use super::scalar::LinxalScalar;
use impl_prelude::*;
use factorization::{QR, QRFactors, LU, LUFactors, Cholesky};
use svd::{SVD, SVDSolution, SVDComputeVectors};
use properties::{self, default_tol};

/// All-encompassing matrix trait, supporting all of the linear
/// algebra operations defined for any `LinxalScalar`.
pub trait LinxalMatrix<F: LinxalScalar> {
    /// Compute the eigenvalues of a matrix.
    fn eigenvalues(&self) -> Result<Array<F::Complex, Ix1>, EigenError>;

    /// Compute the eigenvalues and eigenvectors of a matrix.
    fn eigenvalues_vectors(&self,
                           compute_left: bool,
                           compute_right: bool)
                           -> Result<eigenvalues::Solution<F, F::Complex>, EigenError>;

    /// Compute the eigenvalues of a symmetric matrix.
    fn symmetric_eigenvalues(&self, uplo: Symmetric)
                             -> Result<Array<F::RealPart, Ix1>, EigenError>;


    /// Compute the eigenvalues and eigenvectors of a symmetric matrix.
    fn symmetric_eigenvalues_vectors(&self, uplo: Symmetric)
                                     -> Result<eigenvalues::Solution<F, F::RealPart>, EigenError>;

    /// Solve a single system of linear equations.
    fn solve_linear<D1: Data<Elem = F>>(&self,
                                        b: &ArrayBase<D1, Ix1>)
                                        -> Result<Array<F, Ix1>, SolveError>;

    /// Solve a single system of linear equations with a symmetrix coefficient matrix.
    fn solve_symmetric_linear<D1: Data<Elem = F>>(&self,
                                                  b: &ArrayBase<D1, Ix1>,
                                                  uplo: Symmetric)
                                                  -> Result<Array<F, Ix1>, SolveError>;


    /// Solve a system of linear equations with multiple RHS vectors.
    fn solve_multi_linear<D1: Data<Elem = F>>(&self,
                                              b: &ArrayBase<D1, Ix2>)
                                              -> Result<Array<F, Ix2>, SolveError>;

    /// Solve a system of linear equations with a symmetrix
    /// coefficient matrix for multiple RHS vectors.
    ///
    /// Each column of `b` is a RHS vector to be solved for.
    fn solve_symmetric_multi_linear<D1: Data<Elem = F>>(&self,
                                                        b: &ArrayBase<D1, Ix2>,
                                                        uplo: Symmetric)
                                                        -> Result<Array<F, Ix2>, SolveError>;

    /// Compute the least squares solution for a single RHS.
    fn least_squares<D1, PT>(&self,
                             b: &ArrayBase<D1, Ix1>,
                             problem_type: PT)
                             -> Result<LeastSquaresSolution<F, Ix1>, LeastSquaresError>
        where D1: Data<Elem = F>,
              PT: Into<Option<LeastSquaresType>>;

    /// Compute the least squares solution for a multiple RHS.
    fn multi_least_squares<D1, PT>(&self,
                                   b: &ArrayBase<D1, Ix2>,
                                   problem_type: PT)
                                   -> Result<LeastSquaresSolution<F, Ix2>, LeastSquaresError>
        where D1: Data<Elem = F>,
              PT: Into<Option<LeastSquaresType>>;

    /// Return the QR factorization of the matrix.
    ///
    /// See [QR::compute]().
    fn qr(&self) -> Result<QRFactors<F>, QRError>;

    /// Return the LU factorization of the matrix.
    ///
    /// See [LU::compute]()
    fn lu(&self) -> Result<LUFactors<F>, LUError>;

    /// Return the cholesky factorization of the matrix, via the
    /// upper- or lower-triangular matrix defining it.
    fn cholesky(&self, uplo: Symmetric) -> Result<Array<F, Ix2>, CholeskyError>;

    /// Return the full singular value decomposition of the matrix.
    ///
    /// The `SVDSolution` contains full size matrices `u` (m x m) and `vt` (n x n).
    fn svd_full(&self) -> Result<SVDSolution<F>, SVDError>;

    /// Return the economic singular value decomposition of the matrix.
    ///
    /// The `SVDSolution` contains sufficient matrices `u` (m x p) and
    /// `vt` (p x n), where `p` is the minimum of `m` and `n`.
    fn svd_econ(&self) -> Result<SVDSolution<F>, SVDError>;

    /// Return the full singular value decomposition of the matrix.
    ///
    /// The `SVDSolution` contains full size matrices `u` (m x m) and `vt` (n x n).
    fn singular_values(&self) -> Result<Array<F::RealPart, Ix1>, SVDError>;

    /// Return the inverse of the matrix, if it has one.
    fn inverse(&self) -> Result<Array<F, Ix2>, Error>;

    /// Return the conjugate of the matrix.
    fn conj(&self) -> Array<F, Ix2>;

    /// Return the cojugate transpose of the matrix.
    fn conj_t(&self) -> Array<F, Ix2>;

    // *** property methods ***//
    /// Returns true iff the matrix is square.
    fn is_square(&self) -> bool;

    /// Returns true iff the matrix is the identity matrix, within an
    /// optional tolerance.
    fn is_identity<T: Into<Option<F::RealPart>>>(&self, tolerance: T) -> bool;

    /// Returns true iff the matrix is diagnoal, within a certain
    /// tolerance.
    fn is_diagonal<T: Into<Option<F::RealPart>>>(&self, tolerance: T) -> bool;

    /// Returns true iff the matrix is symmetric (or Hermitian, in the
    /// complex case), within an optional tolerance.
    fn is_symmetric<T: Into<Option<F::RealPart>>>(&self, tolerance: T) -> bool;

    /// Returns true iff the matrix is unitary, within an optional
    /// tolerance.
    fn is_unitary<T: Into<Option<F::RealPart>>>(&self, tolerance: T) -> bool;

    /// Returns true iff the matrix is triangular or trapezoidal, with
    /// size specified by `uplo`.
    fn is_triangular<T: Into<Option<F::RealPart>>>(&self, uplo: Symmetric, tolerance: T) -> bool;
}

impl<F: LinxalScalar, D: Data<Elem = F>> LinxalMatrix<F> for ArrayBase<D, Ix2> {
    fn eigenvalues(&self) -> Result<Array<F::Complex, Ix1>, EigenError> {
        Eigen::compute(self, false, false).map(|s| {
            let sol: eigenvalues::types::Solution<_, _> = s;
            sol.values
        })
    }

    fn eigenvalues_vectors(&self,
                           compute_left: bool,
                           compute_right: bool)
                           -> Result<eigenvalues::Solution<F, F::Complex>, EigenError> {
        Eigen::compute(self, compute_left, compute_right)
    }


    fn symmetric_eigenvalues(&self, uplo: Symmetric)
                             -> Result<Array<F::RealPart, Ix1>, EigenError> {
        SymEigen::compute(self, uplo, false).map(|s| s.values)
    }


    fn symmetric_eigenvalues_vectors(&self, uplo: Symmetric)
                             -> Result<eigenvalues::Solution<F, F::RealPart>, EigenError> {
        SymEigen::compute(self, uplo, true)
    }

    fn solve_linear<D1: Data<Elem = F>>(&self,
                                        b: &ArrayBase<D1, Ix1>)
                                        -> Result<Array<F, Ix1>, SolveError> {
        SolveLinear::compute(self, b)
    }

    fn solve_symmetric_linear<D1: Data<Elem = F>>(&self,
                                                  b: &ArrayBase<D1, Ix1>,
                                                  uplo: Symmetric)
                                                  -> Result<Array<F, Ix1>, SolveError> {
        SymmetricSolveLinear::compute(self, uplo, b)
    }


    fn solve_multi_linear<D1: Data<Elem = F>>(&self,
                                              b: &ArrayBase<D1, Ix2>)
                                              -> Result<Array<F, Ix2>, SolveError> {
        SolveLinear::compute_multi(self, b)
    }

    fn solve_symmetric_multi_linear<D1: Data<Elem = F>>(&self,
                                                        b: &ArrayBase<D1, Ix2>,
                                                        uplo: Symmetric)
                                                        -> Result<Array<F, Ix2>, SolveError> {
        SymmetricSolveLinear::compute_multi(self, uplo, b)
    }

    fn least_squares<D1, PT>(&self,
                             b: &ArrayBase<D1, Ix1>,
                             _problem_type: PT)
                             -> Result<LeastSquaresSolution<F, Ix1>, LeastSquaresError>
        where D1: Data<Elem = F>,
              PT: Into<Option<LeastSquaresType>>
    {
        LeastSquares::compute(self, b)
    }

    fn multi_least_squares<D1, PT>(&self,
                                   b: &ArrayBase<D1, Ix2>,
                                   problem_type: PT)
                                   -> Result<LeastSquaresSolution<F, Ix2>, LeastSquaresError>
        where D1: Data<Elem = F>,
              PT: Into<Option<LeastSquaresType>>
    {

        match problem_type.into() {
            Some(LeastSquaresType::Degenerate) => LeastSquares::compute_multi_degenerate(self, b),
            Some(LeastSquaresType::Full) => LeastSquares::compute_multi_full(self, b),
            None => LeastSquares::compute_multi(self, b),
        }
    }

    fn qr(&self) -> Result<QRFactors<F>, QRError> {
        QR::compute(self)
    }

    fn lu(&self) -> Result<LUFactors<F>, LUError> {
        LU::compute(self)
    }

    fn cholesky(&self, uplo: Symmetric) -> Result<Array<F, Ix2>, CholeskyError> {
        Cholesky::compute(self, uplo)
    }

    fn svd_full(&self) -> Result<SVDSolution<F>, SVDError> {
        SVD::compute(self, SVDComputeVectors::Full)
    }

    fn svd_econ(&self) -> Result<SVDSolution<F>, SVDError> {
        SVD::compute(self, SVDComputeVectors::Economic)
    }

    fn singular_values(&self) -> Result<Array<F::RealPart, Ix1>, SVDError> {
        SVD::compute(self, SVDComputeVectors::None).map(|x| x.values)
    }

    fn inverse(&self) -> Result<Array<F, Ix2>, Error> {
        match LU::compute(self) {
            Ok(factors) => factors.inverse_into().map_err(|x| x.into()),
            Err(lu_error) => Err(lu_error.into()),
        }
    }

    fn conj(&self) -> Array<F, Ix2> {
        self.mapv(|x| x.cj())
    }

    fn conj_t(&self) -> Array<F, Ix2> {
        self.t().conj()
    }

    fn is_square(&self) -> bool {
        match self.dim() {
            (a, b) => a == b,
        }
    }

    fn is_diagonal<T: Into<Option<F::RealPart>>>(&self, tolerance: T) -> bool {
        let tol: F::RealPart = tolerance.into().unwrap_or_else(|| default_tol(self));
        properties::is_diagonal_tol(self, tol)
    }

    fn is_symmetric<T: Into<Option<F::RealPart>>>(&self, tolerance: T) -> bool {
        let tol: F::RealPart = tolerance.into().unwrap_or_else(|| default_tol(self));
        properties::is_symmetric_tol(self, tol)
    }

    fn is_identity<T: Into<Option<F::RealPart>>>(&self, tolerance: T) -> bool {
        let tol: F::RealPart = tolerance.into().unwrap_or_else(|| default_tol(self));
        properties::is_identity_tol(self, tol)
    }

    fn is_unitary<T: Into<Option<F::RealPart>>>(&self, tolerance: T) -> bool {
        let tol: F::RealPart = tolerance.into().unwrap_or_else(|| default_tol(self));
        properties::is_unitary_tol(self, tol)
    }

    fn is_triangular<T: Into<Option<F::RealPart>>>(&self, uplo: Symmetric, tolerance: T) -> bool {
        let tol: F::RealPart = tolerance.into().unwrap_or_else(|| default_tol(self));
        properties::is_triangular_tol(self, uplo, tol)
    }
}
