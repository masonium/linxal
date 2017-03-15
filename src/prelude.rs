//! Common traits, structures, and macros for most user-end applications

pub use svd::general::SVD;
pub use svd::types::{SVDError, SVDSolution};
pub use eigenvalues::general::Eigen;
pub use eigenvalues::types::EigenError;
pub use eigenvalues::symmetric::SymEigen;
pub use types::{Symmetric, Error, c32, c64};
pub use solve_linear::general::SolveLinear;
pub use solve_linear::symmetric::SymmetricSolveLinear;
pub use least_squares::LeastSquares;
pub use factorization::{Cholesky, QR, LU, QRError, LUError, CholeskyError};

pub use util::external::*;
