//! Define scalar types for matrix usage.

use eigenvalues::{Eigen, SymEigen};
use solve_linear::{SolveLinear, SymmetricSolveLinear};
use least_squares::LeastSquares;
use num_traits::Float;
use impl_prelude::*;
use factorization::{QR, LU, Cholesky};
use svd::SVD;
use generate::matgen::MG;

/// Catch-all aggregate trait for computational routines needed by
/// `LinxalMatrix`.
pub trait LinxalScalar: LinxalImplScalar + Eigen + SymEigen + SolveLinear + SymmetricSolveLinear +
    LeastSquares + QR + LU + Cholesky + SVD + MG {}
impl<T: LinxalImplScalar + Eigen + SymEigen + SolveLinear + SymmetricSolveLinear +
     LeastSquares + QR + LU + Cholesky + SVD + MG> LinxalScalar for T {}

/// Narrowing trait for `LinxalScalar`s that are also real.
pub trait LinxalReal: LinxalScalar + Float {}
impl<T: LinxalScalar + Float> LinxalReal for T {}

/// Narrowing trait for `LinxalScalar`s that are also complex.
pub trait LinxalComplex: LinxalScalar {}
impl LinxalComplex for c32 {}
impl LinxalComplex for c64 {}
