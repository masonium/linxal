//! Contains methods for solving eigenvalues, including general and
//! symmetric/Hermitian eigenvalue problems.
//!

pub mod general;
pub mod symmetric;
pub mod types;

pub use self::general::{Eigen};
pub use self::types::{EigenError};
pub use self::symmetric::{SymEigen};
