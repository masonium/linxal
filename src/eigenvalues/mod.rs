//! Contains methods for solving eigenvalues, including general and
//! symmetric/Hermitian eigenvalue problems.
//!
//! The eigenvalue problem is to find solution $\left(x, \lambda\right)$ to the problem:
//!
//! $$A \cdot x = \lambda \cdot x$$
//!
//! for a square matrix \\(A\\).
#![deny(missing_docs)]


pub mod general;
pub mod symmetric;
pub mod types;

pub use self::types::{Solution, EigenError};
pub use self::general::{Eigen};
pub use self::symmetric::{SymEigen};
