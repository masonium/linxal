//! Containts traits and methods to solve sets of linear equations.
//!
//! The primary content of this modules is the `SolveLinear` trait,
//! which computes solutions `X` to the linear system A*X = B, for
//! square matrices A.

pub mod types;
pub mod general;
pub mod symmetric;

pub use self::types::SolveError;
pub use self::general::SolveLinear;
pub use self::symmetric::SymmetricSolveLinear;
