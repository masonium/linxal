//! Solve singular value decomposition problems.
//!
//! The simgular value decomposition of an m x n matrix M is are
//! matrices U, D, and V such that.
//!
//! M = U * D * V^T
//!
//! where U is an orthogonal m x m matrix, V is an orthogonal n x n
//! matrix, and D is a real-valued m x n diagonal matrix. The entries
//! along the diagonal of D, called the _singular values_ of M, are
//! eigenvalues of sqrt(M^T * M), and are guaranteed to be real.

pub mod general;
pub mod types;

pub use self::general::SVD;
pub use self::types::{SVDSolution, SVDError, SingularValue};
