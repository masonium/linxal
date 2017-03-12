//! Types for inputs and outputs in matrix-generating functions.

use impl_prelude::*;
use rand::{Rng};

/// The packing type for a generated matrix.
///
/// For symmetric matrices,, the result can sometimes be
#[repr(u8)]
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Packing {
    /// All of the entries of the matrix will be returned.
    Full = b'N',

    /// Only the upper triangular portion of the matrix is filled in,
    /// with the rest as zeros.
    UpperOnly = b'U',

    /// Only the lower triangular portion of the matrix is filled in,
    /// with the rest as zeros.
    LowerOnly = b'L'
}

/// Errors when attempting to generate a random matrix.
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum GenerateError {
    /// The matrix doesn't have square dimensions, but a square
    /// dimension is required for a certain operation.
    NotSquare,

    /// Incorrect number of eigenvalues or singular values
    NotEnoughValues,

    /// The matrix is square, but the number of bands is not the same
    /// on both sides.
    UnequalBands,

    /// The matrix is not symmetric, but a non-`Full` packing was used.
    InvalidPacking,

    /// The rank is larger than the size of the matrix can support.
    InvalidRank,

    /// Invalid Parameter
    IllegalParameter(i32),
}

/// Create a new seed for matrix generation.
pub fn new_seed<Rand: Rng>(rng: &mut Rand) -> [i32; 4] {
    [rng.gen::<u16>() as i32 % 4096, rng.gen::<u16>() as i32 % 4096,
     rng.gen::<u16>() as i32 % 4096, (rng.gen::<u16>() as i32 * 2 + 1) % 4096]
}

/// Create a new workspace for the matrix generating function based on
/// the size of the matrix.
pub fn new_workspace<T: LinxalImplScalar>(m: usize, n: usize) -> Array<T, Ix1> {
    Array::default(cmp::max(m, n) * 3)
}
