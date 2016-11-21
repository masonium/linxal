use impl_prelude::*;
use rand::{Rng};

#[repr(u8)]
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Packing {
    Full = b'N',
    UpperOnly = b'U',
    LowerOnly = b'L'
}

pub enum ValuesOption<T> {
    EvenUniform(T, T),
    Exact(Vec<T>)
}

#[repr(u8)]
#[derive(PartialEq, Eq, Clone, Copy)]
pub enum GenerateSymmetry {
    Positive = b'P',
    Symmetric = b'H',
    NoSymmetry = b'N'
}

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum GenerateError {
    NotSquare,

    // Incorrect number of eigenvalues or singular values
    NotEnoughValues,

    // Rank
    InvalidRank,

    UnequalBands,
    BadUpperBand,

    /// The packing does not match the shape of the matrix.
    InvalidPacking,
    EigenvalueGeneration,
    EigenvalueScale,
    RawMatrixGeneration,

    IllegalParameter(i32),
}

/// Create a new seed for matrix generation.
pub fn new_seed<Rand: Rng>(rng: &mut Rand) -> [i32; 4] {
    [rng.gen::<u16>() as i32, rng.gen::<u16>() as i32,
     rng.gen::<u16>() as i32, rng.gen::<u16>() as i32 * 2 + 1]
}

/// Create a new workspace for the matrix generating function based on
/// the size of the matrix.
pub fn new_workspace<T>(m: usize, n: usize) -> Vec<T> {
    Vec::with_capacity(cmp::max(m, n) * 3)
}
