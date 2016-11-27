//! Generate random matrices.

#![deny(missing_docs)]


mod ffi;
mod internal;
mod types;
pub mod matgen;

pub use self::types::{Packing, GenerateError};
pub use self::matgen::{MG, RandomUnitary, RandomSemiPositive, RandomSymmetric, RandomGeneral};
