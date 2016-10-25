//! # rula
//!
//! `rula` is a linear algebra package on top of `ndarray`.

extern crate ndarray;
extern crate lapack;

pub mod matrix;

pub mod eigenvalues;
pub mod svd;
pub mod prelude;
