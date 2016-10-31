//! # rula
//!
//! `rula` is a linear algebra package on top of `ndarray`.It
//! currently provides major drivers from LAPACK, but will also
//! support other higher-level tasks in the future, such as linear
//! regression, PCA, etc.
//!
//! # Uasge
//!
//! rula is available as a crate through cargo.
#![macro_use]

#[macro_use]
extern crate ndarray;
extern crate lapack;
extern crate num_traits;

pub mod util;

pub mod eigenvalues;
pub mod svd;
pub mod solve_linear;
pub mod least_squares;
pub mod types;

#[macro_use]
pub mod prelude;

mod impl_prelude;
