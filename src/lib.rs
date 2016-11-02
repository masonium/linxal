//! # Description
//!
//! `rula` is a linear algebra package on top of `ndarray`.It
//! currently provides major drivers from LAPACK, but will also
//! support other higher-level tasks in the future, such as linear
//! regression, PCA, etc.
//!
//! # Uasge
//!
//! rula is available as a crate through cargo. Add the following line
//! to your Cargo.toml, in the `dependencies` section:
//!
//! ```text
//! [dependencies]
//! ...
//! rula = "0.1"
//! ```
//!
//! In your `lib.rs` or `main.rs` file, use
//!
//! ```text
//! extern crate rula;
//! use rula::prelude::*;
//! ```
//!
//! The [`rula::prelude`](./prelude) modules re-exports the most useful functionality.
//!
//! # Organization
//!
//! Most of the useful functionality for `rula` comes in the form of
//! traits, which are implemented in terms of scalars and provide
//! functionality for matrices and vectors composed of the
//! scalars. Most traits have a `compute` function, and variants,
//! which performs the describe behavior.
//!
//! For instance, the `Eigen` trait, implemented for single- and
//! double-precision real and complex-valued matrices, allows one to
//! compute eigenvalues and eigenvectors of square matrices.
//!
//! ```rust
//! #[macro_use]
//! extern crate rula;
//! extern crate ndarray;
//!
//! use rula::prelude::*;
//! use ndarray::prelude::*;
//!
//! fn main() {
//!     let mut m = arr2(&[[1.0 as f32, 2.0],
//!                        [-2.0, 1.0]]);
//!
//!     let r = Eigen::compute_mut(&mut m, false, true);
//!     assert!(r.is_ok());
//!
//!     let r = r.unwrap();
//!     let true_evs = Array::from_vec(vec![c32::new(1.0, 2.0), c32::new(1.0, -2.0)]);
//!     assert_eq_within_tol!(r.values, true_evs, 0.0001);
//! }
//! ```
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
