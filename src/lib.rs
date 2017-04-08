//! # Description
//!
//! `linxal` is a linear algebra package on top of `ndarray`.It
//! currently provides major drivers from LAPACK, but will also
//! support other higher-level tasks in the future, such as linear
//! regression, PCA, etc.
//!
//! The repository for `linxal` can be found
//! [here](https://github.com/masonium/linxal).
//!
//! # Uasge
//!
//! linxal is available as a crate through cargo. Add the following line
//! to your Cargo.toml, in the `dependencies` section:
//!
//! ```text
//! [dependencies]
//! ...
//! linxal = "0.5"
//! ```
//!
//! In your `lib.rs` or `main.rs` file, use
//!
//! ```text
//! extern crate linxal;
//! use linxal::prelude::*;
//! ```
//!
//! The [`linxal::prelude`](./prelude) modules re-exports the most useful functionality.
//!
//! # Organization
//!
//! Most of the useful functionality for `linxal` comes in the form of
//! traits, which are implemented in terms of scalars and provide
//! functionality for matrices and vectors composed of the
//! scalars.
//!
//! The `LinxalMatrix` trait, defined on two-dimensional `ndarray`
//! arrays, contains most of the computational funcationality in
//! `linxal`.
//!
//! ```rust
//! #[macro_use]
//! extern crate linxal;
//! extern crate ndarray;
//!
//! use linxal::types::{c32, LinxalMatrix};
//! use ndarray::{arr1, arr2};
//!
//! fn main() {
//!     let m = arr2(&[[1.0f32, 2.0],
//!                    [-2.0, 1.0]]);
//!
//!     let r = m.eigenvalues();
//!     assert!(r.is_ok());
//!
//!     let r = r.unwrap();
//!     let true_evs = arr1(&[c32::new(1.0, 2.0), c32::new(1.0, -2.0)]);
//!     assert_eq_within_tol!(true_evs, r, 0.01);
//!
//!     let b = arr1(&[-1.0, 1.0]);
//!     let x = m.solve_linear(&b).unwrap();
//!     let true_x = arr1(&[-0.6, -0.2]);
//!     assert_eq_within_tol!(x, true_x, 0.0001);
//! }
//! ```
//!
//! Most functionality is implemented in terms of specific traits
//! defined on scalars, representing computational routines. These
//! traits typically have a `compute` function, and variants, which
//! performs the describe behavior.
//!
//! For instance, the `Eigen` trait, implemented for single- and
//! double-precision real and complex-valued scalars, allows one to
//! compute eigenvalues and eigenvectors of square matrices with that
//! type of scalar as elements. The above example can be implemented
//! in terms of individual computational routines, as follows:
//!
//! ```rust
//! #[macro_use]
//! extern crate linxal;
//! extern crate ndarray;
//!
//! use linxal::eigenvalues::{Eigen};
//! use linxal::solve_linear::{SolveLinear};
//! use linxal::types::{c32, LinxalScalar};
//! use ndarray::{arr1, arr2};
//!
//! fn main() {
//!     let m = arr2(&[[1.0f32, 2.0],
//!                    [-2.0, 1.0]]);
//!
//!     let r = Eigen::compute(&m, false, false);
//!     assert!(r.is_ok());
//!
//!     let r = r.unwrap();
//!     let true_evs = arr1(&[c32::new(1.0, 2.0), c32::new(1.0, -2.0)]);
//!     assert_eq_within_tol!(true_evs, r.values, 0.01);
//!
//!     let b = arr1(&[-1.0, 1.0]);
//!     let x = SolveLinear::compute(&m, &b).unwrap();
//!     let true_x = arr1(&[-0.6, -0.2]);
//!     assert_eq_within_tol!(x, true_x, 0.0001);
//! }
//! ```
//!
//! # Details
//!
//! ## Prelude
//!
//! In practice, you can use the prelude to gain access to the most
//! common features, rather than having to include computational
//! traits individually.
//!
//! For instance, the previous example's `use`s could be replaced by:
//!
//! ```rust
//! use linxal::prelude::*;
//! ```
//!
//! For reference, all tests and examples will include the specific
//! required traits, but this precision is rarely necessary.
//!
//! ## Symmetric Algorithms
//!
//! Some traits and algorithms are designed only to work on symmetric
//! or Hermititan matrices. Throught the library, 'Sym' or 'Symmetric'
//! refers simply to symmetric matrices for real-valued matrices and
//! Hermititan matrices for complex-valued matrices.
//!
//! Symmetric algorithms typically take a (`Symmetric`) enum
//! argument. `Symmetric::Upper` indicates that the values of the
//! matrix are stored in the upper-triangular portion of the
//! matrix. `Symmetric::Lower` corresponds to the lower portion. For
//! algorithms that take this argument, only that portion is read. So,
//! for example:
//!
//! ```rust
//! #[macro_use]
//! extern crate linxal;
//! extern crate ndarray;
//!
//! use ndarray::{arr1, arr2};
//! use linxal::types::{Symmetric};
//! use linxal::eigenvalues::{SymEigen};
//!
//! fn main() {
//!     // `upper_only` is not symmetric, but the portion below the diagonal is  never read.
//!     let upper_only = arr2(&[[1.0f32, 2.0], [-3.0, 1.0]]);
//!
//!     // Since only the upper triangle is read by `SymEigen`, it is equivalent to `full`.
//!     let full = arr2(&[[1.0f32, 2.0], [2.0, 1.0]]);
//!
//!     let upper_only_ev = SymEigen::compute_into(upper_only, Symmetric::Upper).unwrap();
//!     let full_ev = SymEigen::compute_into(full, Symmetric::Upper).unwrap();
//!
//!     assert_eq_within_tol!(upper_only_ev, full_ev, 1e-5);
//! }
//! ```
//!

#![macro_use]

#[macro_use]
extern crate ndarray;

extern crate libc;
extern crate lapack_sys;
extern crate lapack;
extern crate num_traits;
extern crate rand;

pub mod util;
pub mod permute;

pub mod eigenvalues;
pub mod svd;
pub mod solve_linear;
pub mod least_squares;
pub mod types;
pub mod factorization;
pub mod generate;
pub mod properties;

#[macro_use]
pub mod prelude;

mod impl_prelude;
