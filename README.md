# linxal #

## Status ##
[![Crate Version](https://img.shields.io/crates/v/linxal.svg)](https://crates.io/crates/linxal)
[![Build Status](https://travis-ci.org/masonium/linxal.svg?branch=master)](https://travis-ci.org/masonium/linxal)


## Description ##

`linxal` is a linear algebra package for rust. `linxal` uses LAPACK as a
backend, (specifically with the `lapack` package) to execute linear
algebra routines with `rust-ndarray` as inputs and outputs.

## Documentation ##
Documentation can be found at [https://github.masonium.io/rustdoc/linxal/](https://masonium.github.io/rustdoc/linxal).

## Example ##
```rust
#[macro_use]
extern crate linxal;
extern crate ndarray;

use linxal::eigenvalues::{Eigen};
use linxal::types::{c32, Magnitude};
use ndarray::{Array, arr1, arr2};

fn main() {
	let mut m = arr2(&[[1.0f32, 2.0],
					   [-2.0, 1.0]]);

	let r = Eigen::compute_mut(&mut m, false, true);
	assert!(r.is_ok());

	let r = r.unwrap();
	let true_evs = arr1(&[c32::new(1.0, 2.0), c32::new(1.0, -2.0)]);
	assert_eq_within_tol!(true_evs, r.values, 0.01);
}
```

## Priorities ##
- Correctness: `linxal` will strive for correctness in all cases. Any
  function returning a non-`Err` result should return a correct
  result.
- Ease of Use: `linxal` will provide a consistent interface and should
  require minimal setup. Most routine should be high-level and should
  require no knowledge of the underlying LAPACK routines.

  `linxal` will minimize surprising behaviour.

- Documentation: `linxal` will strive to provide documentation for all
  functionality. Undocumented public features are a bug.

- Ergonomics: `linxal` will try to minimize boilerplate whenever
  appropriate.

- Speed

## Non-Goals ##
- Low-dimension arithmetic: `linxal` is not specifically designed or
  optimized for {2,3,4}-D problems, as you would encounter in computer
  graphics, physics, or other domains. There are libraries such
  as [`nalgebra`](https://crates.io/crates/nalgebra)
  and [`cgmath`](https://crates.io/crates/cgmath) that specialize in
  low-dimensional algorithms.

- Representation flexibility: `ndarray` is the only for standard
  matrices, and future representations of specialized formats (packed
  triangular, banded, tridiagonal, etc.) will probably not allow for
  user-defined formats.

## Goals ##
- [ ] Major Linear algbra routines
  - [X] Eigenvalues
  - [X] Singular Value
  - [X] Linear Solvers
  - [X] Linear Least-Squares
  - [ ] Matrix Factorizations (QR, LU, etc.)
  - [ ] Generalized Eigenvalues
  - [ ] Generalized Singular Value Decomposition
- [ ] Multiple matrix formats
  - [X] General (direct via `ndarray`)
  - [ ] Symmetric / Hermitian
  - [ ] Banded (Packed)

## Contributing ##
Pull requests of all kinds (code, documentation, formatting, spell-checks) are welcome!
