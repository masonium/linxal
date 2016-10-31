# rula [![Build Status](https://travis-ci.org/masonium/rula.svg?branch=master)](https://travis-ci.org/masonium/rula) #
`rula` is a linear algebra package for rust. `rula` uses LAPACK as a
backend, (specifically with the `lapack` package) to execute linear
algebra routines with `rust-ndarray` as inputs and outputs.

## Priorities ##
- Correctness: `rula` will strive for correctness in all cases. Any
  function returning a non-`Err` result should return a correct
  result.
- Ease of Use: `rula` will provide a consistent interface and should
  require minimal setup. Most routine should be high-level and should
  require no knowledge of the underlying LAPACK routines.

  `rula` will minimize surprising behaviour.

- Documentation: `rula` will strive to provide documentation for all
  functionality.

- Speed

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
