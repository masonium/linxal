0.4.2:
 - Cholesky factorization
 - depcrecated `RandomGeneral::bands` due to possible `?latmt` bug

0.4.1:
 - Add `GenerateError` option to `LinxalError`

0.4.0:
 - matrix generation API (`RandomGeneral`, `RandomPositive`, `RandomSymmetric`, `RandomUnitary`)
 - matrix properties (`is_*` tests, bandwidth check)
 - restructured `LinxalScalar` (additional traits, remove `Magnitude` trait)
 - simplify `SymEigen` (no associated types, real part inferred from `Self`)
 - simplify `SVD` (no associated types, no generic parameter, real part inferred from `Self`)

0.3.0:
 - Add QR and LU factorizations in `factorization` module
 - Add matrix row permutation module `permute`, to facilitate LU-factorization
 - Normalize eigenvalues error naming
 - Improved documentation across the board

0.2.3:
 - Add `openblas-system` / `netlib-system` features
 - Fix existing `netlib` feature

0.2.2:
 - Changed all relevant `Ix` references to `Ix1` alias (@bluss)
 - Removed old source file

0.2.1:
 - Fixed bug which caused column-major layout matrices to be rejected
   from most major computations
 - Add `poly_interp` example to demonstrate `SolveLinear` trait.
 - Add `LinxalScalar` trait as sugar for features common to all scalar
   inputs for matrix operations.

0.2.0:
 - Incrememnt minor version due to incompatible API changes.
 
0.1.6:
 - Remove all `prelude` and glob imports from the tests and examples.
 - Replace all `*_mut` functions with `*_into` functions in the API,
   to emphasize that values are consumed

