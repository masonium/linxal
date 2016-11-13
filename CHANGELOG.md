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

