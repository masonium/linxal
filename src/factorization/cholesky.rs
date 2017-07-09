//! Compute the Cholesky-factorization of a rectangular matrix.
//!
//! An (n x n) symmetric, positive definite matrix `A` is factored
//! into the product `L` * `L^H` = `A`, for a lower-triangular matrix
//! `L`.

use impl_prelude::*;
use lapack::c::{spotrf, cpotrf, dpotrf, zpotrf};
use util::external::make_triangular_into;

/// Error for Cholesky-based computations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CholeskyError {
    /// The layout of the matrix is not compatible
    BadLayout,

    /// The matrix is not square.
    NotSquare,

    /// The matrix is not positive definite.
    NotPositiveDefinite,

    /// Implementation error, please submit as bug.
    IllegalParameter(i32),
}

/// Trait defined on scalars to support Cholesky-factorization.
pub trait Cholesky: LinxalImplScalar {
    /// Return a triangular matrix satisfying the Cholesky
    /// factorization, consuming the input.
    ///
    /// The layout of the matrix matches the layout of the input. When
    /// the input matrix is specified as `Symmetric::Upper`, the
    /// returned Cholesky factor `U` is upper triangular, so that
    /// `U^H` * `U` = `A`.
    ///
    /// Likewise, when the input is `Symmetric::Lower`, `L` is
    /// returned so that `L` * `L^H` = `A`.
    ///
    /// The elements outside of the triangle are forcibly zero-ed.
    fn compute_into<D>(a: ArrayBase<D, Ix2>, uplo: Symmetric)
                       -> Result<ArrayBase<D, Ix2>, CholeskyError>
        where D: DataOwned<Elem=Self> + DataMut<Elem=Self>;

    /// Return a triangular matrix satisfying the Cholesky
    /// factorization. (see `Self::compute_into`).
    fn compute<D1: Data>(a: &ArrayBase<D1, Ix2>, uplo: Symmetric)
                         -> Result<Array<Self, Ix2>, CholeskyError>
        where D1: Data<Elem = Self>
    {
        Self::compute_into(a.to_owned(), uplo)
    }
}

macro_rules! impl_cholesky {
    ($chol_type:ty, $chol_func:ident) => (

        impl Cholesky for $chol_type {
            fn compute_into<D>(mut a: ArrayBase<D, Ix2>, uplo: Symmetric)
                               -> Result<ArrayBase<D, Ix2>, CholeskyError>
                where D: DataOwned<Elem=Self> + DataMut<Elem=Self> {
                let dim = a.dim();
                if dim.0 != dim.1 {
                    return Err(CholeskyError::NotSquare);
                }

                let info = {
                    let (mut slice, layout, lda) = match slice_and_layout_mut(&mut a) {
                        None => return Err(CholeskyError::BadLayout),
                        Some(x) => x,
                    };

                    // workspace query
                    unsafe {
                        $chol_func(layout,
                                   uplo as u8,
                                   dim.0 as i32,
                                   &mut slice,
                                   lda as i32)
                    }
                };

                if info == 0 {
                    Ok(make_triangular_into(a, uplo))
                } else if info < 0 {
                    Err(CholeskyError::IllegalParameter(-info))
                } else {
                    Err(CholeskyError::NotPositiveDefinite)
                }
            }
        }
    )
}

impl_cholesky!(f32, spotrf);
impl_cholesky!(f64, dpotrf);
impl_cholesky!(c32, cpotrf);
impl_cholesky!(c64, zpotrf);
