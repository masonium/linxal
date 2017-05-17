//! Compute the QR-factorization of a rectangular matrix.
//!
//! An (m x n) rectangular matrix `A` is factored into the produce `Q` * `R`, such that
//!
//! - The columns of `Q` are orthonormal, and the span of the first
//! `k` columns of `Q` contains the subspaces spanned by the first `k`
//! columns of `A`, for all 1 <= `k` <= n.
//!
//! - `R` is an upper triangular or upper-trapezoidal matrix.

use impl_prelude::*;
use lapack::c::{sgeqrf, sorgqr, dgeqrf, dorgqr, cgeqrf, cungqr, zgeqrf, zungqr};
use ndarray as nd;

/// Error for QR-based computations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QRError {
    /// The layout of the matrix is not compatible
    BadLayout,

    /// The dimensions of the raw qr and tau don't match
    InconsistentDimensions,

    /// Implementation error, please submit as bug.
    IllegalParameter(i32),
}

/// Representation of the components Q, R of the factorization of
/// matrix A.
#[derive(Debug)]
pub struct QRFactors<T: QR> {
    mat: Array<T, Ix2>,
    tau: Vec<T>,
}

impl<T: QR> QRFactors<T> {
    /// Create a `QRFactors` object from the output of the LAPACKE
    /// functions.
    fn from_raw<Matrix>(mat: Matrix, tau: Vec<T>) -> Result<QRFactors<T>, QRError>
        where Matrix: Into<Array<T, Ix2>>
    {
        let mut ma = mat.into();

        if slice_and_layout_mut(&mut ma).is_none() {
            return Err(QRError::BadLayout);
        }

        Ok(QRFactors {
            mat: ma,
            tau: tau,
        })
    }

    /// Return the number of rows in the original matrix
    pub fn rows(&self) -> usize {
        self.mat.rows()
    }

    /// Return the number of columns in the original matrix
    pub fn cols(&self) -> usize {
        self.mat.cols()
    }

    fn p(&self) -> usize {
        cmp::min(self.rows(), self.cols())
    }

    /// Return the first `k` columns of the matrix Q of the QR
    /// factorization.
    ///
    /// `Q` is generated such that the columns of `Q` form an
    /// orthogonal basis for the first `k` columns of `A`.
    ///
    /// When `k` is None, compute enough columns (`min(m, n)`) to
    /// faithfully recreate the original matrix A.
    pub fn qk<K: Into<Option<usize>>>(&self, k: K) -> Result<Array<T, Ix2>, QRError> {
        let kr = k.into();
        QR::compute_q(&self.mat, &self.tau, kr.unwrap_or_else(|| self.p()))
    }

    /// Return the `m` by `min(m, n)` matrix `Q`.
    ///
    /// Equivalent to `self.qk(None)`.
    ///
    /// `self.q()` is large enough so that `&self.q() * &self.r()`
    /// will faithfully reproduce the original matrix `A`.
    #[inline]
    pub fn q(&self) -> Array<T, Ix2> {
        self.qk(None).expect("Invalid implementation of Self::qk. Please report.")
    }


    /// Return the first 'k' rows of the matrix R of the QR
    /// factorization.
    ///
    /// When `k` is None, compute enough rows (`min(m, n)`) to
    /// faithfully recreate the original matrix A.
    pub fn rk<K: Into<Option<usize>>>(&self, k: K) -> Result<Array<T, Ix2>, QRError> {
        let kr = k.into();
        let p = kr.unwrap_or_else(|| self.p());
        QR::compute_r(&self.mat, p)
    }

    /// Return the first `min(m, n)` by `n` matrix R of the QR
    /// factorization.
    ///
    /// Equivalent to `self::rk(None)`.
    ///
    /// `self.r()` is large enough so that `&self.q() * &self.r()`
    /// will faithfully reproduce the original matrix `A`.
    #[inline]
    pub fn r(&self) -> Array<T, Ix2> {
        self.rk(None).expect("Invalid implementation of Self::rk. Please report.")
    }

    /// Reconstruct the original matrix `A` from the factorization.
    pub fn reconstruct(&self) -> Array<T, Ix2> {
        self.q().dot(&self.r())
    }
}

/// Trait defined on scalars to support QR-factorization.
pub trait QR: nd::LinalgScalar {
    /// Return a `QRFactors` structure, containing the QR
    /// factorization of the input matrix `A`.
    ///
    /// Similar to `compute`, but consumes the input.
    fn compute_into(a: Array<Self, Ix2>) -> Result<QRFactors<Self>, QRError>;

    /// Return a `QRFactors` structure, containing the QR
    /// factorization of the input matrix `A`.
    fn compute<D1: Data>(a: &ArrayBase<D1, Ix2>) -> Result<QRFactors<Self>, QRError>
        where D1: Data<Elem = Self>
    {
        Self::compute_into(a.to_owned())
    }

    /// Compute Q from raw parts.
    ///
    /// Not intended to be used by end-users.
    fn compute_q<D1>(mat: &ArrayBase<D1, Ix2>,
                     tau: &[Self],
                     k: usize)
                     -> Result<Array<Self, Ix2>, QRError>
        where D1: Data<Elem = Self>;

    /// Compute R from raw parts.
    ///
    /// Not intended to be used by end-users.
    fn compute_r<D1>(mat: &ArrayBase<D1, Ix2>, k: usize) -> Result<Array<Self, Ix2>, QRError>
        where D1: Data<Elem = Self>;
}

macro_rules! impl_qr {
    ($qr_type:ty, $qr_func:ident, $qr_to_q:ident) => (

        impl QR for $qr_type {
            fn compute_into(mut a: Array<Self, Ix2>) -> Result<QRFactors<Self>, QRError> {
                let dim = a.dim();

                let (info, tau) = {
                    let (mut slice, layout, lda) = match slice_and_layout_mut(&mut a) {
                        None => return Err(QRError::BadLayout),
                        Some(x) => x,
                    };

                    let mut tau = Vec::new();
                    tau.resize(cmp::min(dim.0, dim.1), <$qr_type as Zero>::zero());

                    // workspace query
                    ($qr_func(layout,
                            dim.0 as i32,
                            dim.1 as i32,
                            &mut slice,
                            lda as i32,
                            &mut tau),
                     tau)
                };

                if info == 0 {
                    QRFactors::from_raw(a, tau)
                } else if info < 0 {
                    Err(QRError::IllegalParameter(-info))
                } else {
                    unreachable!();
                }
            }

            fn compute_q<D1>(mat: &ArrayBase<D1, Ix2>,
                             tau: &[Self],
                             k: usize)
                             -> Result<Array<Self, Ix2>, QRError>
                where D1: Data<Elem = Self>
            {

                let (m, n) = mat.dim();
                if k > m {
                    return Err(QRError::InconsistentDimensions);
                }

                // Initialize q with the
                let mut q = mat.slice(s![.., ..k as isize]).to_owned();

                let info = {
                    let (slice, layout, ldq) = match slice_and_layout_mut(&mut q) {
                        None => unreachable!(),
                        Some(fwd) => fwd,
                    };

                    $qr_to_q(layout,
                           m as i32,
                           k as i32,
                           cmp::min(k, n) as i32,
                           slice,
                           ldq as i32,
                           tau)
                };
                if info == 0 {
                    Ok(q)
                } else {
                    Err(QRError::IllegalParameter(-info))
                }
            }

            fn compute_r<D1>(mat: &ArrayBase<D1, Ix2>, k: usize) -> Result<Array<Self, Ix2>, QRError>
                where D1: Data<Elem = Self>
            {
                let (m, n) = mat.dim();

                let nn = cmp::min(m, n);
                if k > nn {
                    return Err(QRError::InconsistentDimensions);
                }

                let mut r = Array::zeros((k as usize, n));

                // Copy the upper triangular/trapezoidal part of the matrix to
                // R.
                r.slice_mut(s![..k as isize, ..]).assign(&mat.slice(s![..k as isize, ..]));

                let zero = <$qr_type as Zero>::zero();

                // Replace zeros below the diagonal.
                for (i, mut row) in r.outer_iter_mut().enumerate().take(nn as usize) {
                    row.slice_mut(s![..i as isize]).fill(zero);
                }

                Ok(r)
            }
        }
    )
}

impl_qr!(f32, sgeqrf, sorgqr);
impl_qr!(f64, dgeqrf, dorgqr);
impl_qr!(c32, cgeqrf, cungqr);
impl_qr!(c64, zgeqrf, zungqr);
