use impl_prelude::*;
use lapack::c::{sgeqrf, sorgqr};
use std::fmt::Debug;
use types::Magnitude;

/// Error for QR-based computations.
#[derive(Debug, Clone)]
pub enum QRError {
    /// The layout of the matrix is not compatible
    BadLayout,

    /// The dimensions of the raw qr and tau don't match
    InconsistentDimensions,

    /// Implementation error, please submit as bug.
    IllegalParameter(i32),
}

/// Enum for muliplication by Q.
#[repr(u8)]
#[derive(Debug, Clone)]
pub enum QRMultiply {
    /// Compute Q * A for the input matrix A
    Left = b'L',

    /// Compute A * Q for the input matrix A
    Right = b'R',
}

/// Representation of the components Q, R of the factorization of
/// matrix A.
#[derive(Debug)]
pub struct QRFactors<T: QR> {
    mat: Array<T, Ix2>,
    tau: Vec<T>,

    /// Number of rows in the original matrix
    pub m: usize,

    /// Number of columns in the original matrix
    pub n: usize,
}

impl<T: QR> QRFactors<T> {
    /// Create a `QRFactors` object from the output of the LAPACKE
    /// functions.
    fn from_raw<Matrix>(mat: Matrix, tau: Vec<T>) -> Result<QRFactors<T>, QRError>
        where Matrix: Into<Array<T, Ix2>>
    {
        let mut ma = mat.into();
        let (m, n) = ma.dim();

        if slice_and_layout_mut(&mut ma).is_none() {
            return Err(QRError::BadLayout);
        }

        Ok(QRFactors {
            mat: ma,
            tau: tau,
            m: m,
            n: n,
        })
    }

    pub fn compute<D1: Data<Elem=T>>(mat: &ArrayBase<D1, Ix2>)
                                     -> Result<QRFactors<T>, QRError> {
        QR::compute(mat)
    }


    // /// Return `A * Q`, for the `Q` stored internally.
    // pub fn left_multiply_by_q<D1: Data<Elem=T>>(&self, a: &ArrayBase<D1, Ix2>)
    //                                             -> Result<Array<T, Ix2>, QRError> {
    //     QR::multiply_by_q(&self.mat, &self.tau, a, QRMultiply::Left)
    // }

    // /// Return `A * Q`, for the `Q` stored internally.
    // pub fn right_multiply_by_q<D1: Data<Elem=T>>(&self, a: &ArrayBase<D1, Ix2>)
    //                                              -> Result<Array<T, Ix2>, QRError> {
    //     QR::multiply_by_q(&self.mat, &self.tau, a, QRMultiply::Right)
    // }

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
        QR::compute_q(&self.mat, &self.tau, kr.unwrap_or(cmp::min(self.m, self.n)))
    }

    /// Return the `m` by `min(m, n)` matrix `Q`.
    ///
    /// Equivalent to `self.qk(None)`.
    ///
    /// `self.q()` is large enough so that `&self.q() * &self.r()`
    /// will faithfully reproduce the original matrix `A`.
    #[inline]
    pub fn q(&self) -> Result<Array<T, Ix2>, QRError> {
        self.qk(None)
    }


    /// Return the first 'k' rows of the matrix R of the QR
    /// factorization.
    ///
    /// When `k` is None, compute enough rows (`min(m, n)`) to
    /// faithfully recreate the original matrix A.
    pub fn rk<K: Into<Option<usize>>>(&self, k: K) -> Result<Array<T, Ix2>, QRError> {
        let kr = k.into();
        let p = kr.unwrap_or(cmp::min(self.m, self.n));
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
    pub fn r(&self) -> Result<Array<T, Ix2>, QRError> {
        self.rk(None)
    }
}

/// Trait defined on scalars to support QR-factorization.
pub trait QR: Sized + Clone + Magnitude + Debug {
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

    /// multiply an input matrix by `Q`.
    ///
    /// Not intended to be used by end-users.  `side` indicates
    /// whether to pre-or-post-multiply the input by `Q`.
    // fn multiply_by_q<D1, D2>(mat: &ArrayBase<D1, Ix2>,
    //                          tau: &[Self],
    //                          a: &ArrayBase<D2, Ix2>,
    //                          mult: QRMultiply)
    //                   -> Result<Array<Self, Ix2>, QRError>
    //     where D1: Data<Elem = Self>,
    //           D2: Data<Elem = Self>;

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

impl QR for f32 {
    fn compute_into(mut a: Array<Self, Ix2>) -> Result<QRFactors<Self>, QRError> {
        let dim = a.dim();

        let (info, tau) = {
            let (mut slice, layout, lda) = match slice_and_layout_mut(&mut a) {
                None => return Err(QRError::BadLayout),
                Some(x) => x,
            };

            let mut tau = Vec::new();
            tau.resize(cmp::min(dim.0, dim.1), 0.0);

            // workspace query
            (sgeqrf(layout,
                    dim.0 as i32,
                    dim.1 as i32,
                    &mut slice,
                    lda as i32,
                    &mut tau),
             tau)
        };

        println!("{:?}\n {:?}\n", a, tau);

        if info == 0 {
            QRFactors::from_raw(a, tau)
        } else if info < 0 {
            Err(QRError::IllegalParameter(-info))
        } else {
            unreachable!();
        }
    }

    // fn multiply_by_q<D1, D2>(a: &ArrayBase<D1, Ix2>,
    //                          tau: &[Self],
    //                          c: &ArrayBase<D2, Ix2>,
    //                          mult: QRMultiply)
    //                   -> Result<Array<Self, Ix2>, QRError>
    //     where D1: Data<Elem = Self>,
    //           D2: Data<Elem = Self>
    // {
    //     let (m, n) = a.dim();
    //     let (cm, cn) = c.dim();

    //     let result_dim = match mult {
    //         QRMultiply::Left => cm == n
    //         QRMultiply::Right => {
    //             if cn == m {
    //                 (cm, n)

    //     };

    //     if !matching_dim {
    //         return Err(QRError::InconsistentDimensions);
    //     }

    //     let (slice, layout, lda) = match slice_and_layout(&a) {
    //         None => return Err(QRError::BadLayout),
    //         Some(fwd) => fwd,
    //     };

    //     let mut c_result = matrix_with_layout(c.dim(), layout);

    //     let info = {
    //         let (c_slice, ldc) = match slice_and_layout_mut(&mut c_result) {
    //             None => unreachable!(),
    //             Some((slice, _, ldc)) => (slice, ldc),
    //         };

    //         sormqr(layout,
    //                mult as u8,
    //                b'N',
    //                cm as i32,
    //                cn as i32,
    //                tau.len() as i32,
    //                slice,
    //                lda as i32,
    //                tau,
    //                c_slice,
    //                ldc as i32)
    //     };

    //     if info == 0 {
    //         Ok(c_result)
    //     } else if info < 0 {
    //         Err(QRError::IllegalParameter(-info))
    //     } else {
    //         unreachable!();
    //     }
    // }

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

            sorgqr(layout,
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

        // Replace zeros below the diagonal.
        for (i, mut row) in r.outer_iter_mut().enumerate().take(nn as usize) {
            row.slice_mut(s![..i as isize]).assign_scalar(&0.0);
        }

        Ok(r)
    }
}
