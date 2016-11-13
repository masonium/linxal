//! Compute the LU-factorization of a matrix with factors.
//!
//! A rectangular matrix A is factored into the product `P * `L` * `U`,
//! such that:
//!
//! - `P` is a permutation matrix
//!
//! - `L` is lower triangular (if m <= n) or lower trapezoidal (if m
//! >= n), with unit diagonal.
//!
//! - `U` is upper triangular (if m >= n) or upper trapezodial (if m <
//! n)
//!
//! Currently, the LU-factorization will fail if the matrix is not of
//! full row rank. In some sense, this is acceptable because most
//! operation with LU factorizations are non-sensible with a singular
//! matrix.
//!
//! The LU factorization can be used to solve Ax=b equations or
//! compute the inverse of `A`.

use impl_prelude::*;
use permute::{MatrixPermutation, Permutes};
use ndarray as nd;
use lapack::c::{sgetrf, dgetrf, cgetrf, zgetrf, sgetri, dgetri, cgetri, zgetri};

/// Error for LU-based computations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LUError {
    /// The layout of the matrix is not compatible
    BadLayout,

    /// The matrix is not square, but the operation requires a square
    /// matrix.
    NotSquare,

    /// The matrix is not invertible.
    Singular,

    /// The dimensions of the raw qr and tau don't match
    InconsistentDimensions,

    /// Implementation error, please submit as bug.
    IllegalParameter(i32),
}

/// Representation of the components Q, R of the factorization of
/// matrix A.
#[derive(Debug)]
pub struct LUFactors<T: LU> {
    mat: Array<T, Ix2>,

    /// permutation in native fortran format
    perm: MatrixPermutation,
}

impl<T: LU> LUFactors<T> {
    /// Create a `LUFactors` object from the output of the LAPACKE
    /// functions.
    fn from_raw<Matrix>(mat: Matrix, perm: Vec<i32>) -> Result<LUFactors<T>, LUError>
        where Matrix: Into<Array<T, Ix2>>
    {
        let mut ma = mat.into();

        if slice_and_layout_mut(&mut ma).is_none() {
            return Err(LUError::BadLayout);
        }

        Ok(LUFactors {
            mat: ma,
            perm: MatrixPermutation::from_ipiv(perm),
        })
    }

    /// Given a matrix, compute the LU decomposition of the matrix,
    /// returned as a `LUFactors` object.
    pub fn compute<D1: Data<Elem=T>>(mat: &ArrayBase<D1, Ix2>)
                                     -> Result<LUFactors<T>, LUError> {
        LU::compute(mat)
    }

    /// Returns the inverse of the original matrix, assuming it was
    /// square.
    pub fn inverse(&self) -> Result<Array<T, Ix2>, LUError> {
        LU::compute_inverse(&self.mat, self.perm.ipiv())
    }

    /// Returns the inverse of the original matrix, assuming it was
    /// square.
    pub fn inverse_into(self) -> Result<Array<T, Ix2>, LUError> {
        LU::compute_inverse_into(self.mat, self.perm.ipiv())
    }

    /// Return the number of rows in the original matrix
    pub fn rows(&self) -> usize {
        self.mat.rows()
    }

    /// Return the number of columns in the original matrix
    pub fn cols(&self) -> usize {
        self.mat.cols()
    }

    /// Return a reference to the permutation `P` of the
    /// factorization.
    pub fn perm(&self) -> &MatrixPermutation {
        &self.perm
    }

    fn k(&self) -> usize {
        cmp::min(self.rows(), self.cols())
    }

    /// Return the `m` by `min(m, n)` matrix `L`.
    ///
    /// `L` will be a lower triangular/trapezoidal matrix, with unit diagonal.
    pub fn l(&self) -> Array<T, Ix2> {
        let (m, _) = self.mat.dim();
        let k = self.k();

        let mut l = Array::zeros((m, k));

        // Copy the top-left 'm' x 'k' submatrix.
        let top_left_slice = s![.., ..k as isize];
        l.slice_mut(top_left_slice).assign(&self.mat.slice(top_left_slice));

        let zero = T::zero();
        let one = T::one();

        // Fix the diagonal and upper-triangle.
        for (r, mut row) in l.outer_iter_mut().enumerate().take(k) {
            row[r] = one;
            row.slice_mut(s![r as isize + 1..]).assign_scalar(&zero);
        }

        l
    }

    /// Return the first `min(m, n)` by `n` matrix U of the LU
    /// factorization.
    ///
    /// `U` will be upper triangular/trapezoidal.
    #[inline]
    pub fn u(&self) -> Array<T, Ix2> {
        let (_, n) = self.mat.dim();
        let k = self.k();

        let mut u = Array::zeros((k, n));

        // Copy the top-left 'k' x 'n' submatrix.
        let top_left_slice = s![..k as isize, ..];
        u.slice_mut(top_left_slice).assign(&self.mat.slice(top_left_slice));

        let zero = T::zero();

        // Fix the lower-triangle.
        for (r, mut row) in u.outer_iter_mut().enumerate().take(k) {
            row.slice_mut(s![..r as isize]).assign_scalar(&zero);
        }

        u
    }

    /// Reconstruct the original matrix `A` from the factorization
    pub fn reconstruct(&self) -> Array<T, Ix2> {
        let lu = self.l().dot(&self.u());
        self.perm.permute_into(lu).expect("guarantee that lu is the right size")
    }
}

/// Trait defined on scalars to support LU-factorization.
///
/// Any matrix composed of `LU` scalars can be split into `LUFactors`.
pub trait LU: nd::LinalgScalar + Permutes {
    /// Return a `LUFactors` structure, containing the LU
    /// factorization of the input matrix `A`.
    ///
    /// Similar to `compute`, but consumes the input.
    fn compute_into(a: Array<Self, Ix2>) -> Result<LUFactors<Self>, LUError>;

    /// Return a `LUFactors` structure, containing the LU
    /// factorization of the input matrix `A`.
    fn compute<D1: Data>(a: &ArrayBase<D1, Ix2>) -> Result<LUFactors<Self>, LUError>
        where D1: Data<Elem = Self>
    {
        Self::compute_into(a.to_owned())
    }

    /// Return the inverse of a matrix from its LU factorization,
    /// consuming the matrix in the process.
    fn compute_inverse_into<D1>(mat: ArrayBase<D1, Ix2>, perm: &[i32])
                                -> Result<ArrayBase<D1, Ix2>, LUError>
        where D1: DataOwned<Elem = Self> + DataMut<Elem=Self>;

    /// Return the inverse of matrix, copying it first.
    fn compute_inverse<D1>(mat: &ArrayBase<D1, Ix2>, perm: &[i32])
                           -> Result<Array<Self, Ix2>, LUError>
        where D1: Data<Elem = Self> {
        // Check squareness first, to save a copy that we might not need.
        if mat.rows() != mat.cols() {
            return Err(LUError::NotSquare);
        }

        let copy_mat = mat.to_owned();
        Self::compute_inverse_into(copy_mat, perm)
    }
}

macro_rules! impl_lu {
    ($lu_type:ty, $lu_func:ident, $lu_invert:ident) => (
        impl LU for $lu_type {
            fn compute_into(mut a: Array<Self, Ix2>) -> Result<LUFactors<Self>, LUError> {
                let dim = a.dim();

                let (info, perm_i) = {
                    let (mut slice, layout, lda) = match slice_and_layout_mut(&mut a) {
                        None => return Err(LUError::BadLayout),
                        Some(x) => x,
                    };

                    let mut perm_i = Vec::new();
                    perm_i.resize(cmp::min(dim.0, dim.1), -1);

                    println!("{:?}", lda);
                    // workspace query
                    ($lu_func(layout,
                              dim.0 as i32,
                              dim.1 as i32,
                              &mut slice,
                              lda as i32,
                              &mut perm_i),
                     perm_i)
                };

                println!("{:?}\n", perm_i);

                if info == 0 {
                    LUFactors::from_raw(a, perm_i)
                } else if info < 0 {
                    Err(LUError::IllegalParameter(-info))
                } else {
                    unreachable!();
                }
            }

            fn compute_inverse_into<D1>(mut mat: ArrayBase<D1, Ix2>, perm: &[i32])
                                        -> Result<ArrayBase<D1, Ix2>, LUError>
                where D1: DataOwned<Elem = Self> + DataMut<Elem=Self> {
                let dim = mat.dim();
                if dim.0 != dim.1 {
                    return Err(LUError::NotSquare);
                }

                let info = {
                    let (mut slice, layout, lda) = match slice_and_layout_mut(&mut mat) {
                        None => return Err(LUError::BadLayout),
                        Some(x) => x,
                    };

                    $lu_invert(layout, dim.0 as i32, &mut slice, lda as i32, &perm)
                };
                if info == 0 {
                    Ok(mat)
                } else if info < 0 {
                    Err(LUError::IllegalParameter(-info))
                } else {
                    Err(LUError::NotInvertible)
                }
            }
        }
    )
}

impl_lu!(f32, sgetrf, sgetri);
impl_lu!(f64, dgetrf, dgetri);
impl_lu!(c32, cgetrf, cgetri);
impl_lu!(c64, zgetrf, zgetri);
