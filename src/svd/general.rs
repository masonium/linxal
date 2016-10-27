//! Solve singular value decomposition (SVD) of arbitrary matrices.

use ndarray::prelude::*;
use ndarray::Ix2;
use ndarray::DataMut;
use lapack::c::{sgesvd, sgesdd, dgesvd, dgesdd, cgesvd, cgesdd, zgesvd, zgesdd};
use lapack::{c32, c64};
use super::types::{SVDSolution, SVDError};
use matrix::{slice_and_layout_mut, matrix_with_layout};
use std::cmp;

const SVD_NORMAL_LIMIT: usize = 200;

pub trait SVD: Sized + Clone {
    type SingularValue;
    type Solution;

    /// Compute the singular value decomposition of a matrix.
    ///
    /// The elements of the input matrix `mat` are modified when this
    /// method is called. Use `Self::compute` when you don't wnat to
    /// modify the matrix.
    ///
    /// On success, returns a `Solution`, which always contains the
    /// singular values and optionally contains the left and right
    /// singular vectors. The left vectors (via the matrix `u`) are
    /// returned iff `compute_u` is true, and similarly for `vt` and
    /// `compute_vt`.
    fn compute_mut<D>(mat: &mut ArrayBase<D, Ix2>, compute_u: bool, compute_vt: bool) -> Result<Self::Solution, SVDError>
        where D: DataMut<Elem=Self>;

    /// Comptue the singular value decomposition of a matrix.
    ///
    /// Similar to `compute`, but the values are copied
    /// beforehand. leaving the original matrix un-modified.
    fn compute<D>(mat: &ArrayBase<D, Ix2>, compute_u: bool, compute_vt: bool) -> Result<Self::Solution, SVDError>
        where D: DataMut<Elem=Self> {
        let vec: Vec<Self> = mat.iter().cloned().collect();
        let mut m = Array::from_shape_vec(mat.dim(), vec).unwrap();
        Self::compute_mut(&mut m, compute_u, compute_vt)
    }
}


#[derive(Debug, PartialEq)]
enum SVDMethod {
    Normal,
    DivideAndConquer
}


/// Choose a method based on the problem.
fn select_svd_method(d: &Ix2, compute_either: bool) -> SVDMethod {
    let mx = cmp::max(d.0, d.1);

    // When we're computing one of them singular vector sets, we have
    // to compute both with divide and conquer. So, we're bound by the
    // maximum size of the array.
    if compute_either {
        if mx > SVD_NORMAL_LIMIT {
            SVDMethod::Normal
        } else {
            SVDMethod::DivideAndConquer
        }
    } else {
        SVDMethod::DivideAndConquer
    }
}


macro_rules! impl_svd {
    ($impl_type:ident, $sv_type:ident, $svd_func:ident, $sdd_func:ident) => (
        impl SVD for $impl_type {
            type Solution = SVDSolution<$impl_type, $sv_type>;
            type SingularValue = $sv_type;

            fn compute_mut<D>(mat: &mut ArrayBase<D, Ix2>, mut compute_u: bool, mut compute_vt: bool) -> Result<Self::Solution, SVDError>
                where D: DataMut<Elem=Self> {

                let dim = mat.dim();
                let (m, n) = dim;
                let mut s = Array::default(cmp::min(m, n));

                let (slice, layout, lda) = match slice_and_layout_mut(mat) {
                    Some(x) => x,
                    None => return Err(SVDError::BadLayout)
                };

                let compute_either = compute_u || compute_vt;
                let method = select_svd_method(&dim, compute_either);
                if method == SVDMethod::DivideAndConquer {
                    compute_u = compute_either;
                    compute_vt = compute_either;
                }

                let mut u = matrix_with_layout(if compute_u { (m, m) } else { (0, 0) }, layout);
                let mut vt = matrix_with_layout(if compute_vt { (n, n) } else { (0, 0) }, layout);

                let job_u = if compute_u { b'A' } else { b'N' };
                let job_vt = if compute_vt { b'A' } else { b'N' };

                let info = match method {
                    SVDMethod::Normal => {
                        let mut superb = Array::default(cmp::min(m, n) - 2);

                        $svd_func(layout, job_u, job_vt, m as i32, n as i32, slice,
                                  lda as i32, s.as_slice_mut().expect("bad s implementation"),
                                  u.as_slice_mut().expect("bad u implementation"), m as i32,
                                  vt.as_slice_mut().expect("bad vt implementation"), n as i32,
                                  superb.as_slice_mut().expect("bad superb implementation"))
                    },
                    SVDMethod::DivideAndConquer => {
                        let job_z = if compute_u || compute_vt { b'A' } else { b'N' };
                        $sdd_func(layout, job_z, m as i32, n as i32, slice,
                                  lda as i32,
                                  s.as_slice_mut().expect("bad s implementation"),
                                  u.as_slice_mut().expect("bad u implementation"), m as i32,
                                  vt.as_slice_mut().expect("bad vt implementation"), n as i32)
                    }
                };

                match info {
                    0 => {
                        Ok(SVDSolution {
                            values: s,
                            left_vectors: if compute_u { Some(u) } else { None },
                            right_vectors: if compute_vt { Some(vt) } else { None }
                        })
                    },
                    x if x < 0 => {
                        Err(SVDError::IllegalParameter(-x - 1))
                    },
                    x if x > 0 => {
                        Err(SVDError::Unconverged)
                    },
                    _ => {
                        unreachable!();
                    }
                }
            }
        }
    )
}

impl_svd!(f32, f32, sgesvd, sgesdd);
impl_svd!(f64, f64, dgesvd, dgesdd);
impl_svd!(c32, f32, cgesvd, cgesdd);
impl_svd!(c64, f64, zgesvd, zgesdd);
