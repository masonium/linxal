use ndarray::prelude::*;
use ndarray::DataMut;
use lapack::c::{sgesvd, sgesdd, dgesvd, dgesdd, cgesvd, cgesdd, zgesvd, zgesdd};
use lapack::{c32, c64};
use super::types::{Solution, SVDError};
use super::super::matrix::slice_and_layout_mut;
use std::cmp;

pub enum SVDMethod {
    Normal,
    DivideAndConquer
}

pub trait SVD: Sized + Clone {
    type SingularValue;

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
    ///
    /// # Examples
    /// ```rust
    /// use rula::SVD;
    /// use rula::nd::{arr1, arr2};
    /// let mut m = arr2(&[[8.79, 9.93, 9.83, 5.45, 3.16],
    /// [6.11, 6.91, 5.04, -0.27, 7.98],
    /// [-9.15, -7.93, 4.86, 4.85, 3.01],
    /// [9.57, 1.64, 8.83, 0.74, 5.80],
    /// [-3.49, 4.02, 9.80, 10.00, 4.27],
    /// [ 9.84, 0.15, -8.99, -6.02, -5.31]]);
    ///
    /// let r = SVD::compute_mut(&mut m, false, false, None);
    /// assert!(r.is_ok());
    ///
    /// let vals = r.ok().unwrap().values;
    fn compute_mut<D>(mat: &mut ArrayBase<D, (Ix, Ix)>, compute_u: bool, compute_vt: bool, method: Option<SVDMethod>) -> Result<Solution<Self, Self::SingularValue>, SVDError>
        where D: DataMut<Elem=Self>;

    /// Comptue the singular value decomposition of a matrix.
    ///
    /// Similar to `compute`, but the values are copied
    /// beforehand. leaving the original matrix un-modified.
    fn compute<D>(mat: &ArrayBase<D, (Ix, Ix)>, compute_u: bool, compute_vt: bool, method: Option<SVDMethod>) -> Result<Solution<Self, Self::SingularValue>, SVDError>
        where D: DataMut<Elem=Self> {
        let vec: Vec<Self> = mat.iter().cloned().collect();
        let mut m = Array::from_shape_vec(mat.dim(), vec).unwrap();
        Self::compute_mut(&mut m, compute_u, compute_vt, method)
    }
}

macro_rules! impl_svd {
    ($impl_type:ident, $sv_type:ident, $svd_func:ident, $sdd_func:ident) => (
        impl SVD for $impl_type {
            type SingularValue = $sv_type;

            fn compute_mut<D>(mat: &mut ArrayBase<D, (Ix, Ix)>, compute_u: bool, compute_vt: bool, method: Option<SVDMethod>) -> Result<Solution<$impl_type, $sv_type>, SVDError>
                where D: DataMut<Elem=Self> {

                let (m, n) = mat.dim();
                let mut u = Array::default(if compute_u { (m, m) } else { (0, 0) });
                let mut vt = Array::default(if compute_vt { (n, n) } else { (0, 0) });
                let mut s = Array::default(cmp::min(m, n));
                let mut superb = Array::default(cmp::min(m, n) - 2);

                let (slice, layout, lda) = match slice_and_layout_mut(mat) {
                    Some(x) => x,
                    None => return Err(SVDError::BadLayout)
                };

                let job_u = if compute_u { b'A' } else { b'N' };
                let job_vt = if compute_vt { b'A' } else { b'N' };

                let info = match method.unwrap_or(SVDMethod::Normal) {
                    SVDMethod::Normal => {
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
                        Ok(Solution {
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
