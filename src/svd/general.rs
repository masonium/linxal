//! Solve singular value decomposition (SVD) of arbitrary matrices.

use lapack::c::{sgesvd, sgesdd, dgesvd, dgesdd, cgesvd, cgesdd, zgesvd, zgesdd};
use super::types::{SVDSolution, SVDError};
use impl_prelude::*;

const SVD_NORMAL_LIMIT: usize = 200;

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum SVDComputeVectors {
    Full,
    Economic,
    None,
}

impl SVDComputeVectors {
    /// Return the size of the `U` matrix generated from this type of
    /// computation, given the size of the original matrix.
    fn u_size(&self, m: usize, n: usize) -> (usize, usize) {
        match *self {
            SVDComputeVectors::Full => (m, m),
            SVDComputeVectors::Economic => {(m, cmp::min(m, n))},
            SVDComputeVectors::None => (0, 0)
        }
    }

    /// Return the size of the `V^H` matrix generated from this type of
    /// computation, given the size of the original matrix.
    fn vt_size(&self, m: usize, n: usize) -> (usize, usize) {
        match *self {
            SVDComputeVectors::Full => (n, n),
            SVDComputeVectors::Economic => {(cmp::min(m, n), n)},
            SVDComputeVectors::None => (0, 0)
        }
    }

    /// Return the job descriptor implementaiton for this type.
    fn job_desc(&self) -> u8 {
        match *self {
            SVDComputeVectors::Full => b'A',
            SVDComputeVectors::Economic => b'S',
            SVDComputeVectors::None => b'N'
        }
    }
}

/// Trait for scalars that can implement SVD.
pub trait SVD: LinxalImplScalar {
    /// Compute the singular value decomposition of a matrix.
    ///
    /// Use `Self::compute` when you don't wnat to consume the input
    /// matrix.
    ///
    /// On success, returns an `SVDSolution`, which always contains the
    /// singular values and optionally contains the left and right
    /// singular vectors. The left vectors (via the matrix `u`) are
    /// returned iff `compute_u` is true, and similarly for `vt` and
    /// `compute_vt`.
    fn compute_into<D>(mat: ArrayBase<D, Ix2>,
                       compute_vectors: SVDComputeVectors)
                       -> Result<SVDSolution<Self>, SVDError>
        where D: DataMut<Elem = Self> + DataOwned<Elem = Self>;

    /// Comptue the singular value decomposition of a matrix.
    ///
    /// Similar to [`SVD::compute_into`](#tymethod.compute_into), but
    /// the values are copied beforehand. leaving the original matrix
    /// un-modified.
    fn compute<D>(mat: &ArrayBase<D, Ix2>,
                  compute_vectors: SVDComputeVectors)
                  -> Result<SVDSolution<Self>, SVDError>
        where D: Data<Elem = Self>
    {
        let vec: Vec<Self> = mat.iter().cloned().collect();
        let m = Array::from_shape_vec(mat.dim(), vec).unwrap();
        Self::compute_into(m, compute_vectors)
    }
}


#[derive(Debug, PartialEq)]
enum SVDMethod {
    Normal,
    DivideAndConquer,
}


/// Choose a method based on the problem.
fn select_svd_method(d: &Ix2, compute_vectors: SVDComputeVectors) -> SVDMethod {
    let mx = cmp::max(d[0], d[1]);

    // When we're computing one of them singular vector sets, we have
    // to compute both with divide and conquer. So, we're bound by the
    // maximum size of the array.
    match compute_vectors {
        SVDComputeVectors::None => SVDMethod::DivideAndConquer,
        _ => {
            if mx > SVD_NORMAL_LIMIT {
                SVDMethod::Normal
            } else {
                SVDMethod::DivideAndConquer
            }
        }
    }
}


macro_rules! impl_svd {
    ($impl_type:ident, $svd_func:ident, $sdd_func:ident) => (
        impl SVD for $impl_type {
            fn compute_into<D>(mut mat: ArrayBase<D, Ix2>,
                               compute_vectors: SVDComputeVectors)
                               -> Result<SVDSolution<$impl_type>, SVDError>
                where D: DataMut<Elem=Self> + DataOwned<Elem = Self>{

                let (m, n) = mat.dim();
                let raw_dim = mat.raw_dim();
                let mut s = Array::default(cmp::min(m, n));

                let (slice, layout, lda) = match slice_and_layout_mut(&mut mat) {
                    Some(x) => x,
                    None => return Err(SVDError::BadLayout)
                };

                let method = select_svd_method(&raw_dim, compute_vectors);

                let mut u = matrix_with_layout(compute_vectors.u_size(m, n), layout);
                let mut vt = matrix_with_layout(compute_vectors.vt_size(m, n), layout);

                let job_desc = compute_vectors.job_desc();

                let info = match method {
                    SVDMethod::Normal => {
                        let mut superb = Array::default(cmp::min(m, n) - 2);
                        $svd_func(layout, job_desc, job_desc, m as i32, n as i32, slice,
                                  lda as i32, s.as_slice_mut().expect("bad s implementation"),
                                  u.as_slice_mut().expect("bad u implementation"), m as i32,
                                  vt.as_slice_mut().expect("bad vt implementation"), n as i32,
                                  superb.as_slice_mut().expect("bad superb implementation"))
                    },
                    SVDMethod::DivideAndConquer => {
                        $sdd_func(layout, job_desc, m as i32, n as i32, slice,
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
                            left_vectors: if compute_vectors == SVDComputeVectors::None { None } else { Some(u) },
                            right_vectors: if compute_vectors == SVDComputeVectors::None { None } else { Some(vt) },
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

impl_svd!(f32, sgesvd, sgesdd);
impl_svd!(f64, dgesvd, dgesdd);
impl_svd!(c32, cgesvd, cgesdd);
impl_svd!(c64, zgesvd, zgesdd);
