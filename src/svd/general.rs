use ndarray::{ArrayBase, Array, DataMut, Data, Ix};
use lapack::c::{sgesvd, dgesvd, cgesvd, zgesvd};
use super::types::{Solution, SVDError};
use super::super::matrix::slice_and_layout_mut;
use std::cmp;

pub trait SVD: Sized {
    type SingularValue;

    fn compute<D>(mat: &mut ArrayBase<D, (Ix, Ix)>, compute_u: bool, compute_vt: bool) -> Result<Solution<Self>, SVDError<Self>>
        where D: DataMut<Elem=Self>;
}


impl SVD for f32 {
    type SingularValue = f32;

    fn compute<D>(mat: &mut ArrayBase<D, (Ix, Ix)>, compute_u: bool, compute_vt: bool) -> Result<Solution<Self>, SVDError<Self>>
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

        let info = sgesvd(layout, job_u, job_vt, m as i32, n as i32, slice,
                          lda as i32, s.as_slice_mut().expect("bad s implementation"),
                          u.as_slice_mut().expect("bad u implementation"), m as i32,
                          vt.as_slice_mut().expect("bad vt implementation"), n as i32,
                          superb.as_slice_mut().expect("bad superb implementation"));

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
                Err(SVDError::Unconverged((x, superb)))
            },
            _ => {
                unreachable!();
            }
        }
    }
}
