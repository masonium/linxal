//! Compute eigenvalues and eigenvectors of general matrices.
use ndarray::{ArrayBase, Array, DataMut, Data, Ix};
use lapack::{c32, c64};
use lapack::c::{sgeev, dgeev, cgeev, zgeev};
use matrix::slice_and_layout_mut;
use super::types::{EigenError, Solution};

pub trait Eigen : Sized + Clone
{
    type Eigv;
    type Solution;

    /// Return the eigenvalues and, optionally, the left and/or right eigenvectors of a general matrix.
    ///
    /// The entries in the input matrix `mat` are modified when calculating the eigenvalues.
    fn compute_mut<D>(mat: &mut ArrayBase<D, (Ix, Ix)>, compute_left: bool, compute_right: bool) ->
        Result<Self::Solution, EigenError> where D:DataMut<Elem=Self>;

    /// Return the eigenvvalues and, optionally, the eigenvectors of a general matrix.
    fn compute<D>(mat: &ArrayBase<D, (Ix, Ix)>, compute_left: bool, compute_right: bool) -> Result<Self::Solution, EigenError> where D: Data<Elem=Self> {
        let vec: Vec<Self> = mat.iter().cloned().collect();
        let mut new_mat = Array::from_shape_vec(mat.dim(), vec).unwrap();
        Self::compute_mut(&mut new_mat, compute_left, compute_right)
    }
}

/// Macro for implementing the Eigen trait on real-valued matrices.
macro_rules! impl_eigen_real {
    ($impl_type:ident, $eigv_type:ident, $func:ident)  => (
        impl Eigen for $impl_type {

            type Eigv = $eigv_type;
            type Solution = Solution<$impl_type, $eigv_type>;

            fn compute_mut<D>(mat: &mut ArrayBase<D, (Ix, Ix)>, compute_left: bool, compute_right: bool) ->
                Result<Self::Solution, EigenError>
                where D:DataMut<Elem=Self> {

                let n = mat.dim().0 as i32;


                let mut vl = Array::default(if compute_left { mat.dim() } else { (0, 0) });
                let mut vr = Array::default(if compute_right { mat.dim() } else { (0, 0) });

                let (data_slice, layout, ld) = match slice_and_layout_mut(mat) {
                    Some(s) => s,
                    None => return Err(EigenError::BadLayout)
                };

                let mut values_real_imag = vec![0.0; 2 * n as usize];
                let (mut values_real, mut values_imag) = values_real_imag.split_at_mut(n as usize);

                let vl_opt = (if compute_left {'V'} else {'N'}) as u8;
                let vr_opt = (if compute_right {'V'} else {'N'}) as u8;

                let info = $func(layout, vl_opt, vr_opt, n, data_slice,
                                 ld as i32, &mut values_real, &mut values_imag,
                                 vl.as_slice_mut().unwrap(), n,
                                 vr.as_slice_mut().unwrap(), n);

                if info  == 0 {
                    let vals: Vec<_> = values_real.iter().zip(values_imag.iter()).map(|(x, y)| Self::Eigv::new(*x, *y)).collect();
                    Ok(Solution {
                        values: ArrayBase::from_vec(vals),
                        left_vectors: if compute_left { Some(vl) } else { None },
                        right_vectors: if compute_right { Some(vr) } else { None }})
                } else if info < 0 {
                    Err(EigenError::BadParameter(-info))
                } else {
                    Err(EigenError::Failed)
                }
            }
        }
    )
}

impl_eigen_real!(f32, c32, sgeev);
impl_eigen_real!(f64, c64, dgeev);

macro_rules! impl_eigen_complex {
    ($impl_type:ident, $func:ident)  => (
        impl Eigen for $impl_type {

            type Eigv = $impl_type;
            type Solution = Solution<$impl_type, $impl_type>;

            fn compute_mut<D>(mat: &mut ArrayBase<D, (Ix, Ix)>, compute_left: bool, compute_right: bool) ->
                Result<Self::Solution, EigenError>
                where D:DataMut<Elem=Self> {

                let n = mat.dim().0 as i32;


                let mut vl = Array::default(if compute_left { mat.dim() } else { (0, 0) });
                let mut vr = Array::default(if compute_right { mat.dim() } else { (0, 0) });

                let (data_slice, layout, ld) = match slice_and_layout_mut(mat) {
                    Some(s) => s,
                    None => return Err(EigenError::BadLayout)
                };

                let mut values = Array::default(n as Ix);

                let vl_opt = (if compute_left {'V'} else {'N'}) as u8;
                let vr_opt = (if compute_right {'V'} else {'N'}) as u8;

                let info = $func(layout, vl_opt, vr_opt, n,
                                 data_slice, ld as i32,
                                 values.as_slice_mut().unwrap(),
                                 vl.as_slice_mut().unwrap(), n,
                                 vr.as_slice_mut().unwrap(), n);

                if info  == 0 {
                    Ok(Solution {
                        values: values,
                        left_vectors: if compute_left { Some(vl) } else { None },
                        right_vectors: if compute_right { Some(vr) } else { None }})
                } else if info < 0 {
                    Err(EigenError::BadParameter(-info))
                } else {
                    Err(EigenError::Failed)
                }
            }
        }
    )
}

impl_eigen_complex!(c32, cgeev);
impl_eigen_complex!(c64, zgeev);

#[cfg(test)]
mod tests {
    use super::Eigen;
    use ndarray::{arr2};

    #[test]
    fn try_eig() {
        let mut m = arr2(&[[1.0 as f32, 2.0], [2.0, 1.0]]);

        let r = Eigen::compute_mut(&mut m, false, true);
        assert!(r.is_ok());
    }

}
