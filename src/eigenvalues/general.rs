//! Compute eigenvalues and eigenvectors of general, non-symmetric matrices.
use lapack::c::{sgeev, dgeev, cgeev, zgeev};
use impl_prelude::*;
use super::types::{EigenError, Solution};

/// Scalar trait for computing eigenvalues.
///
/// In order to extract eigenvalues or eigenvectors from a matrix,
/// that matrix with must have  entries implementing the `Eigen` trait.
pub trait Eigen: LinxalImplScalar {
    /// Return the eigenvalues and, optionally, the left and/or right
    /// eigenvectors of a general matrix.
    ///
    /// The entries in the input matrix `mat` are modified when
    /// calculating the eigenvalues.
    fn compute_into<D>(mat: ArrayBase<D, Ix2>,
                       compute_left: bool,
                       compute_right: bool)
                       -> Result<Solution<Self, Self::Complex>, EigenError>
        where D: DataOwned<Elem = Self> + DataMut<Elem = Self>;

    /// Return the eigenvvalues and, optionally, the eigenvectors of a general matrix.
    fn compute<D>(mat: &ArrayBase<D, Ix2>,
                  compute_left: bool,
                  compute_right: bool)
                  -> Result<Solution<Self, Self::Complex>, EigenError>
        where D: Data<Elem = Self>
    {
        let vec: Vec<Self> = mat.iter().cloned().collect();
        let new_mat = Array::from_shape_vec(mat.dim(), vec).unwrap();
        Self::compute_into(new_mat, compute_left, compute_right)
    }
}

/// Macro for implementing the Eigen trait on real-valued matrices.
macro_rules! impl_eigen_real {
    ($impl_type:ident, $eigv_type:ident, $func:ident)  => (
        impl Eigen for $impl_type {
            fn compute_into<D>(mut mat: ArrayBase<D, Ix2>,
                               compute_left: bool, compute_right: bool) ->
                Result<Solution<Self, Self::Complex>, EigenError>
                where D: DataMut<Elem=Self> + DataOwned<Elem=Self> {

                let dim = mat.dim();
                if dim.0 != dim.1 {
                    return Err(EigenError::NotSquare);
                }
                let n = mat.dim().0 as i32;

                let (data_slice, layout, ld) = match slice_and_layout_mut(&mut mat) {
                    Some(s) => s,
                    None => return Err(EigenError::BadLayout)
                };

                let mut vl = matrix_with_layout(if compute_left { dim } else { (0, 0) }, layout);
                let mut vr = matrix_with_layout(if compute_right { dim } else { (0, 0) }, layout);

                let mut values_real_imag = vec![0.0; 2 * n as usize];
                let (mut values_real, mut values_imag) = values_real_imag.split_at_mut(n as usize);

                let vl_opt = (if compute_left {'V'} else {'N'}) as u8;
                let vr_opt = (if compute_right {'V'} else {'N'}) as u8;

                let info = unsafe {
                    $func(layout, vl_opt, vr_opt, n, data_slice,
                          ld as i32, &mut values_real, &mut values_imag,
                          vl.as_slice_mut().unwrap(), n,
                          vr.as_slice_mut().unwrap(), n)
                };

                if info == 0 {
                    let vals: Vec<_> = values_real.iter().zip(values_imag.iter())
                        .map(|(x, y)| Self::Complex::new(*x, *y)).collect();
                    Ok(Solution {
                        values: ArrayBase::from_vec(vals),
                        left_vectors: if compute_left { Some(vl) } else { None },
                        right_vectors: if compute_right { Some(vr) } else { None }})
                } else if info < 0 {
                    Err(EigenError::IllegalParameter(-info))
                } else {
                    Err(EigenError::Failed)
                }
            }
        }
    )
}

impl_eigen_real!(f32, c32, sgeev);
impl_eigen_real!(f64, c64, dgeev);

/// Macro for implementing the Eigen trait on complex-valued matrices.
macro_rules! impl_eigen_complex {
    ($impl_type:ident, $func:ident)  => (
        impl Eigen for $impl_type {

            fn compute_into<D>(mut mat: ArrayBase<D, Ix2>,
                               compute_left: bool, compute_right: bool)
                               -> Result<Solution<Self, Self::Complex>, EigenError>
                where D: DataMut<Elem=Self> + DataOwned<Elem=Self> {

                let dim = mat.dim();
                if dim.0 != dim.1 {
                    return Err(EigenError::NotSquare);
                }

                let n = dim.0 as i32;

                let (data_slice, layout, ld) = match slice_and_layout_mut(&mut mat) {
                    Some(s) => s,
                    None => return Err(EigenError::BadLayout)
                };

                let mut vl = matrix_with_layout(if compute_left { dim } else { (0, 0) }, layout);
                let mut vr = matrix_with_layout(if compute_right { dim } else { (0, 0) }, layout);

                let mut values = Array::default(n as Ix);

                let vl_opt = (if compute_left {'V'} else {'N'}) as u8;
                let vr_opt = (if compute_right {'V'} else {'N'}) as u8;

                let info = unsafe {
                    $func(layout, vl_opt, vr_opt, n,
                          data_slice, ld as i32,
                          values.as_slice_mut().unwrap(),
                          vl.as_slice_mut().unwrap(), n,
                          vr.as_slice_mut().unwrap(), n)
                };

                if info  == 0 {
                    Ok(Solution {
                        values: values,
                        left_vectors: if compute_left { Some(vl) } else { None },
                        right_vectors: if compute_right { Some(vr) } else { None }})
                } else if info < 0 {
                    Err(EigenError::IllegalParameter(-info))
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
mod tests {}
