//! Compute eigenvalues and eigenvectors of symmetric matrices.
//!
//! Symmetric (or Hermitian, for complex) matrices are guaranteed to
//! have real eigenvalues.

use lapack::c::{ssyev, dsyev, cheev, zheev};
use super::types::{Solution, EigenError};
use impl_prelude::*;

/// Scalar trait for computing eigenvalues of a symmetric matrix.
///
/// In order to extract eigenvalues or eigenvectors from a matrix,
/// that matrix with must have  entries implementing the `Eigen` trait.
pub trait SymEigen: LinxalScalar {
    /// Return the real eigenvalues of a symmetric matrix.
    ///
    /// If `with_vectors` is true, the right eigenvectors of 'V' are
    /// stored in the input matrix.
    fn compute_mut<D>(mat: &mut ArrayBase<D, Ix2>,
                      uplo: Symmetric,
                      with_vectors: bool)
                      -> Result<Array<Self::RealPart, Ix1>, EigenError>
        where D: DataMut<Elem = Self>;

    /// Return the real eigenvalues of a symmetric matrix.
    fn compute_into<D>(mut mat: ArrayBase<D, Ix2>,
                       uplo: Symmetric)
                       -> Result<Array<Self::RealPart, Ix1>, EigenError>
        where D: DataMut<Elem = Self> + DataOwned<Elem = Self> {
        Self::compute_mut(&mut mat, uplo, false)
    }

    /// Return the eigenvalues and, optionally, the eigenvectors of a
    /// symmetric matrix.
    ///
    /// # Remarks
    ///
    /// The input matrix is copied before the calculation takes place.
    fn compute<D>(mat: &ArrayBase<D, Ix2>,
                  uplo: Symmetric,
                  with_vectors: bool)
                  -> Result<Solution<Self, Self::RealPart>, EigenError>
        where D: Data<Elem = Self>;
}

macro_rules! impl_sym_eigen {
    ($impl_type:ident, $eigen_type:ident, $func:ident) => (
        impl SymEigen for $impl_type {

            fn compute_mut<D>(mat: &mut ArrayBase<D, Ix2>, uplo: Symmetric, with_vectors: bool) ->
                Result<Array<Self::RealPart, Ix1>, EigenError>
                where D: DataMut<Elem=Self>
            {
                let dim = mat.dim();
                if dim.0 != dim.1 {
                    return Err(EigenError::NotSquare);
                }

                let n = dim.0 as i32;

                let (mut data_slice, layout, ld) = match slice_and_layout_mut(mat) {
                    Some(x) => x,
                    None => return Err(EigenError::BadLayout)
                };

                let mut values = Array::default(n as Ix);
                let job = if with_vectors { b'V' } else { b'N' };

                let info = $func(layout, job, uplo as u8, n, data_slice,
                                 ld as i32, values.as_slice_mut().unwrap());

                if info  == 0 {
                    Ok(values)
                } else if info < 0 {
                    Err(EigenError::IllegalParameter(-info))
                } else {
                    Err(EigenError::Failed)
                }
            }

            fn compute<D>(mat: &ArrayBase<D, Ix2>,
                          uplo: Symmetric,
                          with_vectors: bool) -> Result<Solution<Self, Self::RealPart>, EigenError>
                where D: Data<Elem=Self> {
                let vec: Vec<Self> = mat.iter().cloned().collect();
                let mut new_mat = Array::from_shape_vec(mat.dim(), vec).unwrap();
                let r = Self::compute_mut(&mut new_mat, uplo, with_vectors);
                r.map(|values| {
                    Solution {
                        values: values,
                        left_vectors: None,
                        right_vectors: if with_vectors { Some(new_mat) } else { None }
                    }
                })
            }
        }
    )
}

impl_sym_eigen!(f32, f32, ssyev);
impl_sym_eigen!(c32, f32, cheev);
impl_sym_eigen!(f64, f64, dsyev);
impl_sym_eigen!(c64, f64, zheev);

#[cfg(test)]
mod tests {}
