//! Compute eigenvalues and eigenvectors of symmetric matrices.

use ndarray::{ArrayBase, Array, DataMut, Data, Ix, Ix2};
use lapack::c::{ssyev, dsyev, cheev, zheev};
use lapack::{c32, c64};
use super::types::{Solution, EigenError};
use types::Symmetric;
use util::*;

pub trait SymEigen : Sized
{
    type SingularValue;
    type Solution;

    /// Return the real eigenvalues of a symmetric matrix.
    ///
    /// The input matrix is mutated as part of the computation. Use
    /// [`.values()`](#tymethod.values) if you need to preserve the original
    /// matrix.
    fn compute_mut<D>(mat: &mut ArrayBase<D, Ix2>, uplo: Symmetric, with_vectors: bool) ->
        Result<Array<Self::SingularValue, Ix>, EigenError> where D: DataMut<Elem=Self>;

    /// Return the eigenvalues of a symmetric matrix.
    ///
    /// # Remarks
    ///
    /// The input matrix is copied before the calculation takes place.
    fn compute<D>(mat: &ArrayBase<D, Ix2>, uplo: Symmetric, with_vectors: bool) ->
        Result<Self::Solution, EigenError> where D: Data<Elem=Self>;
}

macro_rules! impl_sym_eigen {
    ($impl_type:ident, $eigen_type:ident, $func:ident) => (
        impl SymEigen for $impl_type {
            type SingularValue = $eigen_type;
            type Solution = Solution<Self, Self::SingularValue>;

            fn compute_mut<D>(mat: &mut ArrayBase<D, Ix2>, uplo: Symmetric, with_vectors: bool) ->
                Result<Array<Self::SingularValue, Ix>, EigenError> where D: DataMut<Elem=Self>
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
                    Err(EigenError::BadParameter(-info))
                } else {
                    Err(EigenError::Failed)
                }
            }


            fn compute<D>(mat: &ArrayBase<D, Ix2>, uplo: Symmetric, with_vectors: bool) -> Result<Self::Solution, EigenError> where D: Data<Elem=Self> {
                let vec: Vec<Self> = mat.iter().cloned().collect();
                let mut new_mat = Array::from_shape_vec(mat.dim(), vec).unwrap();
                Self::compute_mut(&mut new_mat, uplo, with_vectors).map(|values| {
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
mod tests {
    use super::SymEigen;
    use super::super::super::util::Magnitude;
    use ndarray::prelude::*;
    use types::Symmetric;
    use num_traits::ToPrimitive;

    #[test]
    fn try_eig() {
        let mut m = arr2(&[[1.0 as f32, 2.0], [2.0, 1.0]]);

        let r = SymEigen::compute_mut(&mut m, Symmetric::Upper, false);
        assert!(r.is_ok());

        assert_in_tol!(r.unwrap(), arr1(&[-1.0, 3.0]), 0.01);
    }

}
