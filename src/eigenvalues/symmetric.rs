//! Compute eigenvalues and eigenvectors of symmetric matrices.

use ndarray::{ArrayBase, Array, DataMut, Data, Ix};
use lapack::c::{ssyev};
use super::types::{Solution, EigenError};
use matrix::*;

pub trait SymEigen : Sized
{
    type Solution;

    /// Return the real eigenvalues of a symmetric matrix.
    ///
    /// The input matrix is mutated as part of the computation. Use
    /// [`.values()`](#tymethod.values) if you need to preserve the original
    /// matrix.
    fn compute_mut<D>(mat: &mut ArrayBase<D, (Ix, Ix)>,
                      uplo: Symmetric,
                      with_vectors: bool) -> Result<Array<Self, Ix>, EigenError> where D: DataMut<Elem=Self>;

    /// Return the eigenvalues of a symmetric matrix.
    ///
    /// # Remarks
    ///
    /// The input matrix is copied before the calculation takes place.
    fn compute<D>(mat: &ArrayBase<D, (Ix, Ix)>, uplo: Symmetric, with_vectors: bool) -> Result<Self::Solution, EigenError> where D: Data<Elem=Self>;
}

impl SymEigen for f32 {
    type Solution = Solution<Self, Self>;

    fn compute_mut<D>(mat: &mut ArrayBase<D, (Ix, Ix)>, uplo: Symmetric, with_vectors: bool) ->
        Result<Array<Self, Ix>, EigenError> where D:DataMut<Elem=Self>
    {
        let n = mat.dim().0 as i32;

        let (mut data_slice, layout, ld) = match slice_and_layout_mut(mat) {
            Some(x) => x,
            None => return Err(EigenError::BadLayout)
        };

        let mut values = vec![0.0; n as usize];

        let info = ssyev(layout, if with_vectors { b'V' } else { b'N' }, uplo as u8, n, data_slice,
                         ld as i32, &mut values);

        if info  == 0 {
            Ok(ArrayBase::from_vec(values))
        } else if info < 0 {
            Err(EigenError::BadParameter(-info))
        } else {
            Err(EigenError::Failed)
        }
    }


    fn compute<D>(mat: &ArrayBase<D, (Ix, Ix)>, uplo: Symmetric, with_vectors: bool) -> Result<Self::Solution, EigenError> where D: Data<Elem=Self> {
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

#[cfg(test)]
mod tests {
    use super::SymEigen;
    use ndarray::prelude::*;
    use matrix::Symmetric;

    #[test]
    fn try_eig() {
        let mut m = arr2(&[[1.0 as f32, 2.0], [2.0, 1.0]]);

        let r = SymEigen::compute_mut(&mut m, Symmetric::Upper, false);
        assert!(r.is_ok());

        assert_in_tol!(r.unwrap(), arr1(&[-1.0, 3.0]), 0.01);
    }

}
