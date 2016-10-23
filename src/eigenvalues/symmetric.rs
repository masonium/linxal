//! Compute eigenvalues and eigenvectors of symmetric matrices.

use ndarray::{ArrayBase, Array, DataMut, Data, Ix};
use lapack::c::{ssyev, Layout};
use super::types::{EigenError};

pub trait SymEigen : Sized
{
    /// Return the real eigenvalues of a symmetric matrix.
    ///
    /// The input matrix is mutated as part of the computation. Use
    /// [`.values()`](#tymethod.values) if you need to preserve the original
    /// matrix.
    ///
    /// # Examples
    /// ```rust
    /// use rula::SymEigen;
    /// use rula::nd::{arr1, arr2};
    /// use rula::lp::{c32};
    /// let mut m = arr2(&[[1.0 as f32, 2.0], [2.0, 1.0]]);
    ///
    /// let r = SymEigen::values_mut(&mut m);
    /// assert!(r.is_ok());
    ///
    /// let vals = r.ok().unwrap();
    /// assert_eq!(vals, arr1(&[-1.0, 3.0]));
    /// ```
    fn values_mut<D>(mat: &mut ArrayBase<D, (Ix, Ix)>) -> Result<Array<Self, Ix>, EigenError> where D: DataMut<Elem=Self>;

    /// Return the eigenvalues of a symmetric matrix.
    ///
    /// # Remarks
    ///
    /// The input matrix is copied before the calculation takes place.
    fn values<D>(mat: &ArrayBase<D, (Ix, Ix)>) -> Result<Array<Self, Ix>, EigenError> where D: Data<Elem=Self>;
}

impl SymEigen for f32 {
    fn values_mut<D>(mat: &mut ArrayBase<D, (Ix, Ix)>) -> Result<Array<Self, Ix>, EigenError> where D:DataMut<Elem=Self> {
        let layout = if mat.is_standard_layout() { Layout::RowMajor } else { Layout::ColumnMajor };
        let n = mat.dim().0 as i32;

        let data_slice = match mat.as_slice_memory_order_mut() {
            Some(s) => s,
            None => return Err(EigenError::BadInput)
        };

        let mut values = vec![0.0; n as usize];

        let info = ssyev(layout, 'V' as u8, 'U' as u8, n, data_slice,
                         n, &mut values);

        if info  == 0 {
            Ok(ArrayBase::from_vec(values))
        } else if info < 0 {
            Err(EigenError::BadParameter(-info))
        } else {
            Err(EigenError::Failed)
        }
    }

    fn values<D>(mat: &ArrayBase<D, (Ix, Ix)>) -> Result<Array<Self, Ix>, EigenError> where D: Data<Elem=Self> {
        let vec: Vec<Self> = mat.iter().cloned().collect();
        let mut new_mat = Array::from_shape_vec(mat.dim(), vec).unwrap();
        Self::values_mut(&mut new_mat)
    }
}

#[cfg(test)]
mod tests {
    use super::SymEigen;
    use ndarray::{arr2};

    #[test]
    fn try_eig() {
        let mut m = arr2(&[[1.0 as f32, 2.0], [2.0, 1.0]]);

        let r = SymEigen::values_mut(&mut m);
        assert!(r.is_ok());
    }

}
