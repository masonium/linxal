//! Trait for comptuting eigenvalues
use ndarray::{ArrayBase, Array, DataMut, Data, Ix};
use lapack::{c32};
use lapack::c::{ssyev, Layout};

pub enum EigenError {
    Success,
    BadInput,
    BadParameter(i32),
    Failed
}

pub enum ComputeVectors {
    Left,
    Right,
    Both
}

pub trait SymEigen : Sized
{
    /// Return the eigenvvalues of a general matrix.
    fn values_mut<D>(mat: &mut ArrayBase<D, (Ix, Ix)>) -> Result<Array<Self, Ix>, EigenError> where D: DataMut<Elem=Self>;

    /// Return the eigenvvalues of a general matrix.
    fn values<D>(mat: &ArrayBase<D, (Ix, Ix)>) -> Result<Array<Self, Ix>, EigenError> where D: Data<Elem=Self>;
}

impl SymEigen for f32 {
    fn values_mut<D>(mat: &mut ArrayBase<D, (Ix, Ix)>) -> Result<Array<Self, Ix>, EigenError> where D:DataMut<Elem=Self> {
        let empty = [0.0];

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
