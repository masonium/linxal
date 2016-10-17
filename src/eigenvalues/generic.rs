//! Trait for comptuting eigenvalues
use ndarray::{ArrayBase, Array, DataMut, Data, Ix};
use lapack::{c32};
use lapack::c::{sgeev, Layout};

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

pub trait Eigen : Sized
{
    type Eigv;

    /// Return the eigenvvalues of a general matrix.
    fn values_mut<D>(mat: &mut ArrayBase<D, (Ix, Ix)>) -> Result<Array<Self::Eigv, Ix>, EigenError> where D: DataMut<Elem=Self>;

    /// Return the eigenvvalues of a general matrix.
    fn values<D>(mat: &ArrayBase<D, (Ix, Ix)>) -> Result<Array<Self::Eigv, Ix>, EigenError> where D: Data<Elem=Self>;
}

impl Eigen for f32 {
    type Eigv = c32;

    fn values_mut<D>(mat: &mut ArrayBase<D, (Ix, Ix)>) -> Result<Array<Self::Eigv, Ix>, EigenError> where D:DataMut<Elem=Self> {
        let mut vl: Vec<Self> = Vec::new();
        let mut vr: Vec<Self> = Vec::new();

        let layout = if mat.is_standard_layout() { Layout::RowMajor } else { Layout::ColumnMajor };
        let n = mat.dim().0 as i32;

        let data_slice = match mat.as_slice_memory_order_mut() {
            Some(s) => s,
            None => return Err(EigenError::BadInput)
        };

        let mut values_real_imag = vec![0.0; 2 *n as usize];
        let (mut values_real, mut values_imag) = values_real_imag.split_at_mut(n as usize);

        let info = sgeev(layout, 'N' as u8, 'N' as u8, n, data_slice, n, &mut values_real, &mut values_imag, &mut vl, 1, &mut vr, 1);

        if info  == 0 {
            let vals: Vec<_> = values_real.iter().zip(values_imag.iter()).map(|(x, y)| c32::new(*x, *y)).collect();
            Ok(ArrayBase::from_vec(vals))
        } else if info < 0 {
            Err(EigenError::BadParameter(-info))
        } else {
            Err(EigenError::Failed)
        }
    }

    fn values<D>(mat: &ArrayBase<D, (Ix, Ix)>) -> Result<Array<Self::Eigv, Ix>, EigenError> where D: Data<Elem=Self> {
        let vec: Vec<Self> = mat.iter().cloned().collect();
        let mut new_mat = Array::from_shape_vec(mat.dim(), vec).unwrap();
        Self::values_mut(&mut new_mat)
    }
}

#[cfg(test)]
mod tests {
    use super::Eigen;
    use ndarray::{arr2};

    #[test]
    fn try_eig() {
        let mut m = arr2(&[[1.0 as f32, 2.0], [2.0, 1.0]]);

        let r = Eigen::values_mut(&mut m);
        assert!(r.is_ok());
    }

}
