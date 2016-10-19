//! Trait for comptuting eigenvalues
use ndarray::{ArrayBase, Array, DataMut, Data, Ix};
use lapack::{c32, c64};
use lapack::c::{sgeev, dgeev, Layout};
use super::types::{EigenError};

pub trait Eigen : Sized
{
    type Eigv;

    /// Return the eigenvvalues of a general matrix.
    fn values_mut<D>(mat: &mut ArrayBase<D, (Ix, Ix)>) -> Result<Array<Self::Eigv, Ix>, EigenError> where D: DataMut<Elem=Self>;

    //// Return the eigenvalues and, optionally, the eigenvectors of a general matrix.
    fn values_vectors_mut<D>(mat: &mut ArrayBase<D, (Ix, Ix)>, with_vectors: bool) ->
        Result<(Array<Self::Eigv, Ix>, Option<Array<Self, (Ix, Ix)>>), EigenError>
                where D:DataMut<Elem=Self>;

    /// Return the eigenvvalues of a general matrix.
    fn values<D>(mat: &ArrayBase<D, (Ix, Ix)>) -> Result<Array<Self::Eigv, Ix>, EigenError> where D: Data<Elem=Self>;
}

macro_rules! impl_eigen_real {
    ($impl_type:ident, $eigv_type:ident, $func:ident)  => (
        impl Eigen for $impl_type {

            type Eigv = $eigv_type;

            fn values_vectors_mut<D>(mat: &mut ArrayBase<D, (Ix, Ix)>, with_vectors: bool) ->
                Result<(Array<Self::Eigv, Ix>, Option<Array<Self, (Ix, Ix)>>), EigenError>
                where D:DataMut<Elem=Self> {

                let mut vl = [0.0 as Self];
                let mut vr = Array::default(if with_vectors { mat.dim() } else { (0, 0) });

                let layout = if mat.is_standard_layout() { Layout::RowMajor } else { Layout::ColumnMajor };
                let n = mat.dim().0 as i32;

                let data_slice = match mat.as_slice_memory_order_mut() {
                    Some(s) => s,
                    None => return Err(EigenError::BadInput)
                };

                let mut values_real_imag = vec![0.0; 2 * n as usize];
                let (mut values_real, mut values_imag) = values_real_imag.split_at_mut(n as usize);

                let vr_opt = (if with_vectors {'V'} else {'N'}) as u8;

                let info = $func(layout, 'N' as u8, vr_opt, n, data_slice,
                                 n, &mut values_real, &mut values_imag, &mut vl, n, vr.as_slice_mut().expect("just created."), n);

                if info  == 0 {
                    let vals: Vec<_> = values_real.iter().zip(values_imag.iter()).map(|(x, y)| Self::Eigv::new(*x, *y)).collect();
                    Ok((ArrayBase::from_vec(vals), if with_vectors { Some(vr) } else { None }))
                } else if info < 0 {
                    Err(EigenError::BadParameter(-info))
                } else {
                    Err(EigenError::Failed)
                }
            }

            fn values_mut<D>(mat: &mut ArrayBase<D, (Ix, Ix)>) -> Result<Array<Self::Eigv, Ix>, EigenError>
                where D: DataMut<Elem=Self> {

                match Self::values_vectors_mut(mat, false) {
                    Err(e) => Err(e),
                    Ok((eigv, _)) => Ok(eigv)
                }
            }


            fn values<D>(mat: &ArrayBase<D, (Ix, Ix)>) -> Result<Array<Self::Eigv, Ix>, EigenError> where D: Data<Elem=Self> {
                let vec: Vec<Self> = mat.iter().cloned().collect();
                let mut new_mat = Array::from_shape_vec(mat.dim(), vec).unwrap();
                Self::values_mut(&mut new_mat)
            }
        })

}

impl_eigen_real!(f32, c32, sgeev);
impl_eigen_real!(f64, c64, dgeev);

#[cfg(test)]
mod tests {
    use super::Eigen;
    use ndarray::{arr2};

    #[test]
    fn try_eig() {
        let mut m = arr2(&[[1.0 as f32, 2.0], [2.0, 1.0]]);

        let r = Eigen::values_vectors_mut(&mut m, true);
        assert!(r.is_ok());
    }

}
