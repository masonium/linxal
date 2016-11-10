use impl_prelude::*;
use lapack::c::{slaswp, dlaswp, zlaswp, claswp};

/// A compressed representation of matrix permutations.
///
/// The internal representation of a matrix permutation mimics the
/// output of various LAPACK functions.
#[derive(Debug, Clone)]
pub struct MatrixPermutation {
    ipiv: Vec<i32>
}

impl MatrixPermutation {
    /// Return permutation from ipivot result from a LAPACK[E] call.
    pub fn from_ipiv(ipiv: Vec<i32>) -> MatrixPermutation {
        MatrixPermutation { ipiv: ipiv  }
    }

    /// Permute an input matrix `mat` by this permutation.
    pub fn permute_into<T, D1>(&self, mat: ArrayBase<D1, Ix2>) -> Option<ArrayBase<D1, Ix2>>
        where T: Permutable,
              D1: DataOwned<Elem=T> + DataMut<Elem=T> {
        Permutable::permute_into(mat, &self.ipiv)
    }

    /// Permute a matrix `mat` by this permutation.
    pub fn permute<T, D1>(&self, mat: &ArrayBase<D1, Ix2>) -> Option<Array<T, Ix2>>
        where T: Permutable, D1: Data<Elem=T> {
        Permutable::permute(mat, &self.ipiv)
    }

    /// Return the native LAPACKE representation of the permutation.
    pub fn ipiv(&self) -> &[i32] {
        &self.ipiv
    }
}

/// A `Permutable` is a scalar that one can apply a
/// `MatrixPermutation` to.
///
/// The methods of permuatble are not meant to be used by end-users.
pub trait Permutable: Sized + Clone {
    /// Permute a matrix by the pivot representation of a permutation.
    fn permute_into<D1>(mat: ArrayBase<D1, Ix2>, ipiv: &[i32]) -> Option<ArrayBase<D1, Ix2>>
        where D1: DataOwned<Elem=Self> + DataMut<Elem=Self>;

    /// Permute a matrix by the pivot representation of a permutation.
    fn permute<D1>(mat: &ArrayBase<D1, Ix2>, ipiv: &[i32]) -> Option<Array<Self, Ix2>>
        where D1: Data<Elem=Self> {
        Self::permute_into(mat.to_owned(), ipiv)
    }
}

macro_rules! impl_perm {
    ($perm_type:ty, $perm_func:ident) => (
        impl Permutable for $perm_type {
            fn permute_into<D1>(mut mat: ArrayBase<D1, Ix2>, ipiv: &[i32]) -> Option<ArrayBase<D1, Ix2>>
                where D1: DataOwned<Elem=Self> + DataMut<Elem=Self> {

                let (m, n) = mat.dim();

                if ipiv.len() > m {
                    return None
                }


                let info = {
                    let (mut slice, layout, lda) = match slice_and_layout_mut(&mut mat) {
                        None => return None,
                        Some(fwd) => fwd
                    };

                    $perm_func(layout, n as i32, slice, lda as i32, 1, ipiv.len() as i32, ipiv, 1)
                };

                if info == 0 {
                    Some(mat)
                } else {
                    None
                }
            }
        }
    )
}

impl_perm!(f32, slaswp);
impl_perm!(f64, dlaswp);
impl_perm!(c32, claswp);
impl_perm!(c64, zlaswp);
