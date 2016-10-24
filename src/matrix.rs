#![macro_use]

use ndarray::{DataMut, ArrayBase, Ix, Ixs};
use lapack::c::{Layout};

/// enum for symmetric matrix inputs
#[repr(u8)]
pub enum Symmetric {
    /// Read elements from the upper-triangular portion of the matrix
    Upper = b'U',

    /// Read elements from the lower-triangular portion of the matrix
    Lower =  b'L'
}

#[macro_export]
macro_rules! assert_in_tol {
    ($e1:expr, $e2:expr, $tol:expr) => (
        match (&$e1, &$e2, $tol) {
            (x, y, tol) => {
                assert_eq!(x.dim(), y.dim());
                for ((i, a), b) in x.indexed_iter().zip(y.iter()) {
                    if ((*a-*b) as f64).abs() > tol as f64 {
                        panic!(format!("Elements at {:?} not within tolerance: |{} - {}| > {}",
                                       i, a, b, tol));
                    }
                }
            }
        })
}

/// Return true if an array can be used as a matrix input.
///
/// For now, we just require a standard layout. However, this is stricter than necessary.
pub fn slice_and_layout_mut<D, S: DataMut<Elem=D>>(mat: &mut ArrayBase<S, (Ix, Ix)>) -> Option<(&mut [S::Elem], Layout, Ixs)> {
    let strides = {
        let s = mat.strides();
        (s[0], s[1])
    };

    match mat.as_slice_memory_order_mut() {
        Some(slice) => {
            // one of the stides, must be 1
            if strides.1 == 1 {
                let m = strides.0;
                Some((slice, Layout::RowMajor, m))
            }
            else if strides.0 == 0 {
                let n = strides.1;
                Some((slice, Layout::ColumnMajor, n))
            }
            else {
                None
            }
        },
        None => None
    }
}
