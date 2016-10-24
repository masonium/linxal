#![macro_use]

use ndarray::{DataMut, ArrayBase, Ix, Ixs};
use lapack::c::{Layout};
use std::slice;

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
/// For LAPACKE methods, the memory layout does not need to be
/// contiguous. We only required that either rows or columns are
/// contiguous.
pub fn slice_and_layout_mut<D, S: DataMut<Elem=D>>(mat: &mut ArrayBase<S, (Ix, Ix)>) -> Option<(&mut [S::Elem], Layout, Ixs)> {
    let strides = {
        let s = mat.strides();
        (s[0], s[1])
    };

    if strides.0 < 0 || strides.1 < 0 {
        return None;
    }

    let dim = mat.dim();

    // One of the stides, must be 1
    if strides.1 == 1 {
        let m = strides.0;
        let s = unsafe {
            let nelem: usize = (dim.0 - 1) * m as usize + dim.1;
            slice::from_raw_parts_mut(mat.as_mut_ptr(), nelem)
        };
        Some((s, Layout::RowMajor, m))
    }
    else if strides.0 == 0 {
        let n = strides.1;
        let s = unsafe {
            let nelem: usize = (dim.1 - 1) * n as usize + dim.0;
            slice::from_raw_parts_mut(mat.as_mut_ptr(), nelem)
        };
        Some((s, Layout::ColumnMajor, n))
    }
    else {
        None
    }
}
