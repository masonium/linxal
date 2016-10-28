#![macro_use]

use ndarray::prelude::*;
use ndarray::{Ix2, DataMut};
use lapack::c::{Layout};
use std::slice;

#[macro_export]
macro_rules! assert_in_tol {
    ($e1:expr, $e2:expr, $tol:expr) => (
        match (&$e1, &$e2, &$tol) {
            (x, y, tolerance) => {
                assert_eq!(x.dim(), y.dim());
                let t: f64 = tolerance.to_f64().unwrap();
                for (i, a) in x.indexed_iter() {
                    if (*a-y[i]).abs().to_f64().unwrap() > t {
                        panic!(format!("Elements at {:?} not within tolerance: |{} - {}| > {}",
                                       i, a, y[i], tolerance));
                    }
                }
            }
        })
}

/// Return an array with the specified dimensions and layout.
///
/// This function is used internally to ensure that
pub fn matrix_with_layout<T: Default>(d: Ix2, layout: Layout) -> Array<T, Ix2> {
    Array::default(match layout {
        Layout::RowMajor => d.into(),
        Layout::ColumnMajor => d.f()
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
