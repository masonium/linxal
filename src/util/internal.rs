use impl_prelude::LinxalScalar;
use ndarray::prelude::*;
use ndarray::{Data, DataMut};
use lapack::c::Layout;
use std::slice;

/// Return an array with the specified dimensions and layout.
///
/// This function is used internally to ensure that the matrix outputs
/// are created to be compatible with the matrix input.
pub fn matrix_with_layout<T: LinxalScalar, Sh>(d: Sh, layout: Layout)
                                          -> Array<T, Ix2>
    where Sh: ShapeBuilder<Dim=Ix2> {
    let shape = match layout {
            Layout::RowMajor => d.into_shape(),
            Layout::ColumnMajor => d.f(),
    };
    Array::default(shape)
}

/// Return the slice, layout, and leading dimension of the matrix.
///
/// For LAPACKE methods, the memory layout does not need to be
/// contiguous. We only required that either rows or columns are
/// contiguous.
pub fn slice_and_layout_mut<D, S: DataMut<Elem = D>>(mat: &mut ArrayBase<S, Ix2>)
                                                     -> Option<(&mut [S::Elem], Layout, Ixs)> {
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
    } else if strides.0 == 1 {
        let n = strides.1;
        let s = unsafe {
            let nelem: usize = (dim.1 - 1) * n as usize + dim.0;
            slice::from_raw_parts_mut(mat.as_mut_ptr(), nelem)
        };
        Some((s, Layout::ColumnMajor, n))
    } else {
        None
    }
}

/// Return the slice, layout, and leading dimension of the matrix.
///
/// For LAPACKE methods, the memory layout does not need to be
/// contiguous. We only required that either rows or columns are
/// contiguous.
pub fn slice_and_layout<D, S: Data<Elem = D>>(mat: &ArrayBase<S, Ix2>)
                                              -> Option<(&[S::Elem], Layout, Ixs)> {
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
            slice::from_raw_parts(mat.as_ptr(), nelem)
        };
        Some((s, Layout::RowMajor, m))
    } else if strides.0 == 1 {
        let n = strides.1;
        let s = unsafe {
            let nelem: usize = (dim.1 - 1) * n as usize + dim.0;
            slice::from_raw_parts(mat.as_ptr(), nelem)
        };
        Some((s, Layout::ColumnMajor, n))
    } else {
        None
    }
}


/// Return the slice, layout, and leading dimension of the matrix. If
/// the matrix can be interpreted with multiple inputs, assume the
/// previous one.
///
/// For LAPACKE methods, the memory layout does not need to be
/// contiguous. We only required that either rows or columns are
/// contiguous.
///
/// Returns None if the layout can't be matched.
pub fn slice_and_layout_matching_mut<D, S: DataMut<Elem = D>>(mat: &mut ArrayBase<S, Ix2>,
                                                              layout: Layout)
                                                              -> Option<(&mut [S::Elem], Ixs)> {
    let dim = mat.dim();

    // For column vectors, we can choose whatever layout we want.
    if dim.1 == 1 {
        let m = mat.strides()[0];

        let s = unsafe {
            let nelem: usize = (dim.0 - 1) * m as usize + dim.1;
            slice::from_raw_parts_mut(mat.as_mut_ptr(), nelem)
        };
        return Some((s, m));
    }

    // Otherwise, we just use the normal method and check for a match.
    if let Some((slice, lo, ld)) = slice_and_layout_mut(mat) {
        if lo == layout {
            return Some((slice, ld));
        }
    }

    None
}
