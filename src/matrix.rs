use ndarray::{DataMut, ArrayBase, Array, Data, Ix, Ixs};
use lapack::c::{Layout};

/// Return true if an array can be used as a matrix input.
///
/// For now, we just require a standard layout. However, this is stricter than necessary.
pub fn slice_and_layout_mut<D, S: DataMut<Elem=D>>(mat: &ArrayBase<S, (Ix, Ix)>) -> Option<(&mut [S::Elem], Layout, Ixs)> {
    let strides = mat.strides();

    match mat.as_slice_memory_order_mut() {
        Some(slice) => {
            // one of the stides, must be 1
            if strides[1] == 1 {
                let m = strides[0];
                Some((slice, Layout::RowMajor, m))
            }
            else if strides[0] == 0 {
                let n = strides[1];
                Some((slice, Layout::ColumnMajor, n))
            }
            else {
                None
            }
        },
        None => None
    }
}
