use ndarray::{ArrayBase, Array, Data, Ix};

/// Return true if an array can be used as a matrix input.
///
/// For now, we just require a standard layout. However, this is stricter than necessary.
pub fn is_useable_input<S: Data, D>(mat: &ArrayBase<S, (Ix, Ix)>) -> bool {
    mat.ndim() == 2 && mat.is_standard_layout()
}
