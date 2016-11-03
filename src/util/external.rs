/// Assert that two ndarrays are logically equivalent, within
/// tolerance.
///
/// Assert that two ndarrays are the same dimension, and that every element
/// of the first array is equal to the corresponding element of the
/// second array, within a given tolerance.
///
/// # Remarks
///
/// Arrays with different storage layouts are otherwise considered
/// equal. Doesn't perform broadcasting.
#[macro_export]
macro_rules! assert_eq_within_tol {
    ($e1:expr, $e2:expr, $tol:expr) => (
        match (&$e1, &$e2, &$tol) {
            (x, y, tolerance) => {
                assert_eq!(x.dim(), y.dim());
                let t: f64 = *tolerance;
                for (i, a) in x.indexed_iter() {
                    if (*a-y[i]).mag() > t {
                        panic!(format!("Elements at {:?} not within tolerance: |{} - {}| > {}",
                                       i, a, y[i], tolerance));
                    }
                }
            }
        })
}
