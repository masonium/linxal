use impl_prelude::*;
use std::cmp::Ordering;

/// Assert that two ndarrays are logically equivalent, within
/// tolerance.
///
/// Assert that two ndarrays are the same dimension, and that every element
/// of the first array is equal to the corresponding element of the
/// second array, within a given tolerance.
///
/// # Remarks
///
/// Requires the `linxal::LinxalImplScalar` trait to be imported.
/// Arrays with different storage layouts are otherwise considered
/// equal. Doesn't perform broadcasting.
#[macro_export]
macro_rules! assert_eq_within_tol {
    ($e1:expr, $e2:expr, $tol:expr) => (
        {
            use linxal::types::LinxalImplScalar;
            match (&$e1, &$e2, &$tol) {
                (x, y, tolerance) => {
                    assert_eq!(x.dim(), y.dim());
                    let t = *tolerance;
                    for (i, a) in x.indexed_iter() {
                        if (*a-y[i]).mag() > t {
                            panic!(format!("Elements at {:?} not within tolerance: |{} - {}| > {}",
                                           i, a, y[i], tolerance));
                        }
                    }
                }
            }
        }
    )
}

/// Return the conjugate transpose of a matrix.
pub fn conj_t<T: LinxalImplScalar, D: Data<Elem = T>>(a: &ArrayBase<D, Ix2>) -> Array<T, Ix2> {
    a.t().mapv(|x| x.cj())
}

/// Force a matrix to be triangular or trapezoidal, by zero-ing out
/// the other elements.
pub fn make_triangular_into<T, D>(mut a: ArrayBase<D, Ix2>, uplo: Symmetric) -> ArrayBase<D, Ix2>
    where T: LinxalImplScalar,
          D: DataMut<Elem = T> + DataOwned<Elem = T>
{
    let order = match uplo {
        Symmetric::Upper => Ordering::Greater,
        Symmetric::Lower => Ordering::Less,
    };
    for (i, x) in a.indexed_iter_mut() {
        if i.0.cmp(&i.1) == order {
            *x = T::zero();
        }
    }

    a
}

/// Force a matrix to be triangular or trapezoidal, by zero-ing out
/// the other elements.
pub fn make_triangular<T, D>(a: ArrayBase<D, Ix2>, uplo: Symmetric) -> Array<T, Ix2>
    where T: LinxalImplScalar,
          D: Data<Elem = T>
{
    make_triangular_into(a.to_owned(), uplo)
}
