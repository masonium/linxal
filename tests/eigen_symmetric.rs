#[macro_use]
extern crate linxal;
extern crate ndarray;

use ndarray::{arr1, arr2, Array, Ix2};
use linxal::types::{LinxalMatrix, Symmetric};

#[test]
fn try_eig() {
    let m: Array<_, Ix2> = arr2(&[[1.0f32, 2.0], [2.0, 1.0]]);

    let r = m.symmetric_eigenvalues(Symmetric::Upper);
    assert!(r.is_ok());

    let true_values = arr1(&[-1.0f32, 3.0]);

    assert_eq_within_tol!(r.unwrap(), true_values, 0.01);
}

#[test]
fn test_eig_access() {
    let upper_only = arr2(&[[1.0f32, 2.0], [-3.0, 1.0]]);
    let full = arr2(&[[1.0f32, 2.0], [2.0, 1.0]]);

    let upper_only_ev = upper_only.symmetric_eigenvalues(Symmetric::Upper).unwrap();
    let full_ev = full.symmetric_eigenvalues(Symmetric::Upper).unwrap();

    assert_eq_within_tol!(upper_only_ev, full_ev, 1e-5);
}
