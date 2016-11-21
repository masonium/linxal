#[macro_use]
extern crate linxal;
extern crate ndarray;

use ndarray::{arr1, arr2};
use linxal::types::{Symmetric, LinxalScalar};
use linxal::eigenvalues::{SymEigen};

#[test]
fn try_eig() {
    let m = arr2(&[[1.0 as f32, 2.0], [2.0, 1.0]]);

    let r = SymEigen::compute_into(m, Symmetric::Upper);
    assert!(r.is_ok());

    assert_eq_within_tol!(r.unwrap(), arr1(&[-1.0, 3.0]), 0.01);
}

#[test]
fn test_eig_access() {
    let upper_only = arr2(&[[1.0f32, 2.0], [-3.0, 1.0]]);
    let full = arr2(&[[1.0f32, 2.0], [2.0, 1.0]]);

    let upper_only_ev = SymEigen::compute_into(upper_only, Symmetric::Upper).unwrap();
    let full_ev = SymEigen::compute_into(full, Symmetric::Upper).unwrap();

    assert_eq_within_tol!(upper_only_ev, full_ev, 1e-5);
}
