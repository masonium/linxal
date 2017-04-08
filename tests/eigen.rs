#[macro_use]
extern crate linxal;
extern crate ndarray;

use linxal::types::{LinxalMatrixInto, c32};
use linxal::types::error::{EigenError};
use ndarray::{arr1, arr2, Array};

#[test]
fn try_eig() {
    let m = arr2(&[[1.0f32, 2.0], [2.0, 1.0]]);

    let r = m.eigenvalues_into();
    assert!(r.is_ok());
}

#[test]
fn try_eig_func() {
    let m = arr2(&[[1.0f32, 2.0],
                   [-2.0, 1.0]]);

    let r = m.eigenvalues_vectors_into(false, true);
    assert!(r.is_ok());

    let r = r.unwrap();
    let true_evs = arr1(&[c32::new(1.0, 2.0), c32::new(1.0, -2.0)]);
    assert_eq_within_tol!(true_evs, r.values, 0.01);
}

#[test]
fn eig_nonsquare() {
    let m = Array::linspace(0.0, 5.0, 6).into_shape((3, 2)).unwrap();

    let r = m.eigenvalues_into();
    assert!(r.is_err());
    assert_eq!(r.err().unwrap(), EigenError::NotSquare);
}
