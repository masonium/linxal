#[macro_use]
extern crate linxal;
extern crate ndarray;

use linxal::eigenvalues::{Eigen};
use linxal::types::{c32, Magnitude};
use ndarray::{arr1, arr2};

#[test]
fn try_eig() {
    let m = arr2(&[[1.0f32, 2.0], [2.0, 1.0]]);

    let r = Eigen::compute_into(m, false, true);
    assert!(r.is_ok());
}

#[test]
fn try_eig_func() {
    let m = arr2(&[[1.0f32, 2.0],
                   [-2.0, 1.0]]);

    let r = Eigen::compute_into(m, false, true);
    assert!(r.is_ok());

    let r = r.unwrap();
    let true_evs = arr1(&[c32::new(1.0, 2.0), c32::new(1.0, -2.0)]);
    assert_eq_within_tol!(true_evs, r.values, 0.01);
}
