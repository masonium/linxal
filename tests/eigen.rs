#[macro_use]
extern crate linxal;
extern crate ndarray;

use linxal::prelude::*;
use ndarray::prelude::*;

#[test]
fn try_eig() {
    let mut m = arr2(&[[1.0 as f32, 2.0], [2.0, 1.0]]);

    let r = Eigen::compute_mut(&mut m, false, true);
    assert!(r.is_ok());
}

#[test]
fn try_eig_func() {
    let mut m = arr2(&[[1.0 as f32, 2.0], [-2.0, 1.0]]);

    let r = Eigen::compute_mut(&mut m, false, true);
    assert!(r.is_ok());

    let r = r.unwrap();
    let true_evs = Array::from_vec(vec![c32::new(1.0, 2.0), c32::new(1.0, -2.0)]);
    assert_eq_within_tol!(true_evs, r.values, 0.01);
}
