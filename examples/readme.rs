#[macro_use]
extern crate linxal;
extern crate ndarray;

use linxal::types::{c32, LinxalMatrix};
use linxal::eigenvalues::Eigen;
use linxal::solve_linear::SolveLinear;
use ndarray::{arr1, arr2};

fn f1() {
    let m = arr2(&[[1.0f32, 2.0],
                   [-2.0, 1.0]]);

    let r = m.eigenvalues();
    assert!(r.is_ok());

    let r = r.unwrap();
    let true_evs = arr1(&[c32::new(1.0, 2.0), c32::new(1.0, -2.0)]);
    assert_eq_within_tol!(true_evs, r, 0.01);

    let b = arr1(&[-1.0, 1.0]);
    let x = m.solve_linear(&b).unwrap();
    let true_x = arr1(&[-0.6, -0.2]);
    assert_eq_within_tol!(x, true_x, 0.0001);
}


fn f2() {
    let m = arr2(&[[1.0f32, 2.0],
                   [-2.0, 1.0]]);

    let r = Eigen::compute(&m, false, false);
    assert!(r.is_ok());

    let r = r.unwrap();
    let true_evs = arr1(&[c32::new(1.0, 2.0), c32::new(1.0, -2.0)]);
    assert_eq_within_tol!(true_evs, r.values, 0.01);

    let b = arr1(&[-1.0, 1.0]);
    let x = SolveLinear::compute(&m, &b).unwrap();
    let true_x = arr1(&[-0.6, -0.2]);
    assert_eq_within_tol!(x, true_x, 0.0001);
}

fn main() {
    f1();
    f2();
}
