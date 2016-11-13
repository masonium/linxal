#[macro_use]
extern crate linxal;

#[macro_use]
extern crate ndarray;
extern crate num_traits;

use ndarray::{Array, ArrayBase, Data, Ix2};
use linxal::factorization::{LU, LUFactors};
use std::cmp;
use std::fmt::Display;

/// Check that all the properties of the lu factorization are
/// reasonable.
fn check_lu<T: LU + Display, D1: Data<Elem=T>>(m: &ArrayBase<D1, Ix2>, lu: &LUFactors<T>) {
    let l: Array<T, Ix2> = lu.l();
    let u: Array<T, Ix2> = lu.u();

    let k = cmp::min(m.rows(), m.cols());

    assert_eq!(l.rows(), lu.rows());
    assert_eq!(l.cols(), k);
    assert_eq!(u.rows(), k);
    assert_eq!(u.cols(), lu.cols());

    let a = lu.reconstruct();

    // P * L * U needs to match the original matrix.
    assert_eq_within_tol!(a, m, 0.001);

}

#[test]
fn lu_diag() {
    let mut m = Array::zeros((4, 4));
    m.diag_mut().assign(&Array::linspace(1.0, 4.0, 4));

    let lu = LUFactors::compute(&m);
    assert!(lu.is_ok());

    check_lu(&m, &lu.unwrap());
}

#[test]
fn lu_perm_diag() {
    let mut m = Array::zeros((4, 4));
    m[(0, 0)] = 1.0;
    m[(1, 2)] = 2.0;
    m[(2, 1)] = 3.0;
    m[(3, 3)] = 4.0;

    let lu = LUFactors::compute(&m);
    println!("{:?}", lu);
    assert!(lu.is_ok());

    check_lu(&m, &lu.unwrap());
}
