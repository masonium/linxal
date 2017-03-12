#[macro_use]
extern crate linxal;

extern crate ndarray;
extern crate num_traits;

use ndarray::{Array, ArrayBase, Data, Ix2};
use linxal::factorization::{LUError, LUFactors};
use std::cmp;

/// Check that all the properties of the lu factorization are
/// reasonable.
fn check_lu<D1: Data<Elem=f32>>(
    m: &ArrayBase<D1, Ix2>, lu: &LUFactors<f32>, invertible: bool) {
    let l: Array<f32, Ix2> = lu.l();
    let u: Array<f32, Ix2> = lu.u();

    println!("{:?}", l);
    println!("{:?}", u);

    let k = cmp::min(m.rows(), m.cols());

    assert_eq!(l.rows(), lu.rows());
    assert_eq!(l.cols(), k);
    assert_eq!(u.rows(), k);
    assert_eq!(u.cols(), lu.cols());

    let a = lu.reconstruct();

    println!("{:?}", a);

    // P * L * U needs to match the original matrix.
    assert_eq_within_tol!(a, m, 0.001);

    let inverse = lu.inverse();

    if lu.rows() == lu.cols() {
        if invertible {
            assert!(inverse.is_ok());
        } else {
            assert!(inverse.is_err());
            assert_eq!(inverse.err().unwrap(), LUError::Singular);
        }
    } else {
        assert!(inverse.is_err());
        assert_eq!(inverse.err().unwrap(), LUError::NotSquare);
    }
}

#[test]
fn lu_diag() {
    let mut m = Array::zeros((4, 4));
    m.diag_mut().assign(&Array::linspace(1.0, 4.0, 4));

    let lu = LUFactors::compute(&m);
    assert!(lu.is_ok());

    check_lu(&m, &lu.unwrap(), true);
}

#[test]
fn lu_rectanglular() {
    let v: Vec<f32> = (0..12).map(|x| (x * x) as f32).collect();
    let m = Array::from_vec(v).into_shape((3, 4)).unwrap();

    let lu = LUFactors::compute(&m);
    assert!(lu.is_ok());

    check_lu(&m, &lu.unwrap(), true);
}

// TODO: netlib and opneblas seem to disagree on the correct behavior
// for row-rank matrices function.

// #[test]
// fn lu_rectanglular_singular() {
//     let m = Array::linspace(0.0, 11.0, 12).into_shape((4, 3)).unwrap();

//     let lu = LUFactors::compute(&m);
//     assert!(lu.is_err());
//     assert_eq!(lu.err().unwrap(), LUError::Singular);
// }


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

    check_lu(&m, &lu.unwrap(), true);
}
