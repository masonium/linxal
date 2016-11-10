#[macro_use]
extern crate linxal;

#[macro_use]
extern crate ndarray;
extern crate num_traits;

use ndarray::{Array, ArrayBase, arr1, Data, Ix2};
use linxal::factorization::{QRFactors};
use linxal::prelude::{Magnitude};

/// Check that all the properties of the qr factorization are
/// reasonable.
fn check_qr<D1: Data<Elem=f32>>(m: &ArrayBase<D1, Ix2>, qr: &QRFactors<f32>) {
    let q: Array<f32, Ix2> = qr.q();
    let r: Array<f32, Ix2> = qr.r();

    assert_eq!(qr.rows(), q.rows());
    assert_eq!(qr.cols(), r.cols());

    // The reconstruction should match the original.
    let a1 = qr.reconstruct();
    assert_eq_within_tol!(a1, m, 0.001);

    // let mut a1 = Array::zeros((q.dim().0, r.dim().1));
    // general_mat_mul(1.0, &q, &r, 0.0, &mut a1);

    // Q * R needs to match QR.
    let a2 = q.dot(&r);
    println!("{:?}\n{:?}\n", q, r);
    assert_eq_within_tol!(a2, m, 0.001);

    // partial results for Q yield the same value as Q.
    for i in 1 .. q.dim().1 + 1 {
        let qk = qr.qk(i).unwrap();
        assert_eq_within_tol!(qk, q.slice(s![.., ..i as isize]), 0.001);
    }

    // partial results for R yield the same value as R.
    for i in 1 .. r.dim().0 + 1 {
        let rk = qr.rk(i).unwrap();
        assert_eq_within_tol!(rk, r.slice(s![..i as isize, ..]), 0.001);
    }

    // R is upper-triangular/trapezoidal
    for (i, x) in r.indexed_iter() {
        if i.0 > i.1 {
            assert_eq!(*x, 0.0);
        }
    }

    // Q is orthogonal
    let qtq = q.t().dot(&q);
    let eye: Array<f32, Ix2> = Array::eye(q.dim().1);
    assert_eq_within_tol!(qtq, eye, 0.001);
}

#[test]
fn qr_basic_tall() {
    let m = arr1(&[0.0, 2.0, 2.0, -1.0, 2.0, -1.0, 0.0, 1.5, 2.0, -1.0, 2.0, -1.0]).into_shape((6, 2)).unwrap();

    let qr = QRFactors::compute(&m);
    assert!(qr.is_ok());

    check_qr(&m, &qr.unwrap());
}


#[test]
fn qr_basic_wide() {
    let m = arr1(&[0.0, 2.0, 2.0, -1.0, 2.0, -1.0,
                   0.0, 1.5, 2.0, -1.0, 2.0, -1.0]).into_shape((2, 6)).unwrap();

    let qr = QRFactors::compute(&m);
    assert!(qr.is_ok());

    check_qr(&m, &qr.unwrap());
}


#[test]
fn qr_basic_linspace() {
    for m in 1..6 {
        for n in 1..6 {
            let mat = Array::linspace(1.0, (m*n) as f32, m*n).into_shape((m, n)).unwrap();

            let qr = QRFactors::compute(&mat);
            assert!(qr.is_ok());

            check_qr(&mat, &qr.unwrap());
        }
    }

}
