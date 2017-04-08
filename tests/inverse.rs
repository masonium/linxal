#[macro_use]
extern crate linxal;

extern crate ndarray;
extern crate num_traits;
extern crate rand;

use rand::thread_rng;
use ndarray::{Array, Ix2};
use linxal::types::{c32, c64, LinxalMatrix, LinxalScalar};
use linxal::generate::{RandomUnitary};

fn invert_unitary<F: LinxalScalar>() {
    let mut gen = thread_rng();
    let mut rg = RandomUnitary::new(10, &mut gen);

    for _ in 1..10 {
        let m: Array<F, Ix2> = rg.generate().unwrap();
        let m_ct = m.conj_t();

        let inv: Array<F, Ix2> = m.inverse().unwrap();

        assert_eq_within_tol!(&inv, &m_ct, 1e-3.into());
    }
}

#[test]
fn invert_unitary_f32() {
    invert_unitary::<f32>();
}
#[test]
fn invert_unitary_f64() {
    invert_unitary::<f64>();
}
#[test]
fn invert_unitary_c32() {
    invert_unitary::<c32>();
}
#[test]
fn invert_unitary_c64() {
    invert_unitary::<c64>();
}
