extern crate linxal;

extern crate itertools;

extern crate rand;
extern crate ndarray;

use ndarray::prelude::*;
use linxal::generate::{RandomGeneral, RandomUnitary};
use linxal::types::{LinxalMatrix, LinxalScalar, c32, c64};
use rand::thread_rng;

fn test_gen_unitary<T: LinxalScalar> () {
    for t in 1..21 {
        // generate a presumably unitary matrix.
        let m: Array<T, Ix2> = RandomUnitary::new(t, &mut thread_rng()).generate().ok().unwrap();

        // Ensure that it is actually unitary.
        assert!(m.is_unitary(None));
    }
}

#[test]
fn test_gen_unitary_f32() {
    test_gen_unitary::<f32>();
}
#[test]
fn test_gen_unitary_c32() {
    test_gen_unitary::<c32>();
}
#[test]
fn test_gen_unitary_f64() {
    test_gen_unitary::<f64>();
}
#[test]
fn test_gen_unitary_c64() {
    test_gen_unitary::<c64>();
}


////////////////////////////////////////////////////////////////////////////////

/// Create random diagonal matrices
fn test_gen_diagonal<T: LinxalScalar> () {
    for r in 1..15 {
        for c in 1..15 {
        let m: Array<T, Ix2> = RandomGeneral::new(r, c, &mut thread_rng()).diagonal().generate().ok().unwrap();
            assert!(m.is_diagonal(None));
        }
    }
}

#[test]
fn test_gen_diagonal_f32() {
    test_gen_diagonal::<f32>();
}
#[test]
fn test_gen_diagonal_c32() {
    test_gen_diagonal::<c32>();
}
#[test]
fn test_gen_diagonal_f64() {
    test_gen_diagonal::<f64>();
}
#[test]
fn test_gen_diagonal_c64() {
    test_gen_diagonal::<c64>();
}
