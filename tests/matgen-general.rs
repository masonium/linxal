extern crate linxal;

extern crate itertools;

extern crate rand;
extern crate ndarray;

use ndarray::prelude::*;
use linxal::generate::{MG, RandomGeneral, RandomUnitary};
use linxal::types::{c32, c64};
use linxal::properties::{is_unitary, is_diagonal};
use rand::thread_rng;

fn test_gen_unitary<T: MG> () {
    for t in 1..21 {
        // generate a presumably unitary matrix.
        let m: Array<T, Ix2> = RandomUnitary::new(t, &mut thread_rng()).generate().ok().unwrap();

        // Ensure that it is actually unitary.
        assert!(is_unitary(&m));
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
fn test_gen_diagonal<T: MG> () {
    for r in 1..15 {
        for c in 1..15 {
        let m: Array<T, Ix2> = RandomGeneral::new(r, c, &mut thread_rng()).diagonal().generate().ok().unwrap();
            assert!(is_diagonal(&m));
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
