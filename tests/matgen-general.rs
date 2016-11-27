#[macro_use]
extern crate linxal;

#[macro_use]
extern crate itertools;

extern crate rand;
extern crate ndarray;

use ndarray::prelude::*;
use linxal::generate::{MG, RandomGeneral, RandomUnitary};
use linxal::types::{c32, c64};
use linxal::properties::{is_unitary, is_diagonal,
                         get_lower_bandwidth_tol, get_upper_bandwidth_tol};
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

////////////////////////////////////////////////////////////////////////////////

/// Ensures that lower and upper bandwidth call is honored
fn test_gen_bands<T: MG> () {
    for (r, c) in iproduct!(1..10, 1..10) {
        for (l, u) in iproduct!(0..r, 0..c) {
            let m: Array<T, Ix2> = RandomGeneral::new(r, c, &mut thread_rng()).bands(l, u).generate().ok().unwrap();
            println!("{:?}", m);
            assert!(get_lower_bandwidth_tol(&m, 0.0) <= l);
            assert!(get_upper_bandwidth_tol(&m, 0.0) <= u);
        }
    }
}

#[test]
fn test_gen_bands_f32() {
    test_gen_bands::<f32>();
}
#[test]
fn test_gen_bands_f64() {
    test_gen_bands::<f64>();
}
#[test]
fn test_gen_bands_c32() {
    test_gen_bands::<c32>();
}
#[test]
fn test_gen_bands_c64() {
    test_gen_bands::<c64>();
}
