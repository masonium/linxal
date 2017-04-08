#[macro_use]
extern crate linxal;

extern crate itertools;
extern crate num_traits;
extern crate rand;
extern crate ndarray;

use ndarray::prelude::*;
use linxal::generate::{RandomSymmetric};
use linxal::types::{c32, c64, Symmetric, LinxalMatrix, LinxalScalar};
use rand::thread_rng;

fn test_gen_symmetric<T: LinxalScalar> () {
    for t in 1..21 {
        // generate a presumably symmetric matrix.
        let m: Array<T, Ix2> = RandomSymmetric::new(t, &mut thread_rng()).generate().ok().unwrap();

        // Ensure that it is actually symmetric.
        assert!(m.is_symmetric(None));
    }
}

#[test]
fn test_gen_symmetric_f32() {
    test_gen_symmetric::<f32>();
}
#[test]
fn test_gen_symmetric_c32() {
    test_gen_symmetric::<c32>();
}
#[test]
fn test_gen_symmetric_f64() {
    test_gen_symmetric::<f64>();
}
#[test]
fn test_gen_symmetric_c64() {
    test_gen_symmetric::<c64>();
}

fn symmetric_with_gen_eigenvalues<T: LinxalScalar>() {
    for t in 1..21 {
        let pair: (Array<T, Ix2>, Array<T::RealPart, Ix1>) = RandomSymmetric::new(t, &mut thread_rng()).ev_random_uniform(-5.0, 5.0)
            .generate_with_ev().ok().unwrap();
        
        // The generated matrix should be symmetric.
        assert!(pair.0.is_symmetric(None));

        // The calculated eigenvalues of the matrix should match the ones returned by generate*.
        let mut sorted_evs = pair.1.clone();
        sorted_evs.as_slice_mut().unwrap().sort_by(|x, y| x.partial_cmp(y).unwrap());
        
        let ev_solution = pair.0.symmetric_eigenvalues(Symmetric::Upper).unwrap();
        let mut solution_evs = ev_solution.clone();
        solution_evs.as_slice_mut().unwrap().sort_by(|x, y| x.partial_cmp(y).unwrap());

        // Make sure the compute eigenvalues are equivalent to the 
        assert_eq_within_tol!(sorted_evs, solution_evs, 1e-4.into());
    }
}

#[test]
fn symmetric_with_gen_eigenvalues_f32() {
    symmetric_with_gen_eigenvalues::<f32>();
    
}
#[test]
fn symmetric_with_gen_eigenvalues_f64() {
    symmetric_with_gen_eigenvalues::<f64>();
    
}
#[test]
fn symmetric_with_gen_eigenvalues_c32() {
    symmetric_with_gen_eigenvalues::<c32>();
    
}
#[test]
fn symmetric_with_gen_eigenvalues_c64() {
    symmetric_with_gen_eigenvalues::<c64>();
    
}

