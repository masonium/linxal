//! # `poly_interp` example
//!
//! In this example, we generate a random `n` degree polynomial,
//! evaluate it on `n+1` points, and use those points to reconstruct
//! the polynomial by solving a linear equation.

extern crate linxal;
extern crate ndarray;
extern crate rand;

use rand::thread_rng;
use rand::distributions::{Range, IndependentSample};
use linxal::solve_linear::SolveLinear;
use ndarray::{Array, Ix1};

/// Evalutate a polynomial f with coefficients `coefs` at `x`.
///
/// Input coefficients are ordered for lowest order to highest order.
fn eval_poly(x: f32, coefs: &[f32]) -> f32 {
    // horner's rule
    coefs.iter().rev().fold(0.0, |acc, c| acc * x + *c)
}

/// Returns the row (1.0, x, x^2, ..., x^n)
fn vandermonde_row(x: f32, n: usize) -> Array<f32, Ix1> {
    let mut v: Vec<f32> = Vec::with_capacity(n + 1);
    let mut r = 1.0;
    v.push(1.0);
    for _ in 1..(n + 1) {
        r *= x;
        v.push(r);
    }
    Array::from_vec(v)
}

fn main() {
    let mut rng = thread_rng();
    let coef_gen = Range::new(-1.0, 1.0);

    let n = 6;
    let coefs = Array::from_iter((0..n+1).map(|_| coef_gen.ind_sample(&mut rng)));

    let samples = Array::linspace(-1.0, 1.0, n+1);

    // Create the van der monde matrix.
    let mut a = Array::default((n+1, n+1));
    for (x, mut row) in samples.iter().zip(a.outer_iter_mut()) {
        row.assign(&vandermonde_row(*x, n));
    }

    // Create the solution matrix.
    let mut b = Array::default(n+1);
    for (i, x) in samples.iter().enumerate() {
        b[i] = eval_poly(*x, coefs.as_slice().unwrap());
    }

    // Solve the linear equations defined by the matrices.
    let fitted_coefs = SolveLinear::compute(&a, &b);
    assert!(fitted_coefs.is_ok());

    println!("Fitted Coefficients:\n{:?}", fitted_coefs.unwrap());
    println!("");
    println!("Actual Coefficients:\n{:?}", coefs);
}
