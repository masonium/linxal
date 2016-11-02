//! # `poly_fit` example
//!
//! In this example, we attempt to fit the polynomial f(x) = (x-1) *
//! (x-2) * (x-3) using a set of noisy samples from the interval [0,
//! 4].
//!
//! Specifially, we take 41 equally spaced samples, [0.0, 0.1, 0.2,
//! ..., 3.9, 4.0], We evalutate them on the desired polynomial, and
//! add gaussian noise according to the distrbution N(0, 0.1), to get
//! the RHS matrix `b`.  For the coefficient matrix, we compute [1.0,
//! x, .., x^n] for each sample point `x`, up to some desired degree
//! `n`. As long as `n` >= 3, the resulting solution should be close
//! to the original polynomial.

extern crate linxal;
extern crate ndarray;
extern crate rand;

use linxal::prelude::*;
use ndarray::prelude::*;
use rand::{thread_rng};
use rand::distributions::{Normal, Sample};

/// Evalutate a polynomial f with coefficients `coefs` at `x`.
fn eval_poly(x: f32, coefs: &[f32]) -> f32 {
    // horner's rule
    coefs.iter().fold(0.0, |acc, c| acc * x + *c)
}

/// Returns the row (1.0, x, x^2, ..., x^n)
fn vandermonde_row(x: f32, n: usize) -> Array<f32, Ix> {
    let mut v: Vec<f32> = Vec::with_capacity(n+1);
    let mut r = 1.0;
    v.push(1.0);
    for _ in 1..(n+1) {
        r *= x;
        v.push(r);
    }
    Array::from_vec(v)
}

fn main() {
    let mut rng = thread_rng();
    let mut err = Normal::new(0.0, 0.1);

    // (x-1)*(x-2)*(x-3)
    let coefs = vec![1.0, 11.0, -6.0, -6.0];

    const N: usize = 41;

    let samples = Array::linspace(0.0, 4.0, N);

    const FITTED_POLY_DEGREE: usize = 3;

    // Create the van der monde matrix.
    let mut a = Array::default((N, FITTED_POLY_DEGREE + 1));
    for (x, mut row) in samples.iter().zip(a.outer_iter_mut()) {
        row.assign(&vandermonde_row(*x, FITTED_POLY_DEGREE));
    }

    // Create the 'noisy' solution matrix.
    let mut b = Array::default((N, 1));
    for (i, x) in samples.iter().enumerate() {
        b[(i, 0)] = eval_poly(*x, &coefs) + (err.sample(&mut rng) as f32);
    }

    // Use least squares to fit the matrix.
    let fitted_coefs = LeastSquares::compute_multi(&a, &b);
    assert!(fitted_coefs.is_ok());

    println!("{:?}", fitted_coefs.unwrap().solution);
}
