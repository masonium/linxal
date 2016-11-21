extern crate num_traits;
extern crate ndarray;
#[macro_use]
extern crate linxal;
extern crate lapack;

use ndarray::{Array, Ix1, Ix2, Axis};
use linxal::solve_linear::{SolveLinear};
use linxal::types::{LinxalScalar};

#[test]
pub fn solve_linear_vector() {
    let a: Array<f32, Ix2> = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b: Array<f32, Ix1> = Array::from_vec(vec![3.0, 7.0]);

    let x = SolveLinear::compute_into(a, b);

    assert!(x.is_ok());
    let values = x.unwrap();

    let truth = Array::from_vec(vec![1.0, 1.0]);
    assert_eq_within_tol!(&values, &truth, 1e-5);
}


#[test]
pub fn solve_linear_matrix() {
    let a: Array<f32, Ix2> = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b_vec = vec![3.0, -1.0, 2.0, 7.0, -1.0, 6.0];
    let b: Array<f32, Ix2> = Array::from_shape_vec((2, 3), b_vec).unwrap();

    let x = SolveLinear::compute_multi_into(a, b);

    assert!(x.is_ok());
    let values = x.unwrap();

    let truth_vec = vec![1.0, 1.0, 2.0, 1.0, -1.0, 0.0];

    let truth = Array::from_shape_vec((2, 3), truth_vec).unwrap();
    assert_eq_within_tol!(&values, &truth, 1e-5);
}


#[test]
pub fn solve_linear_vector_view() {
    let a: Array<f32, Ix2> = Array::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b_vec = vec![3.0, -1.0, 2.0, 7.0, -1.0, 6.0];
    let b: Array<f32, Ix2> = Array::from_shape_vec((2, 3), b_vec).unwrap();

    let truth_vec = vec![1.0, 1.0, 2.0, 1.0, -1.0, 0.0];

    let truth = Array::from_shape_vec((2, 3), truth_vec).unwrap();

    for (bv, xv) in b.axis_iter(Axis(1)).zip(truth.axis_iter(Axis(1))) {
        println!("b ={:?}, t = {:?}", bv, xv);
        let x = SolveLinear::compute(&a, &bv);
        assert!(x.is_ok());
        let values = x.unwrap();
        assert_eq_within_tol!(&values, &xv, 1e-5);
    }

}
