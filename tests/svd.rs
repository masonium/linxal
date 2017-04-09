extern crate num_traits;
extern crate ndarray;
#[macro_use]
extern crate linxal;
extern crate lapack;

use ndarray::{Array, Ix2};
use linxal::types::{LinxalMatrix, LinxalMatrixInto, LinxalScalar, c32, c64};
use num_traits::{One};

/// Identity matrix SVD
pub fn svd_test_identity<T: LinxalScalar>() {
    const N: usize = 100;
    let m: Array<T, Ix2> = Array::eye(N);
    let solution = m.singular_values();
    assert!(solution.is_ok());
    let values = solution.unwrap();

    let truth = Array::from_vec(vec![T::RealPart::one(); N]);
    assert_eq_within_tol!(&values, &truth, 1e-5.into());

    let full_sol = m.svd_full().unwrap();
    assert_eq_within_tol!(&full_sol.reconstruct().unwrap(), &m, 1e-4.into());
}

#[test]
pub fn svd_test_identity_f32() {
    svd_test_identity::<f32>();
}
#[test]
pub fn svd_test_identity_f64() {
    svd_test_identity::<f64>();
}
#[test]
pub fn svd_test_identity_c32() {
    svd_test_identity::<c32>();
}
#[test]
pub fn svd_test_identity_c64() {
    svd_test_identity::<c64>();
}


/// SVD for linear set of singular values
macro_rules! stlf {
    ($name: ident, $float_type:ty) => (
        #[test]
        pub fn $name() {
            const N: usize = 100;
            let mut m: Array<$float_type, Ix2> = Array::zeros((N, N));
            let svs = Array::linspace(100.0 as $float_type, 1.0 as $float_type, N);

            m.diag_mut().assign(&svs);

            let values = m.singular_values().unwrap();

            let truth = Array::linspace(100.0, 1.0, N);
            assert_eq_within_tol!(&values, &truth, 1e-5);

            let full_sol = m.svd_full().unwrap();
            assert_eq_within_tol!(&full_sol.reconstruct().unwrap(), &m, 1e-4.into());

            let econ_sol = m.clone().svd_econ_into().unwrap();
            assert_eq_within_tol!(&econ_sol.reconstruct().unwrap(), &m, 1e-4.into());

        }
    )
}

stlf!(svd_test_linspace_f32, f32);
stlf!(svd_test_linspace_f64, f64);


/// SVD for linear set of singular values
macro_rules! stlc {
    ($name: ident, $c_type:ty, $sv:ty) => (
        #[test]
        pub fn $name() {
            const N: usize = 100;
            let mut m: Array<$c_type, Ix2> = Array::zeros((N, N));
            let mut svs = Array::default(N);
            for (i, sv) in svs.iter_mut().enumerate() {
                *sv = ((N - i) as $sv).into();
            }

            m.diag_mut().assign(&svs);

            let values = m.singular_values().unwrap();

            let truth = Array::linspace(100.0 as $sv, 1.0, N);
            assert_eq_within_tol!(&values, &truth, 1e-5);


            let econ_sol = m.svd_econ().unwrap();
            assert_eq_within_tol!(&econ_sol.reconstruct().unwrap(), &m, 1e-4.into());

            let full_sol = m.clone().svd_full_into().unwrap();
            assert_eq_within_tol!(&full_sol.reconstruct().unwrap(), &m, 1e-4.into());
        }
    )
}


stlc!(svd_test_linspace_c32, c32, f32);
stlc!(svd_test_linspace_c64, c64, f64);
