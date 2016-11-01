extern crate num_traits;
extern crate ndarray;
#[macro_use]
extern crate rula;
extern crate lapack;

use ndarray::prelude::*;
use rula::prelude::*;
use ndarray::{Ix2};
use num_traits::{One, Zero};

/// Identity matrix SVD
pub fn svd_test_identity<SV: SingularValue + Magnitude, T: SVD<SV> + One + Zero>() {
    const N: usize = 100;
    let m: Array<T, Ix2> = Array::eye(N);
    let solution = SVD::compute(&m, false, false);
    assert!(solution.is_ok());
    let values  = solution.unwrap();

    let truth =  Array::from_vec(vec![SV::one(); N]);
    assert_in_tol!(&values.values, &truth, 1e-5);
}

#[test]
pub fn svd_test_identity_f32() {
    svd_test_identity::<f32, f32>();
}
#[test]
pub fn svd_test_identity_f64() {
    svd_test_identity::<f64, f64>();
}
#[test]
pub fn svd_test_identity_c32() {
    svd_test_identity::<f32, c32>();
}
#[test]
pub fn svd_test_identity_c64() {
    svd_test_identity::<f64, c64>();
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

            let solution = SVD::compute(&m, false, false);
            let values = match solution {
                Err(_) => { assert!(false); return },
                Ok(x) => x.values
            };

            let truth = Array::linspace(100.0, 1.0, N);
            assert_in_tol!(&values, &truth, 1e-5);
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
            for i in 0..N {
                svs[i] = ((N - i) as $sv).into();
            }

            m.diag_mut().assign(&svs);

            let solution = SVD::compute(&m, false, false);
            let values = match solution {
                Err(_) => { assert!(false); return },
                Ok(x) => x.values
            };

            let truth = Array::linspace(100.0 as $sv, 1.0, N);
            assert_in_tol!(&values, &truth, 1e-5);
        }
    )
}


stlc!(svd_test_linspace_c32, c32, f32);
stlc!(svd_test_linspace_c64, c64, f64);
