pub extern crate ndarray;
pub extern crate lapack;

pub mod matrix;

pub mod eigenvalues;
pub mod svd;
pub mod prelude;

pub use ndarray as nd;
pub use lapack as lp;
