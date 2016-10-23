pub extern crate ndarray;
pub extern crate lapack;

pub mod eigenvalues;

pub use eigenvalues::generic::Eigen;
pub use eigenvalues::symmetric::SymEigen;

pub use ndarray as nd;
pub use lapack as lp;
