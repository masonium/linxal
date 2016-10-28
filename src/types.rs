use svd::types::SVDError;
use eigenvalues::types::EigenError;

/// enum for symmetric matrix inputs
#[repr(u8)]
pub enum Symmetric {
    /// Read elements from the upper-triangular portion of the matrix
    Upper = b'U',

    /// Read elements from the lower-triangular portion of the matrix
    Lower =  b'L'
}

/// Global error enum
pub enum Error {
    SVD(SVDError),
    Eigen(EigenError)
}

impl From<SVDError> for Error {
    fn from(e: SVDError) -> Error {
        Error::SVD(e)
    }
}

impl From<EigenError> for Error {
    fn from(e: EigenError) -> Error {
        Error::Eigen(e)
    }
}
