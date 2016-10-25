use svd::types::SVDError;
use eigenvalues::types::EigenError;

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
