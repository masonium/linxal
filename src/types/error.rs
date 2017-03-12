//! Error enum
pub use svd::types::SVDError;
pub use eigenvalues::types::{EigenError};
pub use solve_linear::types::SolveError;
pub use least_squares::LeastSquaresError;
pub use generate::GenerateError;
pub use factorization::qr::QRError;
pub use factorization::lu::LUError;
pub use factorization::cholesky::CholeskyError;

/// Universal `linxal` error enum
///
/// This enum can be used as a catch-all for errors from `linxal`
/// computations.
#[derive(Debug)]
pub enum Error {
    /// Error from an SVD opeartion
    SVD(SVDError),

    /// Error from an eigenvalue operation (general or symmetric)
    Eigen(EigenError),

    /// Error from attempting a least-squares solution
    LeastSquares(LeastSquaresError),

    /// Error from solving a linear equation
    SolveLinear(SolveError),

    /// Error from computing a QR-decomposition
    QR(QRError),

    /// Error from computing an LU-decomposition
    LU(LUError),

    /// Error from computing a Cholesky decomposition
    Cholesky(CholeskyError),

    /// Error from attempting to generate a matrix
    Generate(GenerateError),
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

impl From<LeastSquaresError> for Error {
    fn from(e: LeastSquaresError) -> Error {
        Error::LeastSquares(e)
    }
}

impl From<SolveError> for Error {
    fn from(e: SolveError) -> Error {
        Error::SolveLinear(e)
    }
}

impl From<QRError> for Error {
    fn from(e: QRError) -> Error {
        Error::QR(e)
    }
}

impl From<LUError> for Error {
    fn from(e: LUError) -> Error {
        Error::LU(e)
    }
}

impl From<CholeskyError> for Error {
    fn from(e: CholeskyError) -> Error {
        Error::Cholesky(e)
    }
}

impl From<GenerateError> for Error {
    fn from(e: GenerateError) -> Error {
        Error::Generate(e)
    }
}
