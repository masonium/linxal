/// Error enum returns by various `SolveLinear`-esque compute methods.
#[derive(Debug, Clone)]
pub enum SolveError {
    /// The layout of one of the matrices is not c- or
    /// fortran-contiguous.
    BadLayout,

    /// The layouts of `a` and `b` are different. (i.e. one is
    /// column-major and the other is row-major.)
    InconsistentLayout,

    /// An illegal value was passed into the underlying LAPACK. Users
    /// should never see this error.
    IllegalValue(i32),

    /// The matrix `a`, thus a solution `b` cannot necessarily be
    /// found.
    Singular(i32),

    /// The input `a` matrix is not square.
    NotSquare(usize, usize),

    /// The dimensions of `a` and `b` do not match.
    InconsistentDimensions(usize, usize)
}
