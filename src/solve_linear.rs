use ndarray::prelude::*;
use lapack::c::{sgesv};
use ndarray::{Ix2, Data, DataMut};
use util::*;

#[derive(Debug)]
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

pub trait SolveLinear: Sized + Clone {
    /// Solve the linear system A * x = B for square matrix `a` and rectangular matrix `b`.
    fn compute_multi_into<D1, D2>(a: ArrayBase<D1, Ix2>, b: ArrayBase<D2, Ix2>) ->
        Result<ArrayBase<D2, Ix2>, SolveError>
        where D1: DataMut<Elem=Self>, D2: DataMut<Elem=Self>;

    /// Solve the linear system A * x = b for square matrix `a` and column vector `b`.
    fn compute_into<D1, D2>(a: ArrayBase<D1, Ix2>, b: ArrayBase<D2, Ix>) ->
        Result<Array<Self, Ix>, SolveError>
        where D1: DataMut<Elem=Self>, D2: Data<Elem=Self> {
        let n = b.dim();

        // Create a new view, where the column vector is a degenerate 2-D matrix.
        let b_mat = match b.to_owned().into_shape((n, 1)) {
            Ok(x) => x,
            Err(_) => return Err(SolveError::BadLayout)
        };

        // Call the original
        let res = try!(Self::compute_multi_into(a, b_mat));

        Ok(res.into_shape(n).unwrap())
    }

    /// Solve the linear system A * x = B for square matrix `a` and rectangular matrix `b`.
    fn compute_multi<D1, D2>(a: &ArrayBase<D1, Ix2>, b: &ArrayBase<D2, Ix2>) -> Result<Array<Self, Ix2>, SolveError>
        where D1: Data<Elem=Self>, D2: Data<Elem=Self> {

        let a_copy = a.to_owned();
        let b_copy = b.to_owned();
        Self::compute_multi_into(a_copy,  b_copy)
    }

    /// Solve the linear system A * x = b for square matrix `a` and column vector `b`.
    fn compute<D1, D2>(a: &ArrayBase<D1, Ix2>, b: &ArrayBase<D2, Ix>) -> Result<Array<Self, Ix>, SolveError>
        where D1: Data<Elem=Self>, D2: Data<Elem=Self> {

        let a_copy = a.to_owned();
        let b_copy = b.to_owned();
        Self::compute_into(a_copy, b_copy)
    }

}

impl SolveLinear for f32 {
    fn compute_multi_into<D1, D2>(mut a: ArrayBase<D1, Ix2>, mut b: ArrayBase<D2, Ix2>) ->
        Result<ArrayBase<D2, Ix2>, SolveError>

        where D1: DataMut<Elem=Self>, D2: DataMut<Elem=Self> {

        // Make sure the input is square.
        let dim = a.dim();
        let b_dim = b.dim();

        if dim.0 != dim.1 {
            return Err(SolveError::NotSquare(dim.0, dim.1));
        }
        if dim.0 != b_dim.0 {
            return Err(SolveError::InconsistentDimensions(dim.0, b_dim.0));
        }

        let (slice, layout, lda) = match slice_and_layout_mut(&mut a) {
            Some(x) => x,
            None => return Err(SolveError::BadLayout)
        };

        let (info, ldb) = {
            let (b_slice, ldb) = match slice_and_layout_matching_mut(&mut b, layout) {
                Some(x) => x,
                None => return Err(SolveError::InconsistentLayout)
            };

            let mut perm: Array<i32, Ix> = Array::default(dim.0);

            (sgesv(layout, dim.0 as i32, b_dim.1 as i32,
                  slice, lda as i32,
                  perm.as_slice_mut().unwrap(),
                  b_slice, ldb as i32), ldb)
        };

        if info == 0 {
            Ok(b)
        } else if info < 0 {
            if info == -8 {
                println!("Bad ldb: {}", ldb);
            }
            Err(SolveError::IllegalValue(-info))
        } else {
            Err(SolveError::Singular(info))
        }
    }
}
