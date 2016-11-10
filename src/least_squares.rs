//! This module contains the `LeastSquares` trait, which acts as an
//! entry point, which is used to compute least squares solutions.
use impl_prelude::*;
use lapack::c::{sgels, sgelsd, dgels, dgelsd, cgels, cgelsd, zgels, zgelsd};

pub struct LeastSquaresSolution<T, D: Dimension> {
    pub solution: Array<T, D>,
    pub rank: usize,
}

#[derive(Debug)]
pub enum LeastSquaresError {
    /// One of the matrices has an invalid layout.
    BadLayout,

    /// Matrix `a` and `b` have inconsistent layouts (i.e. row
    /// vs. column-major)
    InconsistentLayout,

    /// `a` and `b` have different numbers of rows.
    InconsistentDimensions(usize, usize),

    /// `a` is not full rank
    Degenerate,

    /// Should never happend.
    IllegalParameter(i32),
}

/// Multivariable least squares problem.
///
/// Find solutions to the following optimization;
///
///  min ||Ax - b||^2
///
/// for an `m` by `n` matrix A. The solution is unique when the matrix
/// A is overdetermined, or of full rank. When A is underdtermined,
/// the solution returned is one of minimum norm.
///
/// The `compute_multi_*` functions compute independents solutions
/// `x_i` to min(||A*`x_i` - `b_i`||) for each column `b_i` of `b`. They
/// do *not* compute the solution `X` of min(||A*X - b||).
pub trait LeastSquares: Sized + Clone {
    /// Returns the solution `x` to the least squares problem
    /// min(||A*x - b||), for a non-degenerate `A`.
    ///
    /// # Errors
    ///
    /// Returns `LeastSquaresError::Degenerate` when the coefficient
    /// matrix `a` is not of full rank (rank(`a`) < min(m, n)).
    fn compute_multi_full_into<D1, D2>
        (a: ArrayBase<D1, Ix2>,
         b: ArrayBase<D2, Ix2>)
         -> Result<LeastSquaresSolution<Self, Ix2>, LeastSquaresError>
        where D1: DataMut<Elem = Self> + DataOwned<Elem = Self>,
              D2: DataMut<Elem = Self> + DataOwned<Elem = Self>;


    /// Similar to `compute_multi_full_into`, but doesn't modify the inputs.
    ///
    /// See [compute_multi_full_into]().
    fn compute_multi_full<D1, D2>(a: &ArrayBase<D1, Ix2>,
                                  b: &ArrayBase<D2, Ix2>)
                                  -> Result<LeastSquaresSolution<Self, Ix2>, LeastSquaresError>
        where D1: Data<Elem = Self>,
              D2: Data<Elem = Self>
    {
        Self::compute_multi_full_into(a.to_owned(), b.to_owned())
    }

    /// Returns the solution `x` to the least squares problem min(||A*x - b||), for any `A`.
    ///
    /// The matrix `a` can possibly be degenerate.
    fn compute_multi_degenerate<D1, D2>
        (a: &ArrayBase<D1, Ix2>,
         b: &ArrayBase<D2, Ix2>)
         -> Result<LeastSquaresSolution<Self, Ix2>, LeastSquaresError>
        where D1: Data<Elem = Self>,
              D2: Data<Elem = Self>
    {
        Self::compute_multi_degenerate_into(a.to_owned(), b.to_owned())
    }

    /// Similar to `compute_multi_degenerate_into`, but doesn't modify the inputs.
    fn compute_multi_degenerate_into<D1, D2>
        (a: ArrayBase<D1, Ix2>,
         b: ArrayBase<D2, Ix2>)
         -> Result<LeastSquaresSolution<Self, Ix2>, LeastSquaresError>
        where D1: DataMut<Elem = Self> + DataOwned<Elem = Self>,
              D2: DataMut<Elem = Self> + DataOwned<Elem = Self>;

    /// Returns the solution `x` to the least squares problem min(||A*x - b||) for any `A`.
    ///
    /// This method first assumes that the coefficent matrix `a` is
    /// non-degenerate and calls `compute_multi_full`. If the matrix
    /// is found to be degenerate, `compute_multi_degenerate` is
    /// called instead.
    ///
    /// # Remarks
    ///
    /// If you know that your matrix is degenerate ahead of
    /// time, it is more effiecient to instead call
    /// `compute_multi_degenerate` instead. If you want to know that
    /// your matrix is non-degenerate and want to do something else in
    /// that case, you should use `compute_multi_full` instead, which
    /// will return a `Degenerate` error.
    ///
    /// This method will never return `LeastSquaresError::Degenerate`.
    fn compute_multi<D1, D2>(a: &ArrayBase<D1, Ix2>,
                             b: &ArrayBase<D2, Ix2>)
                             -> Result<LeastSquaresSolution<Self, Ix2>, LeastSquaresError>
        where D1: Data<Elem = Self>,
              D2: Data<Elem = Self>
    {

        // Assume the matrix is full rank and compute the solution.
        let r = Self::compute_multi_full(a, b);
        match r {
            // For degenerate matrices, call the degenerate version
            Err(LeastSquaresError::Degenerate) => Self::compute_multi_degenerate(a, b),
            // For anything else, just forward the result.
            x => x,
        }
    }

    /// Returns the solution `x` to the least squares problem
    /// min(||A*x - b||) for any `A` and a single column `b`.
    ///
    /// This method first assumes that the coefficent matrix `a` is
    /// non-degenerate and calls `compute_multi_full`. If the matrix
    /// is found to be degenerate, `compute_multi_degenerate` is
    /// called instead.
    ///
    /// # Remarks
    ///
    /// If you know that your matrix is degenerate ahead of
    /// time, it is more effiecient to instead call
    /// `compute_multi_degenerate` instead. If you want to know that
    /// your matrix is non-degenerate and want to do something else in
    /// that case, you should use `compute_multi_full` instead, which
    /// will return a `Degenerate` error.
    ///
    /// This method will never return `LeastSquaresError::Degenerate`.
    fn compute<D1, D2>(a: &ArrayBase<D1, Ix2>,
                       b: &ArrayBase<D2, Ix1>)
                       -> Result<LeastSquaresSolution<Self, Ix1>, LeastSquaresError>
        where D1: Data<Elem = Self>,
              D2: Data<Elem = Self>
    {
        let n = b.dim();

        // Create a new matrix, where the column vector is a degenerate 2-D matrix.
        let b_mat = match b.to_owned().into_shape((n, 1)) {
            Ok(x) => x,
            Err(_) => return Err(LeastSquaresError::BadLayout),
        };

        // Call the original
        let res = try!(Self::compute_multi(a, &b_mat));

        // Return the
        Ok(LeastSquaresSolution {
            solution: res.solution.into_shape(n).unwrap(),
            rank: res.rank,
        })

    }
}

/// Resize the given solution to the correct matrix size, based on the
/// RHS input size.
fn resize_solution<T: Clone + Default, D>(mut b_sol: ArrayBase<D, Ix2>, n: usize) -> Array<T, Ix2>
    where D: DataMut<Elem = T>
{
    let b_dim = b_sol.dim();

    if b_dim.0 > n {
        // If the matrix is overdetermined, we just need to truncate the
        // solution.
        b_sol.slice_mut(s![0..n as isize, ..]).to_owned()
    } else {
        // Otherwise, it's underdetermined, and we need to extend the solution
        let mut extended_sol = Array::default((n, b_dim.1));
        extended_sol.slice_mut(s![0..b_dim.0 as isize, ..]).assign(&b_sol);
        extended_sol
    }
}

macro_rules! impl_least_squares {
    ($impl_type:ty, $sv_type:ty, $full_func:ident, $degen_func:ident) => (
        impl LeastSquares for $impl_type {
            fn compute_multi_full_into<D1, D2>(
                mut a: ArrayBase<D1, Ix2>,
                mut b: ArrayBase<D2, Ix2>)
                -> Result<LeastSquaresSolution<Self, Ix2>, LeastSquaresError>

                where D1: DataMut<Elem=Self> + DataOwned<Elem = Self>,
                      D2: DataMut<Elem=Self> + DataOwned<Elem = Self> {

                let a_dim = a.dim();
                let b_dim = b.dim();

// confirm same number of rows.
                if a_dim.0 != b_dim.0 {
                    return Err(LeastSquaresError::InconsistentDimensions(a_dim.0, b_dim.0));
                }

// confirm layouts
                let (a_slice, layout, lda) = match slice_and_layout_mut(&mut a) {
                    Some(x) => x,
                    None => return Err(LeastSquaresError::BadLayout)
                };

// compute result
                let info = {
                    let (b_slice, ldb) = match slice_and_layout_matching_mut(&mut b, layout) {
                        Some(x) => x,
                        None => return Err(LeastSquaresError::InconsistentLayout)
                    };

                    $full_func(layout, b'N', a_dim.0 as i32 , a_dim.1 as i32, b_dim.1 as i32,
                               a_slice, lda as i32,
                               b_slice, ldb as i32)
                };

                if info == 0 {
                    Ok(LeastSquaresSolution {
                        solution: resize_solution(b, a_dim.1),
                        rank: cmp::min(a_dim.0, a_dim.1)
                    })
                } else if info < 0 {
                    Err(LeastSquaresError::IllegalParameter(-info))
                } else {
                    Err(LeastSquaresError::Degenerate)
                }
            }

            fn compute_multi_degenerate_into<D1, D2>(
                mut a: ArrayBase<D1, Ix2>,
                mut b: ArrayBase<D2, Ix2>)
                -> Result<LeastSquaresSolution<Self, Ix2>, LeastSquaresError>
                where D1: DataMut<Elem=Self> + DataOwned<Elem = Self>,
                      D2: DataMut<Elem=Self> + DataOwned<Elem = Self> {

                let a_dim = a.dim();
                let b_dim = b.dim();

// confirm same number of rows.
                if a_dim.0 != b_dim.0 {
                    return Err(LeastSquaresError::InconsistentDimensions(a_dim.0, b_dim.0));
                }

// confirm layouts
                let (a_slice, layout, lda) = match slice_and_layout_mut(&mut a) {
                    Some(x) => x,
                    None => return Err(LeastSquaresError::BadLayout)
                };

                let mut svs: Array<$sv_type, Ix1> = Array::default(cmp::min(a_dim.0, a_dim.1));
                let mut rank: i32 = 0;

// compute result
                let info = {
                    let (b_slice, ldb) = match slice_and_layout_matching_mut(&mut b, layout) {
                        Some(x) => x,
                        None => return Err(LeastSquaresError::InconsistentLayout)
                    };

                    $degen_func(layout, a_dim.0 as i32 , a_dim.1 as i32, b_dim.1 as i32,
                                a_slice, lda as i32,
                                b_slice, ldb as i32,
                                svs.as_slice_mut().unwrap(), 0.0,
                                &mut rank)
                };

                if info == 0 {
                    Ok(LeastSquaresSolution {
                        solution: resize_solution(b, a_dim.1),
                        rank: rank as usize })
                } else if info < 0 {
                    Err(LeastSquaresError::IllegalParameter(-info))
                } else {
                    unreachable!();
                }
            }
        }
    )
}

impl_least_squares!(f32, f32, sgels, sgelsd);
impl_least_squares!(f64, f64, dgels, dgelsd);
impl_least_squares!(c32, f32, cgels, cgelsd);
impl_least_squares!(c64, f64, zgels, zgelsd);
