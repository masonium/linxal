//! Module for testing various matrix properties, such as being
//! diagonal or unitary.
//!
//! Most tests have a regular and a `_tol` version, which takes a
//! specific tolerance. In the latter case, all floating-point
//! comparisons are done within the tolerance. In the former case, a
//! default tolerance is used (`T::tol()` for type `T`).

use impl_prelude::*;
use util::conj_t;
use num_traits::{Float};

/// Return true iff the matrix is square.
pub fn is_square_size(d: &Ix2) -> bool {
    d[0] == d[1]
}

/// Return the default tolerance used for most property testing.
pub fn default_tol<T, D>(mat: &ArrayBase<D, Ix2>)
                              -> T::RealPart
    where T: LinxalScalar,
          D: Data<Elem=T> {
    T::tol() * mat.iter().map(|x| x.mag())
        .fold(T::RealPart::neg_infinity(), |x, y| if x > y { x } else { y })

}

/// Returns true iff the matrix is diagonal.
pub fn is_diagonal<T: LinxalScalar, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>) -> bool {
    is_diagonal_tol(mat, default_tol(mat))
}

/// Returns true iff the matrix is diagonal, within a given tolerance.
///
/// The input matrix need not be square to be diagonal.
pub fn is_diagonal_tol<T, D, F>(mat: &ArrayBase<D, Ix2>, tolerance: F)
                                -> bool
    where T: LinxalScalar,
          D: Data<Elem=T>,
          F: Into<T::RealPart> {
    let tol = tolerance.into();

    !mat.indexed_iter().any(|(i, x)| i.0 != i.1 && x.mag() > tol)
}

/// Returns true iff the matrix is an identity matrix.
pub fn is_identity<T: LinxalScalar, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>) -> bool {
    is_identity_tol(mat, T::tol())
}

/// Returns true iff the matrix is an identity matrix, within a
/// given tolerance.
pub fn is_identity_tol<T: LinxalScalar, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>, tol: T::RealPart) -> bool {
    if !is_square_size(&mat.raw_dim()) {
        return false;
    }
    !mat.indexed_iter().any(
        |(i, x)| (i.0 != i.1 && x.mag() > tol) || (i.0 == i.1 && (*x - T::one()).mag() > tol))
}

/// Return true iff the matrix if Hermitian (or symmetric in the real
/// case).
pub fn is_symmetric<T: LinxalScalar, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>) -> bool {
    is_symmetric_tol(mat, default_tol(mat))
}

/// Return true iff the matrix if Hermitian (or symmetric in the real
/// case), within a given tolerance.
pub fn is_symmetric_tol<T: LinxalScalar, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>, tol: T::RealPart) -> bool {
    let d = mat.dim();

    if !is_square_size(&mat.raw_dim()) {
        return false;
    }

    for i in 0..d.0 {
        for j in 0..i {
            if (mat[(i, j)] - mat[(j, i)].cj()).mag() > tol {
                return false;
            }
        }
    }

    true
}

/// Return true iff the matrix is unitary, within tolerance.
///
/// A unitary matrix U is a square matrix that satisfiies
///
/// $$U^H\ cdot U = U \cdot U^H = I$$
///
pub fn is_unitary_tol<T: LinxalScalar, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>, tol: T::RealPart) -> bool {
    if !is_square_size(&mat.raw_dim()) {
        return false;
    }
    let prod = mat.dot(&conj_t(&mat));
    is_identity_tol(&prod, tol)
}

/// Return true iff the matrix is unitary
///
/// A unitary matrix U is a square matrix taht satisfiies
///
/// $$U^H\ cdot U = U \cdot U^H = I$$
///
pub fn is_unitary<T: LinxalScalar, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>) -> bool {
    is_unitary_tol(mat, default_tol(mat))
}

/// Return true iff the matrix is triangular / trapezoidal, with the
/// side specified by `uplo`.
pub fn is_triangular_tol<T, D, F>(mat: &ArrayBase<D, Ix2>, uplo: Symmetric, tolerance: F)
                                  -> bool
    where T: LinxalScalar,
          D: Data<Elem=T>,
          F: Into<T::RealPart> {

    match uplo {
        Symmetric::Upper => get_lower_bandwidth_tol(mat, tolerance) == 0,
        Symmetric::Lower => get_upper_bandwidth_tol(mat, tolerance) == 0
    }
}

/// Return true iff the matrix is triangular / trapezoidal, with the
/// side specified by `uplo`.
///
/// Uses the defalut toleranace for comparisons to 0.
pub fn is_triangular<T, D>(mat: &ArrayBase<D, Ix2>, uplo: Symmetric)
                                  -> bool
    where T: LinxalScalar,
          D: Data<Elem=T> {

    is_triangular_tol(mat, uplo, default_tol(&mat))
}

/// Return the lower bandwidth of the matrix, within the specified
/// tolerance.
///
/// Zero-dimension matrices have 0 bandwidth.
pub fn get_lower_bandwidth_tol<T: LinxalScalar, D: Data<Elem=T>, F: Into<T::RealPart>>(mat: &ArrayBase<D, Ix2>, tol: F) -> usize {
    let dim = mat.dim();
    let r = dim.0 - 1;
    let t = tol.into();

    for b in 0..r {
        for i in 0..cmp::min(b+1, dim.1) {
            let a = mat[[r - b + i, i]];
            if a.mag() > t {
                return r - b;
            }
        }
    }
    0
}

/// Return the lower bandwidth of the matrix.
///
/// Uses a tolerance of `T::tol()` * `max |a_ij|`
pub fn get_lower_bandwidth<T: LinxalScalar, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>) -> usize {
    get_lower_bandwidth_tol(mat, default_tol(mat))
}


/// Return the upper bandwidth of the matrix, within the specified
/// tolerance.
///
/// Zero-dimension matrices have 0 bandwidth.
pub fn get_upper_bandwidth_tol<T: LinxalScalar, D: Data<Elem=T>, F: Into<T::RealPart>>(mat: &ArrayBase<D, Ix2>, tol: F) -> usize {
    let dim = mat.dim();
    let c = dim.1 - 1;
    let t = tol.into();

    for b in 0..c {
        for i in 0..cmp::min(b+1, dim.0) {
            let a = mat[[i, c - b + i]];
            if a.mag() > t {
                return c - b;
            }
        }
    }
    0
}

/// Return the upper bandwidth of the matrix.
///
/// Uses a tolerance of `T::tol()` * `max |a_ij|`
pub fn get_upper_bandwidth<T: LinxalScalar, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>) -> usize {
    get_upper_bandwidth_tol(mat, default_tol(mat))
}
