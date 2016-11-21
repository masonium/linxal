//! Various properties of matrices.
use impl_prelude::*;
use util::conj_t;

/// Returns true iff the matrices is diagonal.
pub fn is_diagonal<T: LinxalScalar, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>) -> bool {
    is_diagonal_tol(mat, T::tol())
}

/// Returns true iff the matrix is diagonal, within a given
/// tolerance.
///
/// The input matrix need not be square to be diagonal.
pub fn is_diagonal_tol<T: LinxalScalar, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>, tol: T::RealPart) -> bool {
    !mat.indexed_iter().any(|(i, x)| i.0 != i.1 && x.mag() > tol)
}

/// Returns true iff the matrix is an identity matrix.
pub fn is_identity<T: LinxalScalar, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>) -> bool {
    is_identity_tol(mat, T::tol())
}

/// Returns true iff the matrix is an identity matrix, within a
/// given tolerance.
pub fn is_identity_tol<T: LinxalScalar, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>, tol: T::RealPart) -> bool {
    !mat.indexed_iter().any(
        |(i, x)| (i.0 != i.1 && x.mag() > tol) || (i.0 == i.1 && (*x - T::one()).mag() > tol))
}

/// Return true iff the matrix is square.
fn is_square_size(d: &Ix2) -> bool {
    d[0] == d[1]
}

/// Return true iff the matrix if Hermitian (or symmetric in the real
/// case).
pub fn is_symmetric<T: LinxalScalar, D: Data<Elem=T>>(mat: &ArrayBase<D, Ix2>) -> bool {
    is_symmetric_tol(mat, T::tol())
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
            if (mat[(i, j)] - mat[(j, i)].cj()).mag() < tol {
                return false;
            }
        }
    }

    true
}

/// Return true iff the matrix is unitary, within tolerance.
///
/// A unitary matrix U is a square matrix taht satisfiies
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
    is_unitary_tol(mat, T::tol())
}
