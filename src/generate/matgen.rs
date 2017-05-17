//! matrix generators
//!
//! The tratis and structures in this crate are used to generate
//! random matrices that have specified eigen- or singular values,
//! bands, symmetrty, packing, etc.

use impl_prelude::*;
use super::internal::{slatmt, dlatmt, clatmt, zlatmt};
use super::internal::{slaror, dlaror, claror, zlaror};
use rand::{Rng, thread_rng};
use rand::distributions::{Range, IndependentSample};
use lapack::c::Layout;
use num_traits::{Float};
use generate::types::*;


/// Newtype for functions retuning a matrix and a set of singular values.
pub type MatrixSVPair<T: LinxalImplScalar> = (Array<T, Ix2>, Array<T::RealPart, Ix1>);

/// Scalar trait for generating random matrices.
pub trait MG: LinxalImplScalar {
    /// Create a matrix based on the specified arguments.
    fn general(gen: &mut GenerateArgs<Self>) -> Result<MatrixSVPair<Self>, GenerateError>;

    /// Create a unitary matrix based on the specified arguments.
    fn unitary(gen: &mut GenerateArgs<Self>) -> Result<Array<Self, Ix2>, GenerateError>;
}

/// Determines how the singular/eigenvalues are created.
enum ValuesOption<T> {
    /// Values are created by evenly spacing values within a set of
    /// boundaries.
    EvenUniform(T, T),
    RandomUniform(T, T),
    Exact(Vec<T>)
}

#[repr(u8)]
#[derive(PartialEq, Eq, Clone, Copy)]
enum GenerateSymmetry {
    Positive = b'P',
    Symmetric = b'H',
    NoSymmetry = b'N'
}

/// Generating arguments
pub struct GenerateArgs<T: MG> {
    m: usize,
    n: usize,

    rank: Option<usize>,
    bands: Option<(usize, usize)>,
    symmetry: GenerateSymmetry,
    values: ValuesOption<T::RealPart>,
    packing: Packing,

    seed: [i32; 4],
    workspace: Array<T, Ix1>,
}

impl <T: MG> GenerateArgs<T> {
    /// Returns the desired rank of the output matrix.
    pub fn rank(&self) -> usize {
        let k = cmp::min(self.m, self.n);
        match self.rank {
            None => k,
            Some(i) => cmp::min(k, i)
        }
    }

    /// Returns an error iff calling generate would yield an error.
    pub fn validate(&self) -> Result<(), GenerateError> {
        // Any symmetric matrix needs to be square, with equivalent bands.
        // Asymmetric matrices must have no packing.
        match self.symmetry {
            GenerateSymmetry::Positive | GenerateSymmetry::Symmetric => {
                if self.m != self.n {
                    return Err(GenerateError::NotSquare);
                }
                match self.bands {
                    None => (),
                    Some((kl, ku)) => {
                        if kl != ku {
                            return Err(GenerateError::UnequalBands);
                        }
                    }
                }
            },
            GenerateSymmetry::NoSymmetry => {
                match self.packing {
                    Packing::Full => (),
                    Packing::UpperOnly | Packing::LowerOnly => {
                        return Err(GenerateError::InvalidPacking);
                    }
                }
            }
        };

        // We need at least as many values as the rank of the
        // matrix when exact values are provided.
        if let ValuesOption::Exact(ref v) = self.values {
            if v.len() < self.rank() {
                return Err(GenerateError::NotEnoughValues);
            }
        }

        Ok(())
    }

    /// Generate values based on the `ValuesOption`, rank, and
    /// symmetry.
    fn values(&self) -> Result<Array<T::RealPart, Ix1>, GenerateError> {
        let ns = cmp::min(self.m, self.n);

        let nz = match self.rank {
            None => ns,
            Some(k) => if k > ns { return Err(GenerateError::InvalidRank) } else { k }
        };

        // Create the non-zero entries based on the ValuesOption
        let mut values = match self.values {
            ValuesOption::EvenUniform(a, b) => {
                if nz > 1 {
                    Array::linspace(a, b, nz).into_raw_vec()
                } else if nz == 1 {
                    vec![(a+b) * 0.5.into()]
                } else {
                    Vec::new()
                }
            },
            ValuesOption::RandomUniform(a, b) => {
                let mut rng = thread_rng();
                let range = Range::new(a, b);
                (0..nz).map(|_| range.ind_sample(&mut rng)).collect()
            }
            ValuesOption::Exact(ref v) => {
                if v.len() < nz {
                    return Err(GenerateError::NotEnoughValues);
                }
                let mut vs = v.clone();
                vs.resize(nz, T::RealPart::zero());
                vs
            }
        };

        // Take the absolute value for positive or non-symmetric
        // matrices.
        if self.symmetry != GenerateSymmetry::Symmetric {
            for x in &mut values {
                *x = x.abs();
            }
        }

        // Extend to the full set with 0s
        values.resize(ns, T::RealPart::zero());

        Ok(Array::from_vec(values))
    }
}

macro_rules! impl_mat_gen {
    ($impl_type:ty, $gen_gen:ident, $ortho_gen:ident) => (
        impl MG for $impl_type {
            fn general(gen: &mut GenerateArgs<Self>)
                       -> Result<MatrixSVPair<Self>, GenerateError> {
                /// Validate the option inputs.
                try!(gen.validate());

                let mut arr = matrix_with_layout((gen.m, gen.n), Layout::ColumnMajor);

                let mut values = try!(gen.values());

                assert!(values.len() >= cmp::min(gen.m, gen.n));

                let dist = b'U'; // we always proved values, so this is irrelevant.
                let mode = 0;


                let (kl, ku) = gen.bands.unwrap_or((gen.m, gen.n));

                let info = {
                    let (slice, _, lda) = slice_and_layout_mut(&mut arr).unwrap();
                    $gen_gen(gen.m as i32, gen.n as i32, dist, &mut gen.seed,
                             gen.symmetry as u8, values.as_slice_mut().unwrap(), mode,
                             1.0, 1.0,
                             gen.rank.unwrap_or(cmp::min(gen.m, gen.n)) as i32,
                             kl as i32, ku as i32, gen.packing as u8,
                             slice, lda as i32, gen.workspace.as_slice_mut().unwrap())
                };

                match info {
                    0 => Ok((arr, values)),
                    -1 => Err(GenerateError::NotSquare),
                    -2 | -3 | -5 | -7 | -8 | -10 | -11 | -12 | -14 => unimplemented!(),
                    1 | 2 | 3 => unreachable!(),
                    _ => unreachable!()
                }
            }

            fn unitary(gen: &mut GenerateArgs<Self>)
                       -> Result<Array<Self, Ix2>, GenerateError> {
                let d = (gen.m, gen.n).f();
                let mut arr = Array::default(d);

                let mut info: i32 = 0;
                {
                    let (slice, _, lda) = slice_and_layout_mut(&mut arr).unwrap();
                    $ortho_gen(b'L', b'I', gen.m as i32, gen.n as i32,
                               slice, lda as i32, &mut gen.seed, gen.workspace.as_slice_mut().unwrap(),
                               &mut info);
                }

                if info == 0 {
                    Ok(arr)
                } else if info < 0 {
                    Err(GenerateError::IllegalParameter(-info))
                } else  {
                    unreachable!();
                }

            }
        }
    )
}

impl_mat_gen!(f32, slatmt, slaror);
impl_mat_gen!(f64, dlatmt, dlaror);
impl_mat_gen!(c32, clatmt, claror);
impl_mat_gen!(c64, zlatmt, zlaror);

/// Structure for creating positive semi-definite matrices.
pub struct RandomSemiPositive<T: MG> {
    args: GenerateArgs<T>
}

impl <T: MG> RandomSemiPositive<T> {
    /// Create a new matrix generator for random semi-positive definite matrices.
    pub fn new<Rand: Rng>(n: usize, rand: &mut Rand) -> RandomSemiPositive<T> {
        RandomSemiPositive {
            args:
            GenerateArgs {
                m: n, n: n,

                seed: new_seed(rand),
                workspace: new_workspace(n, n),

                rank: None, bands: None,
                symmetry: GenerateSymmetry::Positive,
                values: ValuesOption::EvenUniform(1.0.into(), 10.0.into()),
                packing: Packing::Full
            }
        }
    }


    /// Set the rank of the matrix.
    ///
    /// The rank is capped to the size of the matrix.
    ///
    /// # Remarks
    ///
    /// The rank is ignored when the eigenvalues / singular
    /// values are given as input.
    pub fn rank(&mut self, n: usize) -> &mut Self {
        self.args.rank = Some(n);
        self
    }

    /// Set the matrix to be full rank.
    ///
    /// # Remarks
    ///
    /// The rank is ignored when the eigenvalues / singular
    /// values are given as input.
    pub fn full_rank(&mut self) -> &mut Self {
        self.args.rank = None;
        self
    }

    /// Set the upper and lower band of the matrix.
    ///
    /// If the band is larger than the matrix, it is interpreted as a
    /// full-sized matrix.
    pub fn bands(&mut self, band: usize) -> &mut Self {
        self.args.bands = Some((band, band));
        self
    }

    /// Set the matrix to be full bandwidth.
    pub fn full_bands(&mut self) -> &mut Self {
        self.args.bands = None;
        self
    }

    /// Set the matrix to be diagonal.
    pub fn diagonal(&mut self) -> &mut Self {
        self.args.bands = Some((0, 0));
        self
    }

    /// Set how the entries of the matrix are packed.
    ///
    /// # Remarks
    /// Only symmetric matrices can have non-`Full` packing.
    pub fn packing(&mut self, packing: Packing) -> &mut Self {
        self.args.packing = packing;
        self
    }

    /// Set the singular values to the specified values.
    ///
    /// When the rank of the matrix is specified as `k`, any values
    /// after the `k`th are ignored and set to zero when the matrix is
    /// generated.
    ///
    /// The absolute value of all entries is taken.
    pub fn singular_values(&mut self, values: &[<T as LinxalImplScalar>::RealPart]) -> &mut Self {
        self.args.values = ValuesOption::Exact(values.iter().map(|x| x.abs()).collect());
        self
    }

    /// Set the singular_values to the specified values.
    #[inline]
    pub fn sv(&mut self, values: &[<T as LinxalImplScalar>::RealPart]) -> &mut Self {
        self.singular_values(values)
    }

    /// Draw the singular_values from a uniform distribution.
    ///
    /// The absolute value of all entries is taken, to ensure positive
    /// semi-definiteness.
    pub fn sv_random_uniform(&mut self, min: T::RealPart, max: T::RealPart) -> &mut Self {
        self.args.values = ValuesOption::RandomUniform(min, max);
        self
    }

    /// Generate a matrix matching the specifications previously
    /// specified.
    pub fn generate(&mut self) -> Result<Array<T, Ix2>, GenerateError> {
        MG::general(&mut self.args).map(|x| x.0)
    }

    /// Generate a matrix matching the specifications, and singular values
    pub fn generate_with_sv(&mut self) -> Result<MatrixSVPair<T>, GenerateError> {
        MG::general(&mut self.args)
    }

}

/// Structure for creating symmetric matrices.
pub struct RandomSymmetric<T: MG> {
    args: GenerateArgs<T>
}

impl <T: MG> RandomSymmetric<T> {
    /// Create a new matrix generator for random semi-positive definite matrices.
    pub fn new<Rand: Rng>(n: usize, rand: &mut Rand) -> RandomSymmetric<T> {
        RandomSymmetric {
            args:
            GenerateArgs {
                m: n, n: n,

                seed: new_seed(rand),
                workspace: new_workspace(n, n),

                rank: None, bands: None,
                symmetry: GenerateSymmetry::Positive,
                values: ValuesOption::EvenUniform(1.0.into(), 10.0.into()),
                packing: Packing::Full
            }
        }
    }


    /// Set the rank of the matrix.
    ///
    /// The rank is capped to the size of the matrix.
    ///
    /// # Remarks
    ///
    /// The rank is ignored when the eigenvalues / singular
    /// values are given as input.
    pub fn rank(&mut self, n: usize) -> &mut Self {
        self.args.rank = Some(n);
        self
    }

    /// Set the matrix to be full rank.
    ///
    /// # Remarks
    ///
    /// The rank is ignored when the eigenvalues / singular
    /// values are given as input.
    pub fn full_rank(&mut self) -> &mut Self {
        self.args.rank = None;
        self
    }

    /// Set the upper and lower band of the matrix.
    ///
    /// If the band is larger than the matrix, it is interpreted as a
    /// full-sized matrix.
    pub fn bands(&mut self, band: usize) -> &mut Self {
        self.args.bands = Some((band, band));
        self
    }

    /// Set the matrix to be full bandwidth.
    pub fn full_bands(&mut self) -> &mut Self {
        self.args.bands = None;
        self
    }

    /// Set the matrix to be diagonal.
    pub fn diagonal(&mut self) -> &mut Self {
        self.args.bands = Some((0, 0));
        self
    }

    /// Set how the entries of the matrix are packed.
    pub fn packing(&mut self, packing: Packing) -> &mut Self {
        self.args.packing = packing;
        self
    }

    /// Set the eigenvalues to the specified values.
    ///
    /// When the rank of the matrix is specified as `k`, any values
    /// after the `k`th are ignored and set to zero when the matrix is
    /// generated.
    pub fn eigenvalues(&mut self, values: &[<T as LinxalImplScalar>::RealPart]) -> &mut Self {
        self.args.values = ValuesOption::Exact(values.iter().map(|x| x.abs()).collect());
        self
    }

    /// Set the eigenvalues to the specified values.
    ///
    /// Equivalent to the `eigenvalues` function.
    #[inline]
    pub fn ev(&mut self, values: &[<T as LinxalImplScalar>::RealPart]) -> &mut Self {
        self.eigenvalues(values)
    }

    /// Draw the eigenvalues from a uniform distribution.
    pub fn ev_random_uniform<F: Into<T::RealPart>>(&mut self, min: F, max: F) -> &mut Self {
        self.args.values = ValuesOption::RandomUniform(min.into(), max.into());
        self
    }

    /// Generate a matrix matching the specifications previously
    /// specified.
    pub fn generate(&mut self) -> Result<Array<T, Ix2>, GenerateError> {
        MG::general(&mut self.args).map(|x| x.0)
    }

    /// Generate a matrix matching the specifications, and return the
    /// eigenvalues of the generated matrix.
    ///
    /// The returned eigenvalues include any changes made
    pub fn generate_with_ev(&mut self) -> Result<MatrixSVPair<T>, GenerateError> {
        MG::general(&mut self.args)
    }
}


/// Structure for creating general rectangular matrices with specific eigenvalues
pub struct RandomGeneral<T: MG> {
    args: GenerateArgs<T>
}

impl <T: MG> RandomGeneral<T> {
    /// Create a new matrix generator for random rectangular matrices.
    pub fn new<Rand: Rng>(m: usize, n: usize, rand: &mut Rand) -> RandomGeneral<T> {
        RandomGeneral {
            args:
            GenerateArgs {
                m: m, n: n,

                seed: new_seed(rand),
                workspace: new_workspace(m, n),

                rank: None,
                bands: None,
                symmetry: GenerateSymmetry::NoSymmetry,
                values: ValuesOption::EvenUniform(1.0.into(), 10.0.into()),
                packing: Packing::Full
            }
        }
    }

    /// Set the rank of the matrix.
    ///
    /// The rank is capped to the size of the matrix.
    ///
    /// # Remarks
    ///
    /// The rank is ignored when the eigenvalues / singular
    /// values are given as input.
    pub fn rank(&mut self, n: usize) -> &mut Self {
        self.args.rank = Some(n);
        self
    }

    /// Set the matrix to be full rank.
    ///
    /// # Remarks
    ///
    /// The rank is ignored when the eigenvalues / singular
    /// values are given as input.
    pub fn full_rank(&mut self) -> &mut Self {
        self.args.rank = None;
        self
    }

    /// Set the upper and lower band of the matrix.
    ///
    /// If the band is larger than the matrix, it is interpreted as a
    /// full-sized matrix.
    #[deprecated]
    pub fn bands(&mut self, lower: usize, upper: usize) -> &mut Self {
        self.args.bands = Some((lower, upper));
        self
    }

    /// Make the returned matrix upper-triangular or upper-trapezoidal.
    pub fn upper(&mut self) -> &mut Self {
        self.args.bands = Some((0, self.args.n));
        self
    }

    /// Make the returned matrix lower-triangular or lower-trapezoidal.
    pub fn lower(&mut self) -> &mut Self {
        self.args.bands = Some((self.args.m, 0));
        self
    }


    /// Set the matrix to be full bandwidth.
    pub fn full_bands(&mut self) -> &mut Self {
        self.args.bands = None;
        self
    }

    /// Set the matrix to be diagonal.
    pub fn diagonal(&mut self) -> &mut Self {
        self.args.bands = Some((0, 0));
        self
    }

    /// Set the singular values to the specified values.
    ///
    /// When the rank of the matrix is specified as `k`, any values
    /// after the `k`th are ignored and set to zero when the matrix is
    /// generated.
    pub fn singular_values(&mut self, values: &[<T as LinxalImplScalar>::RealPart]) -> &mut Self {
        self.args.values = ValuesOption::Exact(values.iter().map(|x| x.abs()).collect());
        self
    }

    /// Set the singular_values to the specified values.
    #[inline]
    pub fn sv(&mut self, values: &[<T as LinxalImplScalar>::RealPart]) -> &mut Self {
        self.singular_values(values)
    }

    /// Draw the singular_values from a uniform distribution.
    pub fn sv_random_uniform<F: Into<T::RealPart>>(&mut self, min: F, max: F) -> &mut Self {
        self.args.values = ValuesOption::RandomUniform(min.into(), max.into());
        self
    }

    /// Generate a matrix
    pub fn generate(&mut self) -> Result<Array<T, Ix2>, GenerateError> {
        MG::general(&mut self.args).map(|x| x.0)
    }

    /// Generate a matrix
    pub fn generate_with_sv(&mut self) -> Result<MatrixSVPair<T>, GenerateError> {
        MG::general(&mut self.args)
    }
}


/// Structure for creating unitary matrices.
pub struct RandomUnitary<T: MG> {
    args: GenerateArgs<T>
}

impl<T: MG> RandomUnitary<T> {
    /// Returns a new unitary matrix generator.
    pub fn new<Rand: Rng>(n: usize, rand: &mut Rand) -> RandomUnitary<T> {
        RandomUnitary {
            args:
            GenerateArgs {
                m: n, n: n,

                seed: new_seed(rand),
                workspace: new_workspace(n, n),

                rank: None, bands: None,
                symmetry: GenerateSymmetry::NoSymmetry,
                values: ValuesOption::EvenUniform(1.0.into(), 10.0.into()),
                packing: Packing::Full
            }
        }
    }

    /// Generate a unitary matrix.
    pub fn generate(&mut self) -> Result<Array<T, Ix2>, GenerateError> {
        MG::unitary(&mut self.args)
    }
}
