use libc::{c_char, c_float, c_double};
use lapack_sys::c::{lapack_int, lapack_complex_float, lapack_complex_double};

extern "C" {
    pub fn slaror_(side: *const c_char, init: *const c_char, m: *const lapack_int,
                   n: *const lapack_int, a: *mut c_float, lda: *const lapack_int,
                   iseed: *mut lapack_int, x: *mut c_float, info: *mut lapack_int);

    pub fn dlaror_(side: *const c_char, init: *const c_char, m: *const lapack_int,
                   n: *const lapack_int, a: *mut c_double, lda: *const lapack_int,
                   iseed: *mut lapack_int, x: *mut c_double, info: *mut lapack_int);
    pub fn claror_(side: *const c_char, init: *const c_char, m: *const lapack_int,
                   n: *const lapack_int, a: *mut lapack_complex_float, lda: *const lapack_int,
                   iseed: *mut lapack_int, x: *mut lapack_complex_float,
                   info: *mut lapack_int);
    pub fn zlaror_(side: *const c_char, init: *const c_char, m: *const lapack_int,
                   n: *const lapack_int, a: *mut lapack_complex_double, lda: *const lapack_int,
                   iseed: *mut lapack_int, x: *mut lapack_complex_double,
                   info: *mut lapack_int);

    pub fn slatmt_(m: *const lapack_int, n: *const lapack_int,
                   dist: *const c_char, iseed: *mut lapack_int,
                   sym: *const c_char, d: *mut c_float, mode: *const lapack_int,
                   cond: *const c_float, dmax: *const c_float,
                   rank: *const lapack_int, kl: *const lapack_int, ku: *const lapack_int,
                   pack: *const c_char, a: *mut c_float,
                   lda: *const lapack_int, work: *mut c_float, info: *const lapack_int);

    pub fn dlatmt_(m: *const lapack_int, n: *const lapack_int,
                   dist: *const c_char, iseed: *mut lapack_int,
                   sym: *const c_char, d: *mut c_double, mode: *const lapack_int,
                   cond: *const c_double, dmax: *const c_double,
                   rank: *const lapack_int, kl: *const lapack_int, ku: *const lapack_int,
                   pack: *const c_char, a: *mut c_double,
                   lda: *const lapack_int, work: *mut c_double, info: *const lapack_int);

    pub fn clatmt_(m: *const lapack_int, n: *const lapack_int,
                   dist: *const c_char, iseed: *mut lapack_int,
                   sym: *const c_char, d: *mut c_float, mode: *const lapack_int,
                   cond: *const c_float, dmax: *const c_float,
                   rank: *const lapack_int, kl: *const lapack_int, ku: *const lapack_int,
                   pack: *const c_char, a: *mut lapack_complex_float,
                   lda: *const lapack_int, work: *mut lapack_complex_float, info: *const lapack_int);

    pub fn zlatmt_(m: *const lapack_int, n: *const lapack_int,
                   dist: *const c_char, iseed: *mut lapack_int,
                   sym: *const c_char, d: *mut c_double, mode: *const lapack_int,
                   cond: *const c_double, dmax: *const c_double,
                   rank: *const lapack_int, kl: *const lapack_int, ku: *const lapack_int,
                   pack: *const c_char, a: *mut lapack_complex_double,
                   lda: *const lapack_int, work: *mut lapack_complex_double, info: *const lapack_int);



// subroutine slatmt       (       integer         M,
//                 integer         N,
//                 character       DIST,
//                 integer, dimension( 4 )         ISEED,
//                 character       SYM,
//                 real, dimension( * )    D,
//                 integer         MODE,
//                 real    COND,
//                 real    DMAX,
//                 integer         RANK,
//                 integer         KL,
//                 integer         KU,
//                 character       PACK,
//                 real, dimension( lda, * )       A,
//                 integer         LDA,
//                 real, dimension( * )    WORK,
//                 integer         INFO
//         )
}
