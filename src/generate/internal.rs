use super::ffi;
use libc::c_char;
use lapack::{c32, c64};

#[inline]
#[cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))]
pub fn slaror(side: u8, init: u8, m: i32, n: i32, a: &mut [f32], lda: i32, iseed: &mut [i32],
              x: &mut [f32], info: &mut i32) {

    unsafe {
        ffi::slaror_(&(side as c_char), &(init as c_char), &m, &n, a.as_mut_ptr(), &lda,
                     iseed.as_mut_ptr(), x.as_mut_ptr(), info)
    }
}

#[inline]
#[cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))]
pub fn dlaror(side: u8, init: u8, m: i32, n: i32, a: &mut [f64], lda: i32, iseed: &mut [i32],
              x: &mut [f64], info: &mut i32) {

    unsafe {
        ffi::dlaror_(&(side as c_char), &(init as c_char), &m, &n, a.as_mut_ptr(), &lda,
                     iseed.as_mut_ptr(), x.as_mut_ptr(), info)
    }
}

#[inline]
#[cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))]
pub fn claror(side: u8, init: u8, m: i32, n: i32, a: &mut [c32], lda: i32, iseed: &mut [i32],
              x: &mut [c32], info: &mut i32) {

    unsafe {
        ffi::claror_(&(side as c_char), &(init as c_char), &m, &n, a.as_mut_ptr() as *mut _, &lda,
                     iseed.as_mut_ptr(), x.as_mut_ptr() as *mut _, info)
    }
}

#[inline]
#[cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))]
pub fn zlaror(side: u8, init: u8, m: i32, n: i32, a: &mut [c64], lda: i32, iseed: &mut [i32],
              x: &mut [c64], info: &mut i32) {

    unsafe {
        ffi::zlaror_(&(side as c_char), &(init as c_char), &m, &n, a.as_mut_ptr() as *mut _, &lda,
                     iseed.as_mut_ptr(), x.as_mut_ptr() as *mut _, info)
    }
}

#[inline]
#[cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))]
pub fn slatmt(m: i32, n: i32, dist: u8, iseed: &mut [i32],
              sym: u8, d: &mut [f32], mode: i32,
              cond: f32, dmax: f32, rank: i32, kl: i32, ku: i32,
              pack: u8, a: &mut [f32], lda: i32,
              work: &mut [f32]) -> i32 {
    let mut info = 0;
    unsafe {
        ffi::slatmt_(&m, &n, &(dist as c_char), iseed.as_mut_ptr(),
                    &(sym as c_char), d.as_mut_ptr(), &mode,
                    &cond, &dmax, &rank, &kl, &ku,
                    &(pack as c_char), a.as_mut_ptr(), &lda,
                    work.as_mut_ptr(), &mut info);
    }
    info
}

#[inline]
#[cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))]
pub fn dlatmt(m: i32, n: i32, dist: u8, iseed: &mut [i32],
              sym: u8, d: &mut [f64], mode: i32,
              cond: f64, dmax: f64, rank: i32, kl: i32, ku: i32,
              pack: u8, a: &mut [f64], lda: i32,
              work: &mut [f64]) -> i32 {
    let mut info = 0;
    unsafe {
        ffi::dlatmt_(&m, &n, &(dist as c_char), iseed.as_mut_ptr(),
                    &(sym as c_char), d.as_mut_ptr(), &mode,
                    &cond, &dmax, &rank, &kl, &ku,
                    &(pack as c_char), a.as_mut_ptr(), &lda,
                    work.as_mut_ptr(), &mut info);
    }
    info
}

#[inline]
#[cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))]
pub fn clatmt(m: i32, n: i32, dist: u8, iseed: &mut [i32],
              sym: u8, d: &mut [f32], mode: i32,
              cond: f32, dmax: f32, rank: i32, kl: i32, ku: i32,
              pack: u8, a: &mut [c32], lda: i32,
              work: &mut [c32]) -> i32 {
    let mut info = 0;
    unsafe {
        ffi::clatmt_(&m, &n, &(dist as c_char), iseed.as_mut_ptr(),
                    &(sym as c_char), d.as_mut_ptr(), &mode,
                    &cond, &dmax, &rank, &kl, &ku,
                    &(pack as c_char), a.as_mut_ptr() as *mut _, &lda,
                    work.as_mut_ptr() as *mut _, &mut info);
    }
    info
}


#[inline]
#[cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))]
pub fn zlatmt(m: i32, n: i32, dist: u8, iseed: &mut [i32],
              sym: u8, d: &mut [f64], mode: i32,
              cond: f64, dmax: f64, rank: i32, kl: i32, ku: i32,
              pack: u8, a: &mut [c64], lda: i32,
              work: &mut [c64]) -> i32 {
    let mut info = 0;
    unsafe {
        ffi::zlatmt_(&m, &n, &(dist as c_char), iseed.as_mut_ptr(),
                    &(sym as c_char), d.as_mut_ptr(), &mode,
                    &cond, &dmax, &rank, &kl, &ku,
                    &(pack as c_char), a.as_mut_ptr() as *mut _, &lda,
                    work.as_mut_ptr() as *mut _, &mut info);
    }
    info
}
