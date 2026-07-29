//! Error codes, mirroring `argon2_error_codes` from `phc-winner-argon2/include/argon2.h`.
//!
//! Every variant carries the *exact* numeric value the C reference uses, so
//! differential tests can compare `Error::as_c_code()` against the `int`
//! returned by `argon2_ctx` / `argon2_hash` / `argon2_verify`.
//!
//! `ARGON2_OK` (0) is deliberately absent: success is `Ok(())` in Rust.

/// An Argon2 error.
///
/// The discriminants are the C error codes, so `error as i32` is the C value
/// (see [`Error::as_c_code`]).
///
/// Some variants are unreachable through this crate's API but are kept so the
/// mapping from C codes is total (see [`Error::from_c_code`]):
///
/// * [`Error::OutputPtrNull`], [`Error::PwdPtrMismatch`],
///   [`Error::SaltPtrMismatch`], [`Error::SecretPtrMismatch`],
///   [`Error::AdPtrMismatch`], [`Error::OutPtrMismatch`] — Rust uses slices,
///   there are no null pointers with a non-zero length.
/// * [`Error::FreeMemoryCbkNull`], [`Error::AllocateMemoryCbkNull`],
///   [`Error::MissingArgs`] — this crate has no allocator callbacks.
/// * [`Error::IncorrectType`] — [`crate::Algorithm`] is a closed enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i32)]
pub enum Error {
    /// `ARGON2_OUTPUT_PTR_NULL` (-1). Unreachable in Rust.
    OutputPtrNull = -1,
    /// `ARGON2_OUTPUT_TOO_SHORT` (-2). Output shorter than [`crate::params::MIN_OUTLEN`].
    OutputTooShort = -2,
    /// `ARGON2_OUTPUT_TOO_LONG` (-3). Output longer than [`crate::params::MAX_OUTLEN`].
    OutputTooLong = -3,
    /// `ARGON2_PWD_TOO_SHORT` (-4). Unreachable: `MIN_PWD_LENGTH` is 0.
    PwdTooShort = -4,
    /// `ARGON2_PWD_TOO_LONG` (-5).
    PwdTooLong = -5,
    /// `ARGON2_SALT_TOO_SHORT` (-6). Salt shorter than [`crate::params::MIN_SALT_LENGTH`].
    SaltTooShort = -6,
    /// `ARGON2_SALT_TOO_LONG` (-7).
    SaltTooLong = -7,
    /// `ARGON2_AD_TOO_SHORT` (-8). Unreachable: `MIN_AD_LENGTH` is 0.
    AdTooShort = -8,
    /// `ARGON2_AD_TOO_LONG` (-9).
    AdTooLong = -9,
    /// `ARGON2_SECRET_TOO_SHORT` (-10). Unreachable: `MIN_SECRET` is 0.
    SecretTooShort = -10,
    /// `ARGON2_SECRET_TOO_LONG` (-11).
    SecretTooLong = -11,
    /// `ARGON2_TIME_TOO_SMALL` (-12).
    TimeTooSmall = -12,
    /// `ARGON2_TIME_TOO_LARGE` (-13).
    TimeTooLarge = -13,
    /// `ARGON2_MEMORY_TOO_LITTLE` (-14).
    MemoryTooLittle = -14,
    /// `ARGON2_MEMORY_TOO_MUCH` (-15).
    MemoryTooMuch = -15,
    /// `ARGON2_LANES_TOO_FEW` (-16).
    LanesTooFew = -16,
    /// `ARGON2_LANES_TOO_MANY` (-17).
    LanesTooMany = -17,
    /// `ARGON2_PWD_PTR_MISMATCH` (-18). Unreachable in Rust.
    PwdPtrMismatch = -18,
    /// `ARGON2_SALT_PTR_MISMATCH` (-19). Unreachable in Rust.
    SaltPtrMismatch = -19,
    /// `ARGON2_SECRET_PTR_MISMATCH` (-20). Unreachable in Rust.
    SecretPtrMismatch = -20,
    /// `ARGON2_AD_PTR_MISMATCH` (-21). Unreachable in Rust.
    AdPtrMismatch = -21,
    /// `ARGON2_MEMORY_ALLOCATION_ERROR` (-22). The block arena could not be allocated.
    MemoryAllocationError = -22,
    /// `ARGON2_FREE_MEMORY_CBK_NULL` (-23). Unreachable: no allocator callbacks.
    FreeMemoryCbkNull = -23,
    /// `ARGON2_ALLOCATE_MEMORY_CBK_NULL` (-24). Unreachable: no allocator callbacks.
    AllocateMemoryCbkNull = -24,
    /// `ARGON2_INCORRECT_PARAMETER` (-25).
    IncorrectParameter = -25,
    /// `ARGON2_INCORRECT_TYPE` (-26). Unreachable: [`crate::Algorithm`] is a closed enum.
    IncorrectType = -26,
    /// `ARGON2_OUT_PTR_MISMATCH` (-27). Unreachable in Rust.
    OutPtrMismatch = -27,
    /// `ARGON2_THREADS_TOO_FEW` (-28).
    ThreadsTooFew = -28,
    /// `ARGON2_THREADS_TOO_MANY` (-29).
    ThreadsTooMany = -29,
    /// `ARGON2_MISSING_ARGS` (-30). Unreachable in Rust.
    MissingArgs = -30,
    /// `ARGON2_ENCODING_FAIL` (-31). The PHC string did not fit the output buffer.
    EncodingFail = -31,
    /// `ARGON2_DECODING_FAIL` (-32). The PHC string is malformed.
    DecodingFail = -32,
    /// `ARGON2_THREAD_FAIL` (-33). A worker thread could not be started or panicked.
    ThreadFail = -33,
    /// `ARGON2_DECODING_LENGTH_FAIL` (-34).
    DecodingLengthFail = -34,
    /// `ARGON2_VERIFY_MISMATCH` (-35). The password does not match the hash.
    VerifyMismatch = -35,
}

impl Error {
    /// The lowest C error code (`ARGON2_VERIFY_MISMATCH`).
    pub const MIN_C_CODE: i32 = -35;
    /// The highest non-OK C error code (`ARGON2_OUTPUT_PTR_NULL`).
    pub const MAX_C_CODE: i32 = -1;

    /// The exact numeric code the C reference returns for this condition.
    #[inline]
    #[must_use]
    pub const fn as_c_code(&self) -> i32 {
        *self as i32
    }

    /// Inverse of [`Error::as_c_code`].
    ///
    /// Returns `None` for `ARGON2_OK` (0) and for any code outside
    /// `-35..=-1`, so differential tests can turn a C return value into a
    /// `Result<(), Error>`.
    #[must_use]
    pub const fn from_c_code(code: i32) -> Option<Error> {
        Some(match code {
            -1 => Error::OutputPtrNull,
            -2 => Error::OutputTooShort,
            -3 => Error::OutputTooLong,
            -4 => Error::PwdTooShort,
            -5 => Error::PwdTooLong,
            -6 => Error::SaltTooShort,
            -7 => Error::SaltTooLong,
            -8 => Error::AdTooShort,
            -9 => Error::AdTooLong,
            -10 => Error::SecretTooShort,
            -11 => Error::SecretTooLong,
            -12 => Error::TimeTooSmall,
            -13 => Error::TimeTooLarge,
            -14 => Error::MemoryTooLittle,
            -15 => Error::MemoryTooMuch,
            -16 => Error::LanesTooFew,
            -17 => Error::LanesTooMany,
            -18 => Error::PwdPtrMismatch,
            -19 => Error::SaltPtrMismatch,
            -20 => Error::SecretPtrMismatch,
            -21 => Error::AdPtrMismatch,
            -22 => Error::MemoryAllocationError,
            -23 => Error::FreeMemoryCbkNull,
            -24 => Error::AllocateMemoryCbkNull,
            -25 => Error::IncorrectParameter,
            -26 => Error::IncorrectType,
            -27 => Error::OutPtrMismatch,
            -28 => Error::ThreadsTooFew,
            -29 => Error::ThreadsTooMany,
            -30 => Error::MissingArgs,
            -31 => Error::EncodingFail,
            -32 => Error::DecodingFail,
            -33 => Error::ThreadFail,
            -34 => Error::DecodingLengthFail,
            -35 => Error::VerifyMismatch,
            _ => return None,
        })
    }

    /// The message `argon2_error_message()` returns for this code, verbatim.
    ///
    /// Kept byte-identical to `src/argon2.c` so differential tests can compare
    /// the strings too.
    #[must_use]
    pub const fn message(&self) -> &'static str {
        match self {
            Error::OutputPtrNull => "Output pointer is NULL",
            Error::OutputTooShort => "Output is too short",
            Error::OutputTooLong => "Output is too long",
            Error::PwdTooShort => "Password is too short",
            Error::PwdTooLong => "Password is too long",
            Error::SaltTooShort => "Salt is too short",
            Error::SaltTooLong => "Salt is too long",
            Error::AdTooShort => "Associated data is too short",
            Error::AdTooLong => "Associated data is too long",
            Error::SecretTooShort => "Secret is too short",
            Error::SecretTooLong => "Secret is too long",
            Error::TimeTooSmall => "Time cost is too small",
            Error::TimeTooLarge => "Time cost is too large",
            Error::MemoryTooLittle => "Memory cost is too small",
            Error::MemoryTooMuch => "Memory cost is too large",
            Error::LanesTooFew => "Too few lanes",
            Error::LanesTooMany => "Too many lanes",
            Error::PwdPtrMismatch => "Password pointer is NULL, but password length is not 0",
            Error::SaltPtrMismatch => "Salt pointer is NULL, but salt length is not 0",
            Error::SecretPtrMismatch => "Secret pointer is NULL, but secret length is not 0",
            Error::AdPtrMismatch => "Associated data pointer is NULL, but ad length is not 0",
            Error::MemoryAllocationError => "Memory allocation error",
            Error::FreeMemoryCbkNull => "The free memory callback is NULL",
            Error::AllocateMemoryCbkNull => "The allocate memory callback is NULL",
            Error::IncorrectParameter => "Argon2_Context context is NULL",
            Error::IncorrectType => "There is no such version of Argon2",
            Error::OutPtrMismatch => "Output pointer mismatch",
            Error::ThreadsTooFew => "Not enough threads",
            Error::ThreadsTooMany => "Too many threads",
            Error::MissingArgs => "Missing arguments",
            Error::EncodingFail => "Encoding failed",
            Error::DecodingFail => "Decoding failed",
            Error::ThreadFail => "Threading failure",
            Error::DecodingLengthFail => "Some of encoded parameters are too long or too short",
            Error::VerifyMismatch => "The password does not match the supplied hash",
        }
    }
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.message())
    }
}

impl core::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::Error;

    #[test]
    fn discriminants_match_c() {
        assert_eq!(Error::OutputPtrNull.as_c_code(), -1);
        assert_eq!(Error::VerifyMismatch.as_c_code(), -35);
        assert_eq!(Error::MemoryTooLittle.as_c_code(), -14);
    }

    #[test]
    fn round_trip_every_code() {
        for code in Error::MIN_C_CODE..=Error::MAX_C_CODE {
            let e = Error::from_c_code(code).expect("every code in range maps");
            assert_eq!(e.as_c_code(), code);
        }
        assert!(Error::from_c_code(0).is_none());
        assert!(Error::from_c_code(-36).is_none());
        assert!(Error::from_c_code(1).is_none());
    }
}
