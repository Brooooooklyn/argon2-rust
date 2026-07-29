//! Backend selection: one `fill_segment` implementation per instruction set,
//! chosen by **runtime** CPU feature detection.
//!
//! # Cost model
//!
//! * Detection runs at most once per process. The result is cached in a
//!   [`AtomicU8`] read with [`Ordering::Relaxed`]. The initialisation race is
//!   benign: every thread computes the same answer, so a duplicated `detect()`
//!   only wastes a `cpuid`.
//! * A hash call resolves the function pointer **once**, before entering the
//!   pass/slice/lane loops (see `core::fill_memory_blocks`). Nothing detects or
//!   dispatches inside the per-block loop.
//! * Per hash: one relaxed load plus one compare. Per segment: one indirect
//!   call. A segment is `segment_length` blocks — thousands at any realistic
//!   `m_cost` — so the per-block overhead is nil.
//!
//! # Why `#[target_feature]` sits on `fill_segment`
//!
//! LLVM will not inline a callee with a higher target-feature set into a caller
//! with a lower one. Putting the attribute on the whole `fill_segment` lets
//! `fill_block` inline into it, which is what preserves the `src/opt.c`
//! optimisation of keeping the 1 KiB `state` in registers across loop
//! iterations instead of reloading it from memory for every block. Do **not**
//! move the attribute down onto `fill_block` or onto individual intrinsics.
//!
//! # `no_std`
//!
//! Runtime detection needs `std`. Without the `std` feature, [`detect`] falls
//! back to compile-time `cfg(target_feature = ...)` and then to
//! [`Backend::Scalar`].
//!
//! # Testing under Rosetta on aarch64-apple-darwin
//!
//! `is_x86_feature_detected!` expands to
//! `cfg!(target_feature = "...") || runtime_cpuid_check()`, so a compile-time
//! `target_feature` short-circuits it to `true`. That matters here because
//! Rosetta 2 (measured on macOS 26.5.2 / Apple M5 Max) *executes* AVX2 but does
//! not advertise it in `cpuid`:
//!
//! * `cargo test --target x86_64-apple-darwin` — [`detect`] returns
//!   [`Backend::Sse2`] and `Backend::Avx2.is_available()` is `false`, so AVX2
//!   gets skipped.
//! * `RUSTFLAGS="-C target-feature=+avx2" cargo test --target x86_64-apple-darwin`
//!   — [`detect`] returns [`Backend::Avx2`] and it really runs.
//! * Never add `+avx512f`: `is_available()` would then report `true` while the
//!   instruction itself traps with `SIGILL`.

use core::sync::atomic::{AtomicU8, Ordering};

use crate::block::{Instance, Position};

pub mod scalar;

#[cfg(target_arch = "aarch64")]
pub mod neon;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod sse2;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "x86_64")]
pub mod avx512;

// ---------------------------------------------------------------------------
// Backend
// ---------------------------------------------------------------------------

/// Which `fill_segment` implementation to use.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Backend {
    /// Portable scalar code. Always available.
    Scalar = 0,
    /// AArch64 NEON.
    Neon = 1,
    /// x86 / x86-64 SSE2.
    Sse2 = 2,
    /// x86-64 AVX2.
    Avx2 = 3,
    /// x86-64 AVX-512F.
    Avx512 = 4,
}

/// The signature every backend's `fill_segment` has.
///
/// # Safety
///
/// Calling one of these requires:
///
/// * the CPU to support the backend's instruction set — check
///   [`Backend::is_available`], or get the pointer from [`backend`];
/// * `instance`'s arena pointer to be valid for `instance.memory_len()` blocks;
/// * `position` to be in range, and no other thread to be writing the segment
///   `(position.lane, position.slice)` at the same time.
pub type FillSegmentFn = unsafe fn(&Instance, Position);

impl Backend {
    /// Every backend, in ascending preference order.
    pub const ALL: [Backend; 5] = [
        Backend::Scalar,
        Backend::Neon,
        Backend::Sse2,
        Backend::Avx2,
        Backend::Avx512,
    ];

    /// Short lowercase name, handy for bench ids and test output.
    #[inline]
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Backend::Scalar => "scalar",
            Backend::Neon => "neon",
            Backend::Sse2 => "sse2",
            Backend::Avx2 => "avx2",
            Backend::Avx512 => "avx512",
        }
    }

    /// Whether this CPU can execute this backend *right now*.
    ///
    /// Unlike `detect`, this asks about one specific backend, so tests and
    /// benches can loop over [`Backend::ALL`] and skip what the host cannot run.
    #[inline]
    #[must_use]
    pub fn is_available(self) -> bool {
        match self {
            Backend::Scalar => true,
            Backend::Neon => have_neon(),
            Backend::Sse2 => have_sse2(),
            Backend::Avx2 => have_avx2(),
            Backend::Avx512 => have_avx512f(),
        }
    }

    #[inline]
    const fn to_u8(self) -> u8 {
        self as u8
    }

    /// Total inverse of [`Backend::to_u8`]; anything unknown maps to
    /// [`Backend::Scalar`] so the cache can never produce a panic.
    #[inline]
    const fn from_u8(value: u8) -> Backend {
        match value {
            1 => Backend::Neon,
            2 => Backend::Sse2,
            3 => Backend::Avx2,
            4 => Backend::Avx512,
            _ => Backend::Scalar,
        }
    }
}

impl core::fmt::Display for Backend {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

// ---------------------------------------------------------------------------
// Per-feature probes
// ---------------------------------------------------------------------------
//
// Each probe is defined twice under mutually exclusive `cfg`s, so exactly one
// definition ever exists and there is no dead or unreachable code. With `std`
// the probe is a real runtime check; without it, it degrades to the
// compile-time `target_feature` cfg.

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[inline]
fn have_avx512f() -> bool {
    std::arch::is_x86_feature_detected!("avx512f")
}
#[cfg(not(all(feature = "std", target_arch = "x86_64")))]
#[inline]
fn have_avx512f() -> bool {
    cfg!(all(target_arch = "x86_64", target_feature = "avx512f"))
}

#[cfg(all(feature = "std", target_arch = "x86_64"))]
#[inline]
fn have_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}
#[cfg(not(all(feature = "std", target_arch = "x86_64")))]
#[inline]
fn have_avx2() -> bool {
    cfg!(all(target_arch = "x86_64", target_feature = "avx2"))
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
fn have_sse2() -> bool {
    std::arch::is_x86_feature_detected!("sse2")
}
#[cfg(not(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64"))))]
#[inline]
fn have_sse2() -> bool {
    cfg!(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse2"
    ))
}

#[cfg(all(feature = "std", target_arch = "aarch64"))]
#[inline]
fn have_neon() -> bool {
    std::arch::is_aarch64_feature_detected!("neon")
}
#[cfg(not(all(feature = "std", target_arch = "aarch64")))]
#[inline]
fn have_neon() -> bool {
    cfg!(all(target_arch = "aarch64", target_feature = "neon"))
}

// ---------------------------------------------------------------------------
// Detection and caching
// ---------------------------------------------------------------------------

/// Sentinel meaning "detection has not run yet". Not a valid [`Backend`] value.
const UNINIT: u8 = 0xFF;

/// Cached [`Backend`] as a `u8`, or [`UNINIT`].
static CACHED_BACKEND: AtomicU8 = AtomicU8::new(UNINIT);

/// Run the feature cascade and return the best backend for this CPU.
///
/// Preference order: `Avx512 > Avx2 > Sse2` on x86-64, `Sse2` on x86, `Neon` on
/// AArch64, `Scalar` everywhere else. The probes are arch-gated, so the single
/// cascade below cannot pick an off-arch backend.
///
/// This always re-runs detection; use [`backend`] for the cached value.
#[must_use]
pub fn detect() -> Backend {
    if have_avx512f() {
        Backend::Avx512
    } else if have_avx2() {
        Backend::Avx2
    } else if have_sse2() {
        Backend::Sse2
    } else if have_neon() {
        Backend::Neon
    } else {
        Backend::Scalar
    }
}

/// Detect and populate the cache. Outlined so [`backend`] stays tiny.
#[cold]
#[inline(never)]
fn detect_and_cache() -> Backend {
    let detected = detect();
    // Relaxed is enough: the value is a plain `u8` with no associated data, and
    // every thread that races here computes the same answer.
    CACHED_BACKEND.store(detected.to_u8(), Ordering::Relaxed);
    detected
}

/// The backend for this process: one relaxed atomic load on the hot path.
///
/// Deliberately not a `OnceLock` — acquire ordering would buy nothing here
/// (there is no data to publish) and `OnceLock` needs `std`.
#[inline]
#[must_use]
pub fn backend() -> Backend {
    let cached = CACHED_BACKEND.load(Ordering::Relaxed);
    if cached == UNINIT {
        detect_and_cache()
    } else {
        Backend::from_u8(cached)
    }
}

/// The `fill_segment` implementation for `backend`.
///
/// Resolve this **once per hash call**, outside every loop.
///
/// On an architecture that has no module for the requested backend, this
/// returns the scalar implementation rather than failing to compile, so tests
/// can iterate over [`Backend::ALL`] on any host. It does **not** check
/// availability: a pointer for a backend this CPU lacks will fault when called.
/// Use [`Backend::is_available`] first, or take the value from [`backend`].
#[must_use]
pub fn fill_segment_fn(backend: Backend) -> FillSegmentFn {
    match backend {
        Backend::Scalar => scalar::fill_segment,

        #[cfg(target_arch = "aarch64")]
        Backend::Neon => neon::fill_segment,
        #[cfg(not(target_arch = "aarch64"))]
        Backend::Neon => scalar::fill_segment,

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Backend::Sse2 => sse2::fill_segment,
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        Backend::Sse2 => scalar::fill_segment,

        #[cfg(target_arch = "x86_64")]
        Backend::Avx2 => avx2::fill_segment,
        #[cfg(not(target_arch = "x86_64"))]
        Backend::Avx2 => scalar::fill_segment,

        #[cfg(target_arch = "x86_64")]
        Backend::Avx512 => avx512::fill_segment,
        #[cfg(not(target_arch = "x86_64"))]
        Backend::Avx512 => scalar::fill_segment,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_u8_round_trip() {
        for b in Backend::ALL {
            assert_eq!(Backend::from_u8(b.to_u8()), b);
        }
        // The sentinel and anything unknown must degrade, never panic.
        assert_eq!(Backend::from_u8(UNINIT), Backend::Scalar);
        assert_eq!(Backend::from_u8(200), Backend::Scalar);
    }

    #[test]
    fn cache_agrees_with_detect() {
        let first = backend();
        assert_eq!(first, detect());
        // Second call takes the cached path.
        assert_eq!(backend(), first);
        assert_ne!(CACHED_BACKEND.load(Ordering::Relaxed), UNINIT);
    }

    #[test]
    fn detected_backend_is_available() {
        assert!(detect().is_available());
        assert!(Backend::Scalar.is_available());
    }

    #[test]
    fn detection_respects_the_architecture() {
        if cfg!(target_arch = "aarch64") {
            assert!(!have_sse2());
            assert!(!have_avx2());
            assert!(!have_avx512f());
            assert_eq!(detect(), Backend::Neon);
        }
        if cfg!(target_arch = "x86_64") {
            // SSE2 is baseline on x86-64.
            assert!(have_sse2());
            assert!(!have_neon());
            assert!(matches!(
                detect(),
                Backend::Sse2 | Backend::Avx2 | Backend::Avx512
            ));
        }
    }

    #[test]
    fn every_backend_resolves_to_a_function() {
        for b in Backend::ALL {
            let f = fill_segment_fn(b);
            // Compare as raw addresses; only Scalar is guaranteed to be itself.
            let scalar = fill_segment_fn(Backend::Scalar);
            if b == Backend::Scalar {
                assert!(core::ptr::fn_addr_eq(f, scalar));
            }
        }
    }
}
