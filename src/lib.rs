//! A pure-Rust port of the reference Argon2 implementation
//! ([phc-winner-argon2](https://github.com/P-H-C/phc-winner-argon2)), with
//! runtime-dispatched SIMD backends.
//!
//! Argon2 is the winner of the 2015 Password Hashing Competition and is
//! specified in [RFC 9106](https://www.rfc-editor.org/rfc/rfc9106). Three
//! variants exist, see [`Algorithm`]:
//!
//! * `Argon2d` — data-dependent addressing: fastest, but leaks a memory access
//!   pattern that depends on the password.
//! * `Argon2i` — data-independent addressing: side-channel resistant.
//! * `Argon2id` — independent for the first half-pass, dependent afterwards.
//!   The default, and what RFC 9106 recommends.
//!
//! # Example
//!
//! ```no_run
//! use argon2_rust::{Algorithm, Argon2, Params, Version};
//!
//! // 64 MiB, 2 passes, 1 lane, 32-byte tag.
//! let params = Params::new(1 << 16, 2, 1, 32)?;
//! let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
//!
//! let mut tag = [0u8; 32];
//! argon2.hash_into(b"password", b"somesalt", &mut tag)?;
//! # Ok::<(), argon2_rust::Error>(())
//! ```
//!
//! # SIMD backends
//!
//! The compression function is selected by **runtime** CPU feature detection,
//! never by `cfg(target_feature)` alone, so one binary runs at full speed on
//! every machine. Detection happens at most once per process and the result is
//! cached; a hash call resolves one function pointer before entering its loops.
//! Ask [`detected_backend`] what this CPU picked. The cost model is documented
//! on the private `fill_block` module.
//!
//! | [`Backend`] | Requires | Source it was ported from |
//! |---|---|---|
//! | `Scalar` | — | `src/ref.c` |
//! | `Neon` | `aarch64` | `src/opt.c` (128-bit path) |
//! | `Sse2` | `x86`/`x86_64` + SSE2 | `src/opt.c` (128-bit path) |
//! | `Avx2` | `x86_64` + AVX2 | `src/opt.c` (`__AVX2__`) |
//! | `Avx512` | `x86_64` + AVX-512F | `src/opt.c` (`__AVX512F__`) |
//!
//! # Features
//!
//! * `std` *(default)* — runtime CPU feature detection.
//! * `parallel` *(default, implies `std`)* — multi-threaded fill. One
//!   [`std::thread::scope`] for the **whole** fill, whose workers meet at a
//!   barrier at each of the `4 * t_cost` algorithmic sync points, rather than a
//!   fresh scope per sync point. Note that the thread count does **not** change
//!   the tag; only [`Params::lanes`] does.
//! * `zeroize-memory` *(default)* — securely wipe internal buffers, the
//!   equivalent of `FLAG_clear_internal_memory` in the C.
//! * `bump-alloc` — adds `bumpalo` and gives `memory::Workspace` a reusable
//!   bump allocator for its small scratch buffers. **Off by default**, and the
//!   crate's only dependency of any kind. Measured worth: 17 ns per hash, which
//!   is 0.00013% of an RFC 9106 hash. Turn it on only if you have measured your
//!   own workload and want it.
//! * `internal-api` — exposes `__internal` for tests and benches. Not stable.
//!
//! The crate is `#![no_std]` and needs only `alloc`; that stays true with every
//! feature turned on. Without `std`, backend selection falls back to
//! compile-time `target_feature` cfgs.
//!
//! # Reusing memory across hashes
//!
//! Each [`Argon2`] hash acquires its block arena once and releases it on the
//! way out. A process that hashes repeatedly can instead keep a [`Hasher`],
//! from [`Argon2::hasher`], which parks the arena between calls and wipes it on
//! release, so the next call gets a zeroed arena that is **already mapped and
//! already resident**.
//!
//! That is worth far more than it used to be. Measured on Linux/x86-64,
//! interleaved against the one-shot API, 15 paired rounds:
//!
//! ```text
//!    m_cost   t   p |  one-shot |    pooled |  delta
//!   ---------|-----|-----------|-----------|--------
//!      1 MiB   1   1 |   212 us |    185 us | -11.7%
//!      4 MiB   1   4 |   806 us |    592 us | -26.9%
//!     64 MiB   1   1 |  25.89 ms|  19.43 ms | -24.9%
//!     64 MiB   1   4 |  11.65 ms|   8.40 ms | -34.0%
//!    256 MiB   1   1 | 111.74 ms|  86.17 ms | -23.3%
//!    256 MiB   1   4 |  46.14 ms|  35.09 ms | -24.0%
//! ```
//!
//! What reuse skips is the `mmap`, the first-touch page faults over the whole
//! arena, and the `munmap` — not "an allocator call", of which there was only
//! ever one and it never mattered. Below about 1 MiB a hash is mostly BLAKE2b
//! and there is little to win (-0.7% at `m_cost = 8 KiB`).
//!
//! ```
//! use argon2_rust::{Algorithm, Argon2, Hasher, Params, Version};
//!
//! // Nameable, so it can be a field, a `thread_local!` or a worker slot —
//! // which is the only shape in which reuse is worth anything.
//! struct Worker {
//!     hasher: Hasher,
//! }
//!
//! let params = Params::new(1 << 8, 1, 1, 32)?;
//! let mut worker = Worker {
//!     hasher: Argon2::new(Algorithm::Argon2id, Version::V0x13, params).hasher(),
//! };
//! let mut tag = [0u8; 32];
//! worker.hasher.hash_into(b"password", b"somesalt", &mut tag)?;
//! # Ok::<(), argon2_rust::Error>(())
//! ```
//!
//! # Panics
//!
//! Nothing reachable through the public API panics. Every failure is an
//! [`Error`], whose numeric [`Error::as_c_code`] matches the C reference.

#![no_std]
#![warn(missing_docs)]
// A `pub` item that a downstream crate cannot *name* is only half-public: it
// works in `let` bindings and nowhere else — not in a struct field, a function
// signature, a `Vec`, or a `thread_local!`. `Hasher` shipped that way once
// (`Argon2::hasher()` returned it, `lib.rs` never re-exported it), which broke
// the one shape the reuse layer exists to serve: one hasher per worker, owned
// by that worker's struct. This lint is the regression guard.
#![warn(unnameable_types)]
#![warn(clippy::undocumented_unsafe_blocks)]

// NOTE FOR EVERY CONTRIBUTOR: this crate has a module named `core`, which
// shadows the `core` crate *in this root module only*. Inside `src/lib.rs`
// always write `::core::...`. Submodules are unaffected — bare `core::` there
// still means the `core` crate.

extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod error;
pub mod params;

// These modules are private, and a good deal of what they expose escapes the
// crate only through `__internal` (below), which tests and benches enable. In a
// plain build those items are legitimately unreachable, so `dead_code` would
// fire on all of them.
//
// Rather than blanket-allowing `dead_code` — which would also hide code that is
// dead by mistake — the allow is tied to `internal-api` being OFF. With the
// feature ON, `__internal` re-exports the intended surface, so anything the
// compiler still calls dead really is dead and gets reported.
macro_rules! private_modules {
    ($($name:ident),* $(,)?) => {
        $(
            #[cfg_attr(not(feature = "internal-api"), allow(dead_code))]
            mod $name;
        )*
    };
}

private_modules!(blake2b, block, core, encoding, fill_block, memory);

pub use crate::core::{Argon2, Hasher};
pub use crate::error::Error;
pub use crate::fill_block::Backend;
pub use crate::params::{Algorithm, Params, Version};

/// The [`Backend`] this CPU resolved to, cached after the first call.
///
/// Diagnostic only — the hashing entry points call this for you.
///
/// ```
/// println!("argon2 backend: {}", argon2_rust::detected_backend());
/// ```
#[inline]
#[must_use]
pub fn detected_backend() -> Backend {
    crate::fill_block::backend()
}

/// Unstable internals, exposed for this crate's own tests and benches.
///
/// Gated behind the non-default `internal-api` feature. **No stability
/// guarantees**: anything here can change in a patch release.
///
/// # Soundness
///
/// Unstable is not the same as unsound. Every entry point here that takes an
/// explicit [`Backend`] — `fill_memory_blocks_traced`, `hash_traced`,
/// `hash_with_backend` — is an `unsafe fn`, and so is each backend's
/// `fill_segment`. They dispatch to a `#[target_feature(enable = ...)]`
/// function, so running one whose feature this CPU lacks is undefined behaviour
/// (`SIGILL` in practice), and only the caller can rule that out:
/// [`Backend::is_available`] is the portable way.
///
/// The safe entry points — [`Argon2`], [`detected_backend`],
/// `fill_memory_blocks` — never let a caller name the backend. They take it from
/// `fill_block::backend()`, the cached runtime cascade, which by construction
/// only ever names a backend this CPU advertises. That is the whole reason they
/// can be safe, and it is why turning on `internal-api` cannot make a
/// `#![forbid(unsafe_code)]` program reachable by UB.
#[cfg(feature = "internal-api")]
#[doc(hidden)]
pub mod __internal {
    pub use crate::blake2b::{
        BLOCKBYTES, Blake2b, IV, KEYBYTES, OUTBYTES, PERSONALBYTES, SALTBYTES, blake2b,
        blake2b_long,
    };
    pub use crate::block::{Block, Instance, Position};
    pub use crate::core::{
        PassTrace, constant_time_eq, fill_first_blocks, fill_memory_blocks,
        fill_memory_blocks_traced, finalize, hash_traced, hash_with_backend, index_alpha,
        initial_hash,
    };
    pub use crate::encoding::{
        Decoded, b64_len, decode_string, encode_string, encode_string_alloc, encoded_len,
        from_base64, num_len, to_base64,
    };
    pub use crate::fill_block::{Backend, FillSegmentFn, backend, detect, fill_segment_fn};
    pub use crate::memory::{
        ARENA_ALIGN, Arena, ArenaGuard, Workspace, clear_internal_memory,
        clear_internal_memory_blocks, clear_internal_memory_u64, secure_wipe, secure_wipe_blocks,
        secure_wipe_raw, secure_wipe_u64,
    };
    /// The arena release-path observation point, for
    /// `tests/allocation_audit.rs`. See [`crate::memory::audit`].
    #[cfg(feature = "std")]
    pub use crate::memory::audit;
    pub use crate::params::validate_inputs;

    /// `bumpalo`, re-exported so callers can name the types
    /// [`Workspace::bump`](crate::memory::Workspace::bump) hands back without
    /// having to match this crate's exact dependency version.
    ///
    /// Only the `try_alloc_*` family is admissible: the infallible `alloc_*`
    /// methods abort on allocation failure, and nothing reachable from this
    /// crate's safe API is allowed to do that.
    #[cfg(feature = "bump-alloc")]
    pub use ::bumpalo;

    /// Each backend's `fill_segment`, reachable directly so a differential test
    /// can pit two backends against each other on the same arena.
    pub mod backends {
        pub use crate::fill_block::scalar;

        #[cfg(target_arch = "aarch64")]
        pub use crate::fill_block::neon;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        pub use crate::fill_block::sse2;

        #[cfg(target_arch = "x86_64")]
        pub use crate::fill_block::avx2;

        #[cfg(target_arch = "x86_64")]
        pub use crate::fill_block::avx512;
    }
}
