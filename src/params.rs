//! Limits, [`Algorithm`], [`Version`] and [`Params`].
//!
//! Every constant here is transcribed from `phc-winner-argon2/include/argon2.h`
//! and `phc-winner-argon2/src/core.h`. [`validate_inputs`] reproduces
//! `validate_inputs()` from `src/core.c` **in the same order**, because the
//! order decides which error code surfaces first.

use crate::error::Error;

// ---------------------------------------------------------------------------
// Limits from include/argon2.h
// ---------------------------------------------------------------------------

/// `ARGON2_MIN_LANES`.
pub const MIN_LANES: u32 = 1;
/// `ARGON2_MAX_LANES`.
pub const MAX_LANES: u32 = 0x00FF_FFFF;

/// `ARGON2_MIN_THREADS`.
pub const MIN_THREADS: u32 = 1;
/// `ARGON2_MAX_THREADS`.
pub const MAX_THREADS: u32 = 0x00FF_FFFF;

/// `ARGON2_SYNC_POINTS`: synchronisation points between lanes per pass.
pub const SYNC_POINTS: u32 = 4;

/// `ARGON2_MIN_OUTLEN`.
pub const MIN_OUTLEN: u32 = 4;
/// `ARGON2_MAX_OUTLEN`.
pub const MAX_OUTLEN: u32 = 0xFFFF_FFFF;

/// `ARGON2_MIN_MEMORY` = `2 * ARGON2_SYNC_POINTS` (two blocks per slice).
pub const MIN_MEMORY: u32 = 2 * SYNC_POINTS;

/// `ARGON2_MAX_MEMORY_BITS` = `min(32, sizeof(void*) * CHAR_BIT - 10 - 1)`.
///
/// 32 on a 64-bit target, 21 on a 32-bit target.
pub const MAX_MEMORY_BITS: u32 = {
    let ptr_bits = (size_of::<*const u8>() * 8) as u32;
    let bits = ptr_bits - 10 - 1;
    if bits < 32 { bits } else { 32 }
};

/// `ARGON2_MAX_MEMORY` = `min(0xFFFFFFFF, 1 << ARGON2_MAX_MEMORY_BITS)`.
///
/// `0xFFFF_FFFF` on a 64-bit target, `0x0020_0000` on a 32-bit target.
/// Verified against the C preprocessor on `aarch64-apple-darwin`.
pub const MAX_MEMORY: u32 = {
    let candidate: u64 = 1u64 << MAX_MEMORY_BITS;
    if candidate < 0xFFFF_FFFF {
        candidate as u32
    } else {
        0xFFFF_FFFF
    }
};

/// `ARGON2_MIN_TIME`.
pub const MIN_TIME: u32 = 1;
/// `ARGON2_MAX_TIME`.
pub const MAX_TIME: u32 = 0xFFFF_FFFF;

/// `ARGON2_MIN_PWD_LENGTH`.
pub const MIN_PWD_LENGTH: u32 = 0;
/// `ARGON2_MAX_PWD_LENGTH`.
pub const MAX_PWD_LENGTH: u32 = 0xFFFF_FFFF;

/// `ARGON2_MIN_AD_LENGTH`.
pub const MIN_AD_LENGTH: u32 = 0;
/// `ARGON2_MAX_AD_LENGTH`.
pub const MAX_AD_LENGTH: u32 = 0xFFFF_FFFF;

/// `ARGON2_MIN_SALT_LENGTH`.
pub const MIN_SALT_LENGTH: u32 = 8;
/// `ARGON2_MAX_SALT_LENGTH`.
pub const MAX_SALT_LENGTH: u32 = 0xFFFF_FFFF;

/// `ARGON2_MIN_SECRET`.
pub const MIN_SECRET: u32 = 0;
/// `ARGON2_MAX_SECRET`.
pub const MAX_SECRET: u32 = 0xFFFF_FFFF;

// ---------------------------------------------------------------------------
// Internal constants from src/core.h
// ---------------------------------------------------------------------------

/// `ARGON2_BLOCK_SIZE`: memory block size in bytes.
pub const BLOCK_SIZE: usize = 1024;
/// `ARGON2_QWORDS_IN_BLOCK`: 64-bit words per block.
pub const QWORDS_IN_BLOCK: usize = BLOCK_SIZE / 8;
/// `ARGON2_OWORDS_IN_BLOCK`: 128-bit lanes per block (SSE2).
pub const OWORDS_IN_BLOCK: usize = BLOCK_SIZE / 16;
/// `ARGON2_HWORDS_IN_BLOCK`: 256-bit lanes per block (AVX2).
pub const HWORDS_IN_BLOCK: usize = BLOCK_SIZE / 32;
/// `ARGON2_512BIT_WORDS_IN_BLOCK`: 512-bit lanes per block (AVX-512).
pub const BITS512_WORDS_IN_BLOCK: usize = BLOCK_SIZE / 64;

/// `ARGON2_ADDRESSES_IN_BLOCK`: pseudo-random values one address block holds.
pub const ADDRESSES_IN_BLOCK: usize = 128;

/// `ARGON2_PREHASH_DIGEST_LENGTH`: length of `H0`.
pub const PREHASH_DIGEST_LENGTH: usize = 64;
/// `ARGON2_PREHASH_SEED_LENGTH`: `H0` plus the 4-byte block index and 4-byte lane index.
pub const PREHASH_SEED_LENGTH: usize = 72;

// ---------------------------------------------------------------------------
// Limits from src/encoding.h
// ---------------------------------------------------------------------------
//
// Mirrored for completeness, and unused — the C defines all three in
// `encoding.h:22-24` and then never reads them, so `decode_string` here does
// not either. Keeping them (rather than dropping them) is what makes the
// header-for-header correspondence with the C checkable; do not add a use for
// them without checking the C grew one first.

/// `ARGON2_MAX_DECODED_LANES`.
pub const MAX_DECODED_LANES: u32 = 255;
/// `ARGON2_MIN_DECODED_SALT_LEN`.
pub const MIN_DECODED_SALT_LEN: u32 = 8;
/// `ARGON2_MIN_DECODED_OUT_LEN`.
pub const MIN_DECODED_OUT_LEN: u32 = 12;

// ---------------------------------------------------------------------------
// Algorithm
// ---------------------------------------------------------------------------

/// The Argon2 primitive type (`argon2_type`).
///
/// The numeric values matter: `initial_hash` hashes them, and `fill_segment`
/// puts `instance->type` into `input_block.v[5]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(u32)]
pub enum Algorithm {
    /// `Argon2_d` (0): data-dependent addressing.
    Argon2d = 0,
    /// `Argon2_i` (1): data-independent addressing.
    Argon2i = 1,
    /// `Argon2_id` (2): first half-pass independent, rest dependent. The default.
    #[default]
    Argon2id = 2,
}

impl Algorithm {
    /// The `argon2_type` numeric value.
    #[inline]
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// Parse an `argon2_type` numeric value.
    #[inline]
    #[must_use]
    pub const fn from_u32(value: u32) -> Option<Algorithm> {
        match value {
            0 => Some(Algorithm::Argon2d),
            1 => Some(Algorithm::Argon2i),
            2 => Some(Algorithm::Argon2id),
            _ => None,
        }
    }

    /// `argon2_type2string(type, 0)`: the lowercase name used in PHC strings.
    ///
    /// Note `"argon2i"` is a prefix of `"argon2id"`; the C decoder relies on
    /// the *next* character failing to parse, and the Rust decoder must too.
    #[inline]
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Algorithm::Argon2d => "argon2d",
            Algorithm::Argon2i => "argon2i",
            Algorithm::Argon2id => "argon2id",
        }
    }

    /// `argon2_type2string(type, 1)`: the capitalised name (used by genkat).
    #[inline]
    #[must_use]
    pub const fn as_str_uppercase(self) -> &'static str {
        match self {
            Algorithm::Argon2d => "Argon2d",
            Algorithm::Argon2i => "Argon2i",
            Algorithm::Argon2id => "Argon2id",
        }
    }

    /// All three variants, in `argon2_type` order.
    pub const ALL: [Algorithm; 3] = [Algorithm::Argon2d, Algorithm::Argon2i, Algorithm::Argon2id];
}

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------

/// The Argon2 version (`argon2_version`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(u32)]
pub enum Version {
    /// `ARGON2_VERSION_10` (0x10). Blocks are always overwritten, never XORed.
    V0x10 = 0x10,
    /// `ARGON2_VERSION_13` (0x13) — `ARGON2_VERSION_NUMBER`, the default.
    #[default]
    V0x13 = 0x13,
}

impl Version {
    /// `ARGON2_VERSION_NUMBER`.
    pub const DEFAULT: Version = Version::V0x13;

    /// Both variants, ascending.
    pub const ALL: [Version; 2] = [Version::V0x10, Version::V0x13];

    /// The `argon2_version` numeric value.
    #[inline]
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self as u32
    }

    /// Parse an `argon2_version` numeric value.
    #[inline]
    #[must_use]
    pub const fn from_u32(value: u32) -> Option<Version> {
        match value {
            0x10 => Some(Version::V0x10),
            0x13 => Some(Version::V0x13),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// validate_inputs
// ---------------------------------------------------------------------------

/// `validate_inputs()` from `src/core.c`, in the exact same order.
///
/// The order is load-bearing: when several inputs are invalid, the C reference
/// returns the error for whichever check runs first, and the differential tests
/// compare error codes.
///
/// Checks the C performs that are omitted here, with the reason:
///
/// * `context == NULL` → `ARGON2_INCORRECT_PARAMETER`: no null contexts in Rust.
/// * `out == NULL` → `ARGON2_OUTPUT_PTR_NULL`: no null slices in Rust.
/// * the four `*_PTR_MISMATCH` checks: a Rust slice always has a valid pointer.
/// * `ARGON2_MIN_PWD_LENGTH > pwdlen`, `ARGON2_MIN_AD_LENGTH > adlen`,
///   `ARGON2_MIN_SECRET > secretlen`: those minima are all 0, so the checks can
///   never fire (and would be tautological comparisons in Rust).
/// * the two allocator-callback checks: this crate has no allocator callbacks.
///
/// Note the C computes `8 * context->lanes` in `uint32_t`, *before* `lanes` has
/// been range-checked, so it can wrap. [`u32::wrapping_mul`] reproduces that:
/// `lanes = 0xFFFF_FFFF` yields `MemoryTooLittle`, not `LanesTooMany`.
// `MAX_TIME` is `u32::MAX`, and so is `MAX_MEMORY` on a 64-bit target, which
// makes those two upper-bound checks tautologically false there. They are kept
// verbatim so the check order matches the C exactly, and because `MAX_MEMORY` is
// `0x20_0000` on a 32-bit target, where the check is real.
#[allow(clippy::absurd_extreme_comparisons)]
// Nine parameters, one per `argon2_context` field the C checks. Grouping them
// would obscure the 1:1 correspondence with `validate_inputs()`.
#[allow(clippy::too_many_arguments)]
pub const fn validate_inputs(
    out_len: usize,
    pwd_len: usize,
    salt_len: usize,
    secret_len: usize,
    ad_len: usize,
    m_cost: u32,
    t_cost: u32,
    lanes: u32,
    threads: u32,
) -> Result<(), Error> {
    // Validate output length.
    if out_len < MIN_OUTLEN as usize {
        return Err(Error::OutputTooShort);
    }
    if out_len > MAX_OUTLEN as usize {
        return Err(Error::OutputTooLong);
    }

    // Validate password (required param).
    if pwd_len > MAX_PWD_LENGTH as usize {
        return Err(Error::PwdTooLong);
    }

    // Validate salt (required param). Note the C checks the length even when
    // `salt == NULL`, so an empty salt is `SaltTooShort`, not a ptr mismatch.
    if salt_len < MIN_SALT_LENGTH as usize {
        return Err(Error::SaltTooShort);
    }
    if salt_len > MAX_SALT_LENGTH as usize {
        return Err(Error::SaltTooLong);
    }

    // Validate secret (optional param).
    if secret_len > MAX_SECRET as usize {
        return Err(Error::SecretTooLong);
    }

    // Validate associated data (optional param).
    if ad_len > MAX_AD_LENGTH as usize {
        return Err(Error::AdTooLong);
    }

    // Validate memory cost. Three checks, in this order.
    if m_cost < MIN_MEMORY {
        return Err(Error::MemoryTooLittle);
    }
    if m_cost > MAX_MEMORY {
        return Err(Error::MemoryTooMuch);
    }
    if m_cost < 8u32.wrapping_mul(lanes) {
        return Err(Error::MemoryTooLittle);
    }

    // Validate time cost.
    if t_cost < MIN_TIME {
        return Err(Error::TimeTooSmall);
    }
    if t_cost > MAX_TIME {
        return Err(Error::TimeTooLarge);
    }

    // Validate lanes.
    if lanes < MIN_LANES {
        return Err(Error::LanesTooFew);
    }
    if lanes > MAX_LANES {
        return Err(Error::LanesTooMany);
    }

    // Validate threads.
    if threads < MIN_THREADS {
        return Err(Error::ThreadsTooFew);
    }
    if threads > MAX_THREADS {
        return Err(Error::ThreadsTooMany);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------

/// Validated Argon2 cost parameters.
///
/// Holds exactly the fields of `argon2_context` that are *not* byte buffers:
/// `m_cost`, `t_cost`, `lanes`, `threads` and `outlen`. Password, salt, secret
/// and associated data are passed per call.
///
/// A `Params` value can only be built through a constructor that runs
/// [`validate_inputs`], so `lanes >= 1` always holds and the derived values
/// below never divide by zero.
///
/// # Known divergence from the C reference
///
/// The constructors validate the cost parameters immediately, whereas the C
/// checks salt length *before* `m_cost`. If a caller supplies both a bad
/// `m_cost` and a short salt, this crate reports the `m_cost` error at
/// `Params` construction time while the C reports `ARGON2_SALT_TOO_SHORT`.
/// Call [`validate_inputs`] directly to reproduce the C ordering exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Params {
    m_cost: u32,
    t_cost: u32,
    lanes: u32,
    threads: u32,
    output_len: u32,
}

impl Params {
    /// Default memory cost in KiB (19 MiB), per the OWASP Argon2id guidance.
    pub const DEFAULT_M_COST: u32 = 19456;
    /// Default number of passes.
    pub const DEFAULT_T_COST: u32 = 2;
    /// Default degree of parallelism.
    pub const DEFAULT_LANES: u32 = 1;
    /// Default tag length in bytes.
    pub const DEFAULT_OUTPUT_LEN: usize = 32;

    /// Validate and build parameters, with `threads == lanes`.
    ///
    /// This matches `argon2_hash()`, which sets both `context.lanes` and
    /// `context.threads` from its single `parallelism` argument.
    ///
    /// # Errors
    ///
    /// Any of the cost-parameter errors from [`validate_inputs`].
    pub const fn new(
        m_cost: u32,
        t_cost: u32,
        lanes: u32,
        output_len: usize,
    ) -> Result<Params, Error> {
        Params::new_with_threads(m_cost, t_cost, lanes, lanes, output_len)
    }

    /// Validate and build parameters with an explicit thread count.
    ///
    /// `threads` is a pure performance knob: it does **not** affect the tag.
    /// Only `lanes` does. The effective count is `min(threads, lanes)`, see
    /// [`Params::effective_threads`].
    ///
    /// # Errors
    ///
    /// Any of the cost-parameter errors from [`validate_inputs`].
    pub const fn new_with_threads(
        m_cost: u32,
        t_cost: u32,
        lanes: u32,
        threads: u32,
        output_len: usize,
    ) -> Result<Params, Error> {
        // Feed placeholder lengths that always pass their own checks, so the
        // *relative* order of the checks that do apply is exactly the C's.
        match validate_inputs(
            output_len,
            0,
            MIN_SALT_LENGTH as usize,
            0,
            0,
            m_cost,
            t_cost,
            lanes,
            threads,
        ) {
            Ok(()) => {}
            Err(e) => return Err(e),
        }
        Ok(Params {
            m_cost,
            t_cost,
            lanes,
            threads,
            // `output_len <= MAX_OUTLEN == u32::MAX` was just checked.
            output_len: output_len as u32,
        })
    }

    /// Run the full `validate_inputs()` sequence for a concrete call.
    ///
    /// `core` calls this on every hash so the salt/password/secret/ad checks
    /// fire in the C's order.
    ///
    /// # Errors
    ///
    /// Any error from [`validate_inputs`].
    pub const fn validate_for(
        &self,
        pwd_len: usize,
        salt_len: usize,
        secret_len: usize,
        ad_len: usize,
    ) -> Result<(), Error> {
        validate_inputs(
            self.output_len as usize,
            pwd_len,
            salt_len,
            secret_len,
            ad_len,
            self.m_cost,
            self.t_cost,
            self.lanes,
            self.threads,
        )
    }

    /// Requested memory in KiB (`context.m_cost`).
    #[inline]
    #[must_use]
    pub const fn m_cost(&self) -> u32 {
        self.m_cost
    }

    /// Number of passes (`context.t_cost`, `instance.passes`).
    #[inline]
    #[must_use]
    pub const fn t_cost(&self) -> u32 {
        self.t_cost
    }

    /// Degree of parallelism (`context.lanes`). Affects the tag.
    #[inline]
    #[must_use]
    pub const fn lanes(&self) -> u32 {
        self.lanes
    }

    /// Requested worker threads (`context.threads`). Does not affect the tag.
    #[inline]
    #[must_use]
    pub const fn threads(&self) -> u32 {
        self.threads
    }

    /// Tag length in bytes (`context.outlen`).
    #[inline]
    #[must_use]
    pub const fn output_len(&self) -> usize {
        self.output_len as usize
    }

    /// `min(threads, lanes)`, as `argon2_ctx` computes it.
    #[inline]
    #[must_use]
    pub const fn effective_threads(&self) -> u32 {
        if self.threads > self.lanes {
            self.lanes
        } else {
            self.threads
        }
    }

    /// Step 2 of `argon2_ctx()`: align the memory size.
    ///
    /// ```text
    /// memory_blocks = m_cost;
    /// if (memory_blocks < 2 * SYNC_POINTS * lanes)
    ///     memory_blocks = 2 * SYNC_POINTS * lanes;
    /// segment_length = memory_blocks / (lanes * SYNC_POINTS);
    /// memory_blocks  = segment_length * (lanes * SYNC_POINTS);
    /// lane_length    = segment_length * SYNC_POINTS;
    /// ```
    ///
    /// Returns `(memory_blocks, segment_length, lane_length)`. No overflow is
    /// possible: `lanes <= MAX_LANES` (`0xFF_FFFF`), so `lanes * SYNC_POINTS`
    /// fits comfortably in `u32`, and `segment_length * lanes * SYNC_POINTS`
    /// is bounded by the original `memory_blocks <= MAX_MEMORY`.
    #[inline]
    #[must_use]
    pub const fn memory_layout(&self) -> (u32, u32, u32) {
        let lanes_x_sync = self.lanes * SYNC_POINTS;
        let min_blocks = 2 * SYNC_POINTS * self.lanes;

        let mut memory_blocks = self.m_cost;
        if memory_blocks < min_blocks {
            memory_blocks = min_blocks;
        }

        let segment_length = memory_blocks / lanes_x_sync;
        memory_blocks = segment_length * lanes_x_sync;
        let lane_length = segment_length * SYNC_POINTS;

        (memory_blocks, segment_length, lane_length)
    }

    /// Number of 1 KiB blocks the arena needs (`instance.memory_blocks`).
    #[inline]
    #[must_use]
    pub const fn memory_blocks(&self) -> u32 {
        self.memory_layout().0
    }

    /// Blocks per segment (`instance.segment_length`).
    #[inline]
    #[must_use]
    pub const fn segment_length(&self) -> u32 {
        self.memory_layout().1
    }

    /// Blocks per lane (`instance.lane_length` = `segment_length * SYNC_POINTS`).
    #[inline]
    #[must_use]
    pub const fn lane_length(&self) -> u32 {
        self.memory_layout().2
    }
}

impl Default for Params {
    /// `m_cost = 19456` KiB, `t_cost = 2`, `lanes = 1`, `output_len = 32`.
    fn default() -> Params {
        Params {
            m_cost: Params::DEFAULT_M_COST,
            t_cost: Params::DEFAULT_T_COST,
            lanes: Params::DEFAULT_LANES,
            threads: Params::DEFAULT_LANES,
            output_len: Params::DEFAULT_OUTPUT_LEN as u32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_the_c_preprocessor() {
        // Printed by compiling include/argon2.h on aarch64-apple-darwin:
        //   MAX_MEMORY_BITS = 32, MAX_MEMORY = 4294967295, MIN_MEMORY = 8,
        //   MAX_OUTLEN = 4294967295, MAX_LANES = 16777215
        assert_eq!(MIN_MEMORY, 8);
        assert_eq!(MAX_LANES, 16_777_215);
        assert_eq!(MAX_OUTLEN, 4_294_967_295);
        if size_of::<*const u8>() == 8 {
            assert_eq!(MAX_MEMORY_BITS, 32);
            assert_eq!(MAX_MEMORY, 4_294_967_295);
        }
        assert_eq!(BLOCK_SIZE, 1024);
        assert_eq!(QWORDS_IN_BLOCK, 128);
        assert_eq!(OWORDS_IN_BLOCK, 64);
        assert_eq!(HWORDS_IN_BLOCK, 32);
        assert_eq!(BITS512_WORDS_IN_BLOCK, 16);
        assert_eq!(PREHASH_SEED_LENGTH - PREHASH_DIGEST_LENGTH, 8);
    }

    #[test]
    fn default_params_are_valid() {
        let d = Params::default();
        assert!(d.validate_for(0, 8, 0, 0).is_ok());
    }

    #[test]
    fn validate_order_salt_before_m_cost() {
        // Both are bad; the C checks salt first.
        assert_eq!(
            validate_inputs(32, 0, 0, 0, 0, 0, 1, 1, 1),
            Err(Error::SaltTooShort)
        );
    }

    #[test]
    fn validate_order_out_len_first() {
        assert_eq!(
            validate_inputs(0, 0, 0, 0, 0, 0, 0, 0, 0),
            Err(Error::OutputTooShort)
        );
    }

    #[test]
    fn m_cost_lanes_product_wraps_like_c() {
        // 8 * 0xFFFF_FFFF wraps to 0xFFFF_FFF8, so any sane m_cost is "too
        // little" and LanesTooMany never gets a chance to fire.
        assert_eq!(
            validate_inputs(32, 0, 8, 0, 0, 1 << 16, 1, 0xFFFF_FFFF, 1),
            Err(Error::MemoryTooLittle)
        );
        // With lanes in range, the 8*lanes rule is the third memory check.
        assert_eq!(
            validate_inputs(32, 0, 8, 0, 0, 16, 1, 4, 4),
            Err(Error::MemoryTooLittle)
        );
        assert_eq!(validate_inputs(32, 0, 8, 0, 0, 32, 1, 4, 4), Ok(()));
    }

    #[test]
    fn lanes_zero_is_lanes_too_few() {
        // 8 * 0 == 0, so the memory checks pass and LanesTooFew surfaces.
        assert_eq!(
            validate_inputs(32, 0, 8, 0, 0, 8, 1, 0, 1),
            Err(Error::LanesTooFew)
        );
    }

    #[test]
    fn memory_layout_matches_argon2_ctx() {
        // m_cost below the floor gets bumped to 2 * SYNC_POINTS * lanes.
        let p = Params::new(8, 1, 1, 32).unwrap();
        assert_eq!(p.memory_layout(), (8, 2, 8));

        // 1 << 16 KiB, one lane: 65536 blocks, 16384 per segment.
        let p = Params::new(1 << 16, 2, 1, 32).unwrap();
        assert_eq!(p.memory_layout(), (65536, 16384, 65536));

        // Four lanes: segment_length = 65536 / 16 = 4096, lane_length = 16384.
        let p = Params::new(1 << 16, 2, 4, 32).unwrap();
        assert_eq!(p.memory_layout(), (65536, 4096, 16384));

        // Not a multiple of lanes * SYNC_POINTS: truncated down.
        let p = Params::new(100, 1, 3, 32).unwrap();
        let (blocks, seg, lane) = p.memory_layout();
        assert_eq!(seg, 100 / 12);
        assert_eq!(blocks, seg * 12);
        assert_eq!(lane, seg * 4);
    }

    #[test]
    fn effective_threads_is_min() {
        let p = Params::new_with_threads(1 << 16, 1, 2, 8, 32).unwrap();
        assert_eq!(p.threads(), 8);
        assert_eq!(p.effective_threads(), 2);
    }

    #[test]
    fn algorithm_and_version_round_trip() {
        for a in Algorithm::ALL {
            assert_eq!(Algorithm::from_u32(a.as_u32()), Some(a));
        }
        assert_eq!(Algorithm::from_u32(3), None);
        assert_eq!(Algorithm::Argon2id.as_str(), "argon2id");
        assert_eq!(Algorithm::Argon2i.as_str_uppercase(), "Argon2i");
        for v in Version::ALL {
            assert_eq!(Version::from_u32(v.as_u32()), Some(v));
        }
        assert_eq!(Version::from_u32(0x11), None);
        assert_eq!(Version::default(), Version::V0x13);
        assert_eq!(Algorithm::default(), Algorithm::Argon2id);
    }
}
