//! The arena-reuse layer, checked from outside the crate through the public
//! API only.
//!
//! # What this file is for
//!
//! `Argon2::hasher()` hands back a hasher that parks its block arena between
//! calls instead of allocating a fresh one each time. The optimisation is
//! invisible when it works and catastrophic when it does not: a reused arena
//! still holds the *previous* password's derived material at the moment the
//! next hash starts, so a single block read before it is written would make
//! hash N+1 depend on hash N. The tag would still look random. Nothing would
//! crash. It would simply be wrong — differently on every machine, and only for
//! the second and later hashes of a process.
//!
//! So the standard this file holds reuse to is not "the tags look plausible",
//! it is **byte-identity with the one-shot API**, which allocates a fresh
//! zeroed arena every call and is what `tests/kat.rs`, `tests/vectors.rs` and
//! `tests/differential.rs` already pin against the C reference. Every test here
//! computes the same tag twice — once through [`Argon2`], once through a
//! long-lived hasher — and demands they agree byte for byte.
//!
//! # Why the *sequence* is the interesting part
//!
//! A single pooled hash proves almost nothing: its arena was freshly allocated,
//! so it is indistinguishable from the one-shot path. Every test below runs a
//! **sequence**, because only the second and later hashes run on a dirty arena.
//! Three shapes of sequence matter, and each has its own test:
//!
//! | shape | what it could break | test |
//! |---|---|---|
//! | same params, many times | a block read before pass 0 writes it | [`a_long_fixed_parameter_run_never_drifts`] |
//! | A, B, A | the second A picking up B's leftovers | [`interleaving_two_hashes_gives_each_the_same_tag_every_time`] |
//! | big `m_cost`, then small, then big | the shrink/regrow window logic | [`shrinking_and_regrowing_the_arena_is_never_silently_wrong`] |
//!
//! # The leak test, and why "pooled == one-shot" is not quite enough alone
//!
//! [`the_predecessor_cannot_change_the_next_tag`] is the direct measurement:
//! hold one victim hash fixed, vary what ran *before* it through the same
//! arena, and require the victim's tag never to move. That is a stronger
//! statement than "pooled equals one-shot", because it does not lean on the
//! one-shot arena being zeroed — it would catch a leak even if both paths
//! leaked the same way.
//!
//! # A note on naming
//!
//! [`hash_case`] takes an `argon2_rust::Hasher` by `&mut`, and that is worth a
//! sentence because it did not used to compile. `Argon2::hasher()` was `pub` and
//! returned a `pub` type, but `src/lib.rs` re-exported only `Argon2`, so
//! `argon2_rust::Hasher` did not resolve — a hard `E0425`. Inference covers a
//! `let` binding, so every call site here was fine; what it cannot do is let a
//! *function* take one as a parameter, which is why this helper was a macro. A
//! downstream user hit the same wall the moment they wanted a hasher in a struct
//! field or a worker slot, which is the exact shape the reuse layer exists to
//! serve. One `pub use` line fixed it, and `#![warn(unnameable_types)]` in
//! `src/lib.rs` is what stops it coming back.
//!
//! ```text
//! cargo test --test reuse --features internal-api --release
//! ```

use std::collections::BTreeSet;

use argon2_rust::{Algorithm, Argon2, Error, Hasher, Params, Version};

// ---------------------------------------------------------------------------
// Deterministic case generation
// ---------------------------------------------------------------------------

/// SplitMix64, the same generator `tests/differential.rs` uses. No `rand`, no
/// clock, no thread-order dependence: a failure here reproduces by re-running.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> usize {
        (self.next_u64() % n) as usize
    }

    fn pick<T: Copy>(&mut self, choices: &[T]) -> T {
        let index = self.below(choices.len() as u64);
        choices[index]
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next_u64() as u8).collect()
    }
}

/// Everything one hash needs.
#[derive(Clone, PartialEq, Eq)]
struct Case {
    algorithm: Algorithm,
    version: Version,
    params: Params,
    pwd: Vec<u8>,
    salt: Vec<u8>,
    secret: Vec<u8>,
    ad: Vec<u8>,
}

impl std::fmt::Debug for Case {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:?} v={:#x} m={} t={} p={} threads={} outlen={} blocks={} \
             pwd[{}] salt[{}] secret[{}] ad[{}]",
            self.algorithm,
            self.version.as_u32(),
            self.params.m_cost(),
            self.params.t_cost(),
            self.params.lanes(),
            self.params.threads(),
            self.params.output_len(),
            self.params.memory_blocks(),
            self.pwd.len(),
            self.salt.len(),
            self.secret.len(),
            self.ad.len(),
        )
    }
}

impl Case {
    fn argon2(&self) -> Argon2 {
        Argon2::new(self.algorithm, self.version, self.params)
    }

    fn blocks(&self) -> usize {
        self.params.memory_blocks() as usize
    }

    /// The reference answer: a brand-new `alloc_zeroed` arena, freed before
    /// this returns. Every pooled tag in this file is compared against one of
    /// these.
    fn one_shot(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.params.output_len()];
        self.argon2()
            .hash_into_with_ad(&self.pwd, &self.salt, &self.secret, &self.ad, &mut out)
            .unwrap_or_else(|e| panic!("one-shot hash failed for {self:?}: {e:?}"));
        out
    }
}

/// Point a hasher at `case`'s configuration and run it, returning the tag.
fn hash_case(hasher: &mut Hasher, case: &Case) -> Result<Vec<u8>, Error> {
    let mut out = vec![0u8; case.params.output_len()];
    hasher.set_argon2(case.argon2());
    hasher
        .hash_into_with_ad(&case.pwd, &case.salt, &case.secret, &case.ad, &mut out)
        .map(|()| out)
}

/// `m_cost`s spanning three orders of magnitude, deliberately including values
/// that are **not** multiples of `4 * lanes`, so `argon2_ctx` step 2 truncates
/// `memory_blocks` and the arena a hash sees ends up smaller than the request.
/// That truncation is one way `len()` and `capacity()` diverge inside a reused
/// arena, so it has to be in the mix.
const M_COSTS: &[u32] = &[8, 15, 33, 64, 100, 257, 512, 1024, 2048];
const T_COSTS: &[u32] = &[1, 2, 3];
const LANES: &[u32] = &[1, 2, 3, 4];
const OUT_LENS: &[usize] = &[4, 16, 32, 33, 64, 100];

/// `count` cases with every axis varying: algorithm, version, all four cost
/// knobs, and all four input buffers.
fn mixed_cases(seed: u64, count: usize) -> Vec<Case> {
    let mut rng = Rng::new(seed);
    let mut cases = Vec::with_capacity(count);

    while cases.len() < count {
        let lanes = rng.pick(LANES);
        let m_cost = rng.pick(M_COSTS);
        let t_cost = rng.pick(T_COSTS);
        let outlen = rng.pick(OUT_LENS);
        // `threads` is a scheduling knob only — it must never move a tag — so
        // let it disagree with `lanes` half the time.
        let threads = if rng.next_u64() & 1 == 0 { lanes } else { 1 };

        let Ok(params) = Params::new_with_threads(m_cost, t_cost, lanes, threads, outlen) else {
            // `m_cost` below `8 * lanes` is rejected; draw again.
            continue;
        };

        let algorithm = rng.pick(&Algorithm::ALL);
        let version = rng.pick(&Version::ALL);
        let pwd_len = rng.below(40);
        let salt_len = 8 + rng.below(25);
        let secret_len = rng.below(9);
        let ad_len = rng.below(13);

        cases.push(Case {
            algorithm,
            version,
            params,
            pwd: rng.bytes(pwd_len),
            salt: rng.bytes(salt_len),
            secret: rng.bytes(secret_len),
            ad: rng.bytes(ad_len),
        });
    }
    cases
}

/// Fail with the case and the first differing byte, not two walls of hex.
fn assert_same_tag(what: &str, case: &Case, expected: &[u8], actual: &[u8]) {
    if expected == actual {
        return;
    }
    let first = expected
        .iter()
        .zip(actual)
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| expected.len().min(actual.len()));
    panic!(
        "\n{what}: REUSED-ARENA TAG MISMATCH\n\
         \x20 case: {case:?}\n\
         \x20 fresh arena:  {}\n\
         \x20 reused arena: {}\n\
         \x20 first differing byte: index {first} ({:#04x} vs {:#04x})\n\n\
         Arena reuse changed the answer. That is a real bug in the port, not a \
         test problem — fix it, do not loosen this test.\n",
        hex(expected),
        hex(actual),
        expected.get(first).copied().unwrap_or(0),
        actual.get(first).copied().unwrap_or(0),
    );
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

// ---------------------------------------------------------------------------
// 1. The headline claim: a long mixed sequence through one arena
// ---------------------------------------------------------------------------

/// Many hashes through **one** reused arena must produce exactly the tags the
/// same hashes produce each through a **fresh** arena.
///
/// This is the property the whole change rests on, and the reason it is worth
/// 200 cases rather than 2: every axis moves between consecutive hashes —
/// algorithm, version, `m_cost`, `t_cost`, `lanes`, `threads`, output length
/// and all four input buffers — so within a single allocation's lifetime the
/// arena is grown, re-lent as a narrow window over a larger allocation, and
/// handed to both the single-threaded fill and the `std::thread::scope` one.
///
/// Both halves run in the *same order*, so exactly one variable differs: where
/// the arena came from.
#[test]
fn a_reused_arena_reproduces_the_one_shot_tag_for_a_long_mixed_sequence() {
    let cases = mixed_cases(0x_5EED_0001, 200);

    // Reference pass: every case on its own fresh allocation.
    let expected: Vec<Vec<u8>> = cases.iter().map(Case::one_shot).collect();

    // Pooled pass: the same order, one arena.
    let mut hasher = cases[0].argon2().hasher();
    let mut high_water = 0usize;
    let mut grows = 0usize;

    for (index, case) in cases.iter().enumerate() {
        let got = hash_case(&mut hasher, case)
            .unwrap_or_else(|e| panic!("pooled hash failed at case {index} ({case:?}): {e:?}"));
        assert_same_tag(
            &format!("mixed-sequence[{index}]"),
            case,
            &expected[index],
            &got,
        );

        // Reuse is a claim about allocations, so check it rather than assume
        // it: the arena may only ever grow, and only when a case needs more
        // blocks than every case before it did.
        let reserved = hasher.reserved_blocks();
        assert!(
            reserved >= case.blocks(),
            "case {index}: the hasher kept {reserved} blocks but the hash needed {}",
            case.blocks()
        );
        if reserved > high_water {
            grows += 1;
            high_water = reserved;
        }
        assert_eq!(
            reserved, high_water,
            "case {index}: the arena went from {high_water} to {reserved} blocks; \
             a smaller m_cost must re-lend the existing allocation, not replace it"
        );
    }

    // The sweep has to actually exercise reuse rather than degenerate into
    // "every case is bigger than the last", which would reallocate every time.
    let distinct_sizes: BTreeSet<u32> = cases.iter().map(|c| c.params.memory_blocks()).collect();
    assert!(
        distinct_sizes.len() >= 8,
        "the sweep must span many arena sizes, saw {distinct_sizes:?}"
    );
    assert!(
        grows <= 12,
        "{grows} allocations across {} hashes means the arena is barely being \
         reused, so this test proves much less than it appears to",
        cases.len()
    );
    println!(
        "mixed-sequence: {} hashes, {} distinct arena sizes, {grows} allocations, \
         settled at {} blocks",
        cases.len(),
        distinct_sizes.len(),
        hasher.reserved_blocks(),
    );
}

// ---------------------------------------------------------------------------
// 2. Interleaving
// ---------------------------------------------------------------------------

/// `A, B, A` through one arena: both `A`s must give the same tag, and it must
/// be `A`'s one-shot tag.
///
/// Run for every `(algorithm, version)` pair, with `A` and `B` differing in
/// *every* way at once — password, salt, secret, associated data, and all four
/// cost parameters — so the arena between two `A`s has been resized, refilled
/// by a different configuration, and left holding a different password's
/// material. The three rounds catch the case where only the *first* repeat is
/// clean.
#[test]
fn interleaving_two_hashes_gives_each_the_same_tag_every_time() {
    for algorithm in Algorithm::ALL {
        for version in Version::ALL {
            // Small, single-threaded, multi-pass.
            let a = Case {
                algorithm,
                version,
                params: Params::new(64, 3, 1, 32).expect("A params"),
                pwd: b"password-A".to_vec(),
                salt: b"salt-for-A".to_vec(),
                secret: b"secretAA".to_vec(),
                ad: b"assoc-data-A".to_vec(),
            };
            // Larger, four lanes, four threads, longer tag, no secret or ad.
            let b = Case {
                algorithm,
                version,
                params: Params::new_with_threads(1024, 1, 4, 4, 64).expect("B params"),
                pwd: b"a completely different password, B".to_vec(),
                salt: b"salt-for-B-which-is-longer".to_vec(),
                secret: Vec::new(),
                ad: Vec::new(),
            };

            let want_a = a.one_shot();
            let want_b = b.one_shot();
            assert_ne!(want_a[..], want_b[..want_a.len()], "A and B must differ");

            let mut hasher = a.argon2().hasher();
            let label = format!("{algorithm:?} {version:?}");

            for round in 0..3 {
                for (name, case, want) in [("A", &a, &want_a), ("B", &b, &want_b)] {
                    let got = hash_case(&mut hasher, case).expect("interleaved hash");
                    assert_same_tag(
                        &format!("{label} round {round} hash {name}"),
                        case,
                        want,
                        &got,
                    );
                }
            }

            // One allocation serves the pair: B is the larger, so after the
            // first B nothing may grow again.
            assert_eq!(
                hasher.reserved_blocks(),
                b.blocks(),
                "{label}: the arena should have settled at B's size"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// 3. The leak test
// ---------------------------------------------------------------------------

/// **A reused arena must not leak the previous password's material into the
/// next result.** Measured directly: fix one victim hash, vary what ran before
/// it through the same arena, and require the victim's tag never to move.
///
/// Deliberately not phrased as "pooled equals one-shot". If both paths somehow
/// leaked, that comparison would still pass; this one cannot, because the only
/// thing differing between the arms *is* the arena's previous contents.
///
/// Ten predecessors chosen to leave maximally different residue: different
/// passwords, all three algorithms, both versions, one to four lanes, and
/// `m_cost`s both larger and smaller than the victim's — so the victim
/// sometimes runs over a window that is a strict subset of a larger dirty
/// allocation, which is the case with the most room to go wrong.
///
/// The control is at the bottom: the predecessors' own tags must be pairwise
/// distinct. Ten predecessors that all computed the same thing would make
/// everything above vacuous.
#[test]
fn the_predecessor_cannot_change_the_next_tag() {
    let victim = Case {
        algorithm: Algorithm::Argon2id,
        version: Version::V0x13,
        params: Params::new_with_threads(256, 2, 2, 2, 32).expect("victim params"),
        pwd: b"the-victim-password".to_vec(),
        salt: b"the-victim-salt!".to_vec(),
        secret: b"vsecret1".to_vec(),
        ad: b"victim-assoc".to_vec(),
    };
    let want = victim.one_shot();

    let predecessors: Vec<Case> = [
        (Algorithm::Argon2d, Version::V0x13, 8u32, 1u32, 1u32),
        (Algorithm::Argon2i, Version::V0x10, 4096, 1, 1),
        (Algorithm::Argon2id, Version::V0x10, 33, 3, 1),
        (Algorithm::Argon2d, Version::V0x10, 1024, 2, 4),
        (Algorithm::Argon2i, Version::V0x13, 100, 1, 2),
        (Algorithm::Argon2id, Version::V0x13, 2048, 1, 3),
        (Algorithm::Argon2d, Version::V0x13, 15, 2, 1),
        (Algorithm::Argon2i, Version::V0x13, 512, 3, 4),
        (Algorithm::Argon2id, Version::V0x13, 257, 1, 1),
        (Algorithm::Argon2d, Version::V0x10, 64, 1, 2),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (algorithm, version, m_cost, t_cost, lanes))| Case {
        algorithm,
        version,
        params: Params::new_with_threads(m_cost, t_cost, lanes, lanes, 32).expect("pred params"),
        pwd: format!("predecessor-{i}").into_bytes(),
        salt: format!("predecessor-salt-{i}").into_bytes(),
        secret: vec![i as u8; 8],
        ad: vec![0xA0 ^ i as u8; 12],
    })
    .collect();

    // Arm 0: no predecessor at all — a virgin hasher.
    let mut virgin = victim.argon2().hasher();
    let got = hash_case(&mut virgin, &victim).expect("victim on a virgin hasher");
    assert_same_tag("no predecessor", &victim, &want, &got);

    // Arms 1..=N: exactly one predecessor each, through one hasher.
    let mut residues = BTreeSet::new();
    for (index, pred) in predecessors.iter().enumerate() {
        let mut hasher = pred.argon2().hasher();

        let pred_tag = hash_case(&mut hasher, pred).expect("predecessor hash");
        residues.insert(hex(&pred_tag));

        let got = hash_case(&mut hasher, &victim).expect("victim after a predecessor");
        assert_same_tag(
            &format!("after predecessor {index} ({pred:?})"),
            &victim,
            &want,
            &got,
        );
    }

    // And with every predecessor run first, in order, through one arena.
    let mut hasher = victim.argon2().hasher();
    for pred in &predecessors {
        hash_case(&mut hasher, pred).expect("predecessor hash");
    }
    let got = hash_case(&mut hasher, &victim).expect("victim after all predecessors");
    assert_same_tag("after all predecessors", &victim, &want, &got);

    // CONTROL: the predecessors must really have differed from one another.
    assert_eq!(
        residues.len(),
        predecessors.len(),
        "the predecessors must produce distinct material, otherwise this test \
         proves nothing about leakage"
    );
    assert!(
        !residues.contains(&hex(&want)),
        "a predecessor accidentally computed the victim's tag"
    );
}

// ---------------------------------------------------------------------------
// 4. Growing and shrinking
// ---------------------------------------------------------------------------

/// Reuse across different parameters, in the order most likely to break it:
/// large, then small, then large again.
///
/// A small hash after a large one runs over a *window* — `len()` blocks of a
/// larger `capacity()` — which is the one situation where the arena the hash
/// sees and the arena the allocator owns are different sizes. If the fill ever
/// addressed past `len()`, or the release wipe ever covered the wrong range,
/// this ladder is where it shows.
///
/// Every rung is compared to its own one-shot tag, so correctness is checked at
/// every step rather than only at the end.
#[test]
fn shrinking_and_regrowing_the_arena_is_never_silently_wrong() {
    // Not monotone, with the truncating `m_cost`s (33, 257, 15) sitting next to
    // the round ones.
    let ladder: &[(u32, u32, u32)] = &[
        (4096, 1, 4),
        (8, 1, 1),
        (2048, 2, 2),
        (33, 1, 1),
        (1024, 1, 3),
        (257, 3, 1),
        (4096, 1, 4),
        (15, 2, 1),
        (4096, 2, 4),
        (8, 3, 1),
    ];

    for algorithm in Algorithm::ALL {
        let mut hasher = Argon2::new(
            algorithm,
            Version::V0x13,
            Params::new(8, 1, 1, 32).expect("seed params"),
        )
        .hasher();
        let mut high_water = 0usize;

        for (step, &(m_cost, t_cost, lanes)) in ladder.iter().enumerate() {
            let case = Case {
                algorithm,
                version: Version::V0x13,
                params: Params::new_with_threads(m_cost, t_cost, lanes, lanes, 32)
                    .expect("ladder params"),
                pwd: format!("step-{step}-password").into_bytes(),
                salt: format!("step-{step}-salt-abc").into_bytes(),
                secret: vec![step as u8; 8],
                ad: vec![step as u8 ^ 0x5A; 12],
            };

            let want = case.one_shot();
            let got = hash_case(&mut hasher, &case).expect("ladder hash");
            assert_same_tag(
                &format!("{algorithm:?} ladder step {step}"),
                &case,
                &want,
                &got,
            );

            high_water = high_water.max(case.blocks());
            assert_eq!(
                hasher.reserved_blocks(),
                high_water,
                "{algorithm:?} step {step}: the arena must sit at the high-water \
                 mark — never shrink, never over-allocate"
            );
        }

        // The ladder revisits 4096 three times; only the first may allocate.
        assert_eq!(
            hasher.reserved_blocks(),
            Params::new_with_threads(4096, 1, 4, 4, 32)
                .expect("params")
                .memory_blocks() as usize
        );
    }
}

/// The same parameters, over and over — the simplest sequence, and the one a
/// real server actually runs.
///
/// Separated from the mixed sweep because it isolates a different failure. Here
/// the arena is never resized, so the only thing that changes between rounds is
/// that it starts dirty. 64 rounds with a **different password each time**, so
/// a leak would surface as a tag that depends on the round number.
#[test]
fn a_long_fixed_parameter_run_never_drifts() {
    for (m_cost, t_cost, lanes) in [(64u32, 1u32, 1u32), (512, 2, 4), (33, 3, 2)] {
        let params = Params::new_with_threads(m_cost, t_cost, lanes, lanes, 32).expect("params");
        let mut hasher = Argon2::new(Algorithm::Argon2id, Version::V0x13, params).hasher();

        let mut seen = BTreeSet::new();
        for round in 0..64u32 {
            let case = Case {
                algorithm: Algorithm::Argon2id,
                version: Version::V0x13,
                params,
                pwd: format!("password-{round}").into_bytes(),
                salt: b"a-fixed-salt-16b".to_vec(),
                secret: Vec::new(),
                ad: Vec::new(),
            };
            let want = case.one_shot();
            let got = hash_case(&mut hasher, &case).expect("fixed-parameter hash");
            assert_same_tag(
                &format!("m={m_cost} p={lanes} round {round}"),
                &case,
                &want,
                &got,
            );
            seen.insert(hex(&got));
        }

        assert_eq!(seen.len(), 64, "64 distinct passwords must give 64 tags");
        assert_eq!(
            hasher.reserved_blocks(),
            params.memory_blocks() as usize,
            "64 hashes must have shared exactly one arena"
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Errors: correct, or a clean error — never silently wrong
// ---------------------------------------------------------------------------

/// Rejected inputs must leave the hasher usable and still right.
///
/// Reuse adds a way to get this wrong that the one-shot path did not have: an
/// arena on loan when the call bails out. The guard's `Drop` is supposed to
/// hand it back, wiped, on every exit including `?`. If it did not, the next
/// hash would either see a lost allocation (merely slow) or a half-released one
/// (wrong). So every rejection below is followed by a real hash checked against
/// its one-shot tag.
#[test]
fn a_rejected_input_leaves_the_hasher_correct() {
    let good = Case {
        algorithm: Algorithm::Argon2id,
        version: Version::V0x13,
        params: Params::new_with_threads(256, 2, 2, 2, 32).expect("params"),
        pwd: b"password".to_vec(),
        salt: b"somesalt-16-byte".to_vec(),
        secret: Vec::new(),
        ad: Vec::new(),
    };
    let want = good.one_shot();

    let mut hasher = good.argon2().hasher();

    // Warm the arena up first, so a botched release has something to lose.
    let got = hash_case(&mut hasher, &good).expect("warm-up");
    assert_same_tag("warm-up", &good, &want, &got);
    let reserved = hasher.reserved_blocks();
    assert_eq!(reserved, good.blocks());

    // (a) `out.len()` disagrees with `params.output_len()`.
    let mut too_short = [0u8; 16];
    assert_eq!(
        hasher.hash_into(&good.pwd, &good.salt, &mut too_short),
        Err(Error::OutPtrMismatch)
    );

    // (b) salt below `ARGON2_MIN_SALT_LENGTH`.
    let mut out = [0u8; 32];
    assert_eq!(
        hasher.hash_into(&good.pwd, b"short", &mut out),
        Err(Error::SaltTooShort)
    );

    // (c) a `verify` that legitimately fails to match. This one does allocate
    // and run a whole hash before failing, so the arena really is on loan when
    // the error is produced.
    assert_eq!(
        hasher.verify(&good.pwd, &good.salt, &[0u8; 32]),
        Err(Error::VerifyMismatch)
    );

    // (d) an encoded string that does not parse.
    assert!(
        hasher
            .verify_encoded(
                "$argon2id$not-a-real-phc-string",
                b"password",
                Algorithm::Argon2id
            )
            .is_err()
    );

    // The hasher must be untouched: same arena, same answers.
    assert_eq!(
        hasher.reserved_blocks(),
        reserved,
        "a rejected call must not resize the arena"
    );
    for round in 0..3 {
        let got = hash_case(&mut hasher, &good).expect("after rejections");
        assert_same_tag(
            &format!("after rejections, round {round}"),
            &good,
            &want,
            &got,
        );
    }
}

// ---------------------------------------------------------------------------
// 6. reserve / clear
// ---------------------------------------------------------------------------

/// `reserve()` and `clear()` move the allocation around without moving a single
/// output bit.
///
/// `clear()` is the interesting half: it returns the arena to the allocator, so
/// the hash after it runs on a *fresh* allocation again. Alternating cleared
/// and warm hashes puts the first-allocation path and the reuse path
/// side by side, and they must not disagree.
#[test]
fn reserve_and_clear_do_not_change_any_answer() {
    let case = Case {
        algorithm: Algorithm::Argon2id,
        version: Version::V0x13,
        params: Params::new_with_threads(512, 2, 2, 2, 32).expect("params"),
        pwd: b"password".to_vec(),
        salt: b"somesalt-16-byte".to_vec(),
        secret: b"secret!!".to_vec(),
        ad: b"associated..".to_vec(),
    };
    let want = case.one_shot();
    let blocks = case.blocks();

    let mut hasher = case.argon2().hasher();
    assert_eq!(hasher.reserved_blocks(), 0, "nothing allocated up front");

    hasher.reserve().expect("reserve");
    assert_eq!(
        hasher.reserved_blocks(),
        blocks,
        "reserve must allocate the arena now"
    );

    for round in 0..3 {
        let got = hash_case(&mut hasher, &case).expect("warm hash");
        assert_same_tag(&format!("warm, round {round}"), &case, &want, &got);
        assert_eq!(hasher.reserved_blocks(), blocks);

        hasher.clear();
        assert_eq!(hasher.reserved_blocks(), 0, "clear must release the arena");

        let got = hash_case(&mut hasher, &case).expect("hash after clear");
        assert_same_tag(&format!("after clear, round {round}"), &case, &want, &got);
        assert_eq!(
            hasher.reserved_blocks(),
            blocks,
            "the hash after a clear must allocate again"
        );

        // `reserve` on an already-large-enough arena is a no-op, not a
        // reallocation.
        hasher.reserve().expect("reserve again");
        assert_eq!(hasher.reserved_blocks(), blocks);
    }
}

// ---------------------------------------------------------------------------
// 7. Encoded strings through a pooled hasher
// ---------------------------------------------------------------------------

/// `hash_encoded` / `verify_encoded` through one hasher across a mix of
/// `m_cost`s, verified out of order.
///
/// `verify_encoded` is the one method that takes its parameters from the
/// *string* rather than from the hasher, so a single hasher ends up sized to
/// the largest `m_cost` it has ever been shown. Verifying in an order that
/// jumps between sizes is the natural stress for the window-over-a-larger-
/// allocation path, and it is exactly what a server migrating stored hashes
/// does.
#[test]
fn encoded_hashes_round_trip_through_one_hasher_across_mixed_costs() {
    let costs: &[(u32, u32, u32)] = &[(256, 2, 1), (8, 1, 1), (2048, 1, 4), (64, 3, 2), (33, 1, 1)];

    let mut hasher = Argon2::new(
        Algorithm::Argon2id,
        Version::V0x13,
        Params::new(8, 1, 1, 32).expect("seed"),
    )
    .hasher();

    // Produce the strings, cross-checking each against the one-shot encoder.
    let mut encoded = Vec::new();
    for (index, &(m_cost, t_cost, lanes)) in costs.iter().enumerate() {
        let params = Params::new_with_threads(m_cost, t_cost, lanes, lanes, 32).expect("params");
        let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
        let pwd = format!("password-{index}");
        let salt = format!("salt-number-{index}--");

        hasher.set_argon2(argon2);
        let pooled = hasher
            .hash_encoded(pwd.as_bytes(), salt.as_bytes())
            .expect("pooled encode");
        let one_shot = argon2
            .hash_encoded(pwd.as_bytes(), salt.as_bytes())
            .expect("one-shot encode");
        assert_eq!(
            pooled, one_shot,
            "the encoded form differs at m_cost {m_cost} (index {index})"
        );
        encoded.push((pwd, pooled));
    }

    // Verify out of order: big, small, big, small. Each `verify_encoded` uses
    // the string's own parameters, so the arena is re-lent at a different size
    // almost every call.
    let order = [2usize, 1, 4, 0, 3, 2, 0, 4, 1, 3];
    for (round, &index) in order.iter().enumerate() {
        let (pwd, string) = &encoded[index];
        hasher
            .verify_encoded(string, pwd.as_bytes(), Algorithm::Argon2id)
            .unwrap_or_else(|e| panic!("round {round}: verifying {string} failed: {e:?}"));

        // ...and the wrong password must still be rejected, on the same arena.
        assert_eq!(
            hasher.verify_encoded(string, b"wrong password", Algorithm::Argon2id),
            Err(Error::VerifyMismatch),
            "round {round}: a wrong password was accepted for {string}"
        );
    }

    // The arena settled at the largest cost the hasher was ever shown.
    let largest = Params::new_with_threads(2048, 1, 4, 4, 32)
        .expect("params")
        .memory_blocks() as usize;
    assert_eq!(hasher.reserved_blocks(), largest);
}

// ---------------------------------------------------------------------------
// 8. Threads
// ---------------------------------------------------------------------------

/// A hasher is `Send`: a warm one must keep answering correctly after moving to
/// another thread, and several of them running at once must not interfere.
///
/// The second half is the real check. The type is deliberately not `Sync`, so
/// two threads cannot share one — but they *can* each hold their own, and a bug
/// that put arena state anywhere process-wide (a `static`, a lazily initialised
/// pool) would surface as cross-talk here and nowhere else in this file.
///
/// Not on wasi: there are no threads there (`spawn` always fails), so the
/// premise of this test cannot be set up at all.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn hashers_on_several_threads_stay_independent() {
    let cases = mixed_cases(0x_5EED_0002, 24);
    let expected: Vec<Vec<u8>> = cases.iter().map(Case::one_shot).collect();

    // (a) Move a warm hasher across a thread boundary.
    let mut hasher = cases[0].argon2().hasher();
    let warm = hash_case(&mut hasher, &cases[0]).expect("warm-up");
    assert_eq!(warm, expected[0]);

    let case0 = cases[0].clone();
    let want0 = expected[0].clone();
    std::thread::spawn(move || {
        let got = hash_case(&mut hasher, &case0).expect("hash on the moved hasher");
        assert_same_tag("after moving threads", &case0, &want0, &got);
    })
    .join()
    .expect("the moved hasher panicked");

    // (b) Four threads, four hashers, the same sequence in each.
    std::thread::scope(|scope| {
        for worker in 0..4 {
            let cases = &cases;
            let expected = &expected;
            scope.spawn(move || {
                let mut hasher = cases[0].argon2().hasher();
                for (index, case) in cases.iter().enumerate() {
                    let got = hash_case(&mut hasher, case).expect("worker hash");
                    assert_same_tag(
                        &format!("worker {worker} case {index}"),
                        case,
                        &expected[index],
                        &got,
                    );
                }
            });
        }
    });
}

// ---------------------------------------------------------------------------
// 9. Negative control
// ---------------------------------------------------------------------------

/// Everything above claims "the tags matched". That claim is worth exactly as
/// much as [`assert_same_tag`]'s ability to notice when they do not, so prove
/// it notices — and that it names the case and the first differing byte, which
/// is what makes a real failure diagnosable.
/// Not on wasi: the check works by `catch_unwind`, and wasi is panic=abort.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn the_comparator_catches_a_changed_tag() {
    let case = Case {
        algorithm: Algorithm::Argon2id,
        version: Version::V0x13,
        params: Params::new(64, 1, 1, 32).expect("params"),
        pwd: b"password".to_vec(),
        salt: b"somesalt-16-byte".to_vec(),
        secret: Vec::new(),
        ad: Vec::new(),
    };
    let want = case.one_shot();

    // Baseline: identical tags must pass.
    assert_same_tag("control-baseline", &case, &want, &want);

    // One flipped bit, in the middle of the tag, must not.
    let mut flipped = want.clone();
    flipped[17] ^= 0x40;

    let payload = std::panic::catch_unwind(|| {
        assert_same_tag("negative-control", &case, &want, &flipped);
    })
    .expect_err("a changed tag must fail the comparison");

    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("");

    assert!(
        message.contains("REUSED-ARENA TAG MISMATCH") && message.contains("index 17"),
        "the failure message must localise the divergence, got:\n{message}"
    );
}
