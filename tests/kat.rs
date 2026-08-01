//! Golden known-answer tests against the six trace files in
//! `phc-winner-argon2/kats/`.
//!
//! Those files are the stdout of `src/genkat.c`, built with `-DGENKAT` so that
//! `src/core.c` calls three hooks:
//!
//! | hook | `core.c` site | what it prints |
//! |---|---|---|
//! | `initial_kat(blockhash, context, type)` | `core.c:638`, inside `initialize()` right after `initial_hash()` | the 9-line header, ending with the 64-byte pre-hashing digest `H0` |
//! | `internal_kat(instance, r)` | `core.c:271` / `core.c:360`, at the bottom of each pass of `fill_memory_blocks_{st,mt}` | every word of every block in the arena |
//! | `print_tag(context->out, context->outlen)` | `core.c:180`, at the end of `finalize()` | the final tag |
//!
//! `generate_testvectors()` in `genkat.c` pins the parameters:
//!
//! ```text
//! t_cost 3, m_cost 32, lanes 4, threads = lanes, outlen 32
//! pwd    = 32 bytes of 0x01     salt = 16 bytes of 0x02
//! secret =  8 bytes of 0x03     ad   = 12 bytes of 0x04
//! ```
//!
//! `m_cost = 32` with `lanes = 4` aligns to `memory_blocks = 32`, and 32 is
//! **not** greater than `ARGON2_QWORDS_IN_BLOCK` (128), so `internal_kat`'s
//! ternary
//!
//! ```c
//! uint32_t how_many_words =
//!     (instance->memory_blocks > ARGON2_QWORDS_IN_BLOCK) ? 1
//!                                                        : ARGON2_QWORDS_IN_BLOCK;
//! ```
//!
//! selects the full 128-word dump. Each file is therefore
//! `9 + 3 * (2 + 32 * 128) + 1 = 12304` lines, and this test reproduces every
//! one of them.
//!
//! # Why the whole trace and not just the tag
//!
//! A tag comparison tells you *that* the port is wrong. A block-level trace
//! tells you *where*: the failure message names the pass, the block index and
//! the word index of the first divergence. Since the arena is dumped after
//! every pass, a bug in `fill_block` shows up at the first block it touches
//! rather than being smeared across 12288 words by the finalisation hash.
//!
//! # Backends
//!
//! Every file is regenerated once per [`Backend`] this CPU can actually
//! execute, via `__internal::hash_traced`, which takes the backend explicitly.
//! [`every_runnable_backend_is_kat_checked`] prints and pins that list so it
//! cannot quietly shrink to `[Scalar]`.
//!
//! # Fresh arena and reused arena
//!
//! `hash_traced` allocates a fresh, zeroed arena per call, so the six trace
//! tests say nothing about the arena-reuse layer behind `Argon2::hasher()`.
//! [`a_pooled_hasher_reproduces_every_kat_tag`] closes that: it runs all six
//! files through **one** hasher, rotating the order over four rounds, so each
//! golden tag is also reproduced on memory that a *different* algorithm or
//! version just filled. Its own doc comment records what it cannot reach and
//! where that coverage lives instead.
//!
//! ```text
//! cargo test --test kat --features internal-api --release
//! cargo test --test kat --features internal-api --release --target x86_64-apple-darwin
//! ```
//!
//! Measured on this host (Apple M5 Max, macOS 25.5):
//!
//! | invocation | backends covered |
//! |---|---|
//! | `--target aarch64-apple-darwin` | `Scalar`, `Neon` |
//! | `--target x86_64-apple-darwin` (Rosetta) | `Scalar`, `Sse2` |
//! | …`+ RUSTFLAGS="-C target-feature=+avx2"` | `Scalar`, `Sse2`, `Avx2` |
//! | …`+ -- --ignored forced_avx2` | `Avx2` alone, detection bypassed |
//!
//! Rosetta on this machine does **not** advertise AVX: `hw.optional.avx2_0`
//! is 0 and `is_x86_feature_detected!("avx2")` is `false`, so plain runtime
//! detection can never pick `Avx2` here — which is right, not a bug. The list
//! is computed from [`Backend::is_available`], so it widens by itself on a host
//! that does advertise them, and the two extra invocations above cover AVX2
//! anyway. See `forced_avx2_reproduces_every_golden_file` (x86-64 only, hence
//! not linked) for which to prefer. `Avx512` traps with `SIGILL` under Rosetta
//! even when forced, so it is compile-verified only.

use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use argon2_rust::__internal::{Block, PassTrace, hash_traced};
use argon2_rust::params::QWORDS_IN_BLOCK;
use argon2_rust::{Algorithm, Argon2, Backend, Params, Version};

// ---------------------------------------------------------------------------
// genkat.c's `generate_testvectors()` parameters
// ---------------------------------------------------------------------------

/// `unsigned t_cost = 3;`
const T_COST: u32 = 3;
/// `unsigned m_cost = 32;`
const M_COST: u32 = 32;
/// `unsigned lanes = 4;` — and `context.threads = lanes`.
const LANES: u32 = 4;
/// `#define TEST_OUTLEN 32`
const OUTLEN: usize = 32;

/// `memset(pwd, 1, TEST_OUTLEN);` — note `genkat.c` uses `TEST_OUTLEN`, not
/// `TEST_PWDLEN`, but both are 32 so the buffer is filled completely.
const PWD: [u8; 32] = [0x01; 32];
/// `memset(salt, 2, TEST_SALTLEN);`
const SALT: [u8; 16] = [0x02; 16];
/// `memset(secret, 3, TEST_SECRETLEN);`
const SECRET: [u8; 8] = [0x03; 8];
/// `memset(ad, 4, TEST_ADLEN);`
const AD: [u8; 12] = [0x04; 12];

/// Lines in every golden file: header + 3 passes + tag.
const EXPECTED_LINES: usize = 9 + 3 * (2 + 32 * QWORDS_IN_BLOCK) + 1;

// ---------------------------------------------------------------------------
// Regenerating the trace
// ---------------------------------------------------------------------------

/// `printf("%2.2x ", b)` in a loop, then `printf("\n")`.
///
/// Note the **trailing space** before the newline: `genkat.c` emits a separator
/// after every byte, including the last one. Dropping it is the easiest way to
/// fail this test for a reason that has nothing to do with Argon2.
fn push_hex_bytes(sink: &mut String, bytes: &[u8]) {
    for b in bytes {
        let _ = write!(sink, "{b:02x} ");
    }
    sink.push('\n');
}

/// Reproduce one golden file exactly, using `backend` for every `fill_segment`.
///
/// # Safety
///
/// As [`hash_traced`]: this CPU must be able to execute `backend`.
/// [`runnable_backends`] is the ordinary way to establish that. The single
/// exception is `forced_avx2_reproduces_every_golden_file`, which relies on this
/// host having been *measured* to execute AVX2 while hiding it from `cpuid`.
unsafe fn generate(algorithm: Algorithm, version: Version, backend: Backend) -> String {
    // `argon2_ctx` takes `threads` from `context.threads`, which `genkat.c`
    // sets to `lanes`. `threads` cannot change the tag (only `lanes` can), but
    // matching it keeps this identical to the C run in every respect.
    let params = Params::new_with_threads(M_COST, T_COST, LANES, LANES, OUTLEN)
        .expect("genkat.c's parameters are valid");

    let mut tag = [0u8; OUTLEN];

    // 12288 block lines of 26 bytes each, plus the pass banners.
    let mut passes = String::with_capacity(330 * 1024);

    // `internal_kat(instance, r)` from genkat.c.
    let mut trace = |pass: u32, blocks: &[Block]| {
        assert_eq!(
            blocks.len(),
            32,
            "memory_blocks should align to 32 for m_cost=32, lanes=4"
        );

        let _ = write!(passes, "\n After pass {pass}:\n");

        // The C ternary, verbatim. It is `1` only when the arena is larger than
        // one block's worth of words, which these parameters never are.
        let how_many_words = if blocks.len() > QWORDS_IN_BLOCK {
            1
        } else {
            QWORDS_IN_BLOCK
        };

        for (i, block) in blocks.iter().enumerate() {
            for j in 0..how_many_words {
                // printf("Block %.4u [%3u]: %016llx\n", i, j, block->v[j]);
                let _ = writeln!(passes, "Block {i:04} [{j:3}]: {:016x}", block[j]);
            }
        }
    };

    // SAFETY: forwarded verbatim from this function's own contract — the caller
    // guarantees this CPU can execute `backend`.
    let h0 = unsafe {
        hash_traced(
            backend,
            algorithm,
            version,
            &params,
            &PWD,
            &SALT,
            &SECRET,
            &AD,
            &mut tag,
            Some(&mut trace as PassTrace),
        )
    }
    .expect("hashing genkat.c's parameters cannot fail");

    // `initial_kat()` runs *before* the passes but needs `H0`, which
    // `hash_traced` only hands back at the end, so the header is assembled last
    // and the pass dump appended to it.
    let mut out = String::with_capacity(passes.len() + 1024);

    out.push_str("=======================================\n");
    let _ = writeln!(
        out,
        "{} version number {}",
        algorithm.as_str_uppercase(),
        version.as_u32()
    );
    out.push_str("=======================================\n");
    let _ = writeln!(
        out,
        "Memory: {M_COST} KiB, Iterations: {T_COST}, Parallelism: {LANES} lanes, \
         Tag length: {OUTLEN} bytes"
    );

    let _ = write!(out, "Password[{}]: ", PWD.len());
    push_hex_bytes(&mut out, &PWD);
    let _ = write!(out, "Salt[{}]: ", SALT.len());
    push_hex_bytes(&mut out, &SALT);
    let _ = write!(out, "Secret[{}]: ", SECRET.len());
    push_hex_bytes(&mut out, &SECRET);
    let _ = write!(out, "Associated data[{}]: ", AD.len());
    push_hex_bytes(&mut out, &AD);
    out.push_str("Pre-hashing digest: ");
    push_hex_bytes(&mut out, &h0);

    out.push_str(&passes);

    // `print_tag(context->out, context->outlen)` from `finalize()`.
    out.push_str("Tag: ");
    push_hex_bytes(&mut out, &tag);

    out
}

// ---------------------------------------------------------------------------
// Comparison with localised diagnostics
// ---------------------------------------------------------------------------

/// Decode `Block %.4u [%3u]: %016llx` back into `(block, word, value)`.
fn parse_block_line(line: &str) -> Option<(u32, u32, u64)> {
    let rest = line.strip_prefix("Block ")?;
    let open = rest.find('[')?;
    let close = rest.find(']')?;
    let block = rest[..open].trim().parse().ok()?;
    let word = rest[open + 1..close].trim().parse().ok()?;
    let value = u64::from_str_radix(rest[close + 1..].trim_start_matches(':').trim(), 16).ok()?;
    Some((block, word, value))
}

/// Compare the regenerated trace with the golden file, line by line, and turn
/// the first difference into a message that names the pass, block and word.
fn assert_trace_matches(label: &str, expected: &str, actual: &str) {
    let mut section = String::from("the header (before pass 0)");
    let mut expected_lines = expected.lines();
    let mut actual_lines = actual.lines();
    let mut line_no = 0usize;

    loop {
        match (expected_lines.next(), actual_lines.next()) {
            (Some(want), Some(got)) => {
                line_no += 1;

                if want != got {
                    let mut msg = format!(
                        "\n{label}: KAT MISMATCH at line {line_no}, in {section}\n\
                         \x20 golden (C): {want}\n\
                         \x20 ours (Rust): {got}\n"
                    );
                    if let (Some((wb, ww, wv)), Some((gb, gw, gv))) =
                        (parse_block_line(want), parse_block_line(got))
                    {
                        let _ = write!(
                            msg,
                            "\x20 -> {section}, block {wb} word {ww}: \
                             expected {wv:#018x}, got {gv:#018x}\n\
                             \x20    (positions agree: {})\n",
                            wb == gb && ww == gw
                        );
                    }
                    msg.push_str(
                        "\nThis is a real divergence in the port, not a test problem. \
                         Fix the implementation, do not loosen this test.\n",
                    );
                    panic!("{msg}");
                }

                if let Some(rest) = got.strip_prefix(" After pass ") {
                    section = format!("pass {}", rest.trim_end_matches(':'));
                }
            }
            (Some(want), None) => panic!(
                "{label}: our trace ended early after {line_no} lines; \
                 the golden file still has {want:?}"
            ),
            (None, Some(got)) => {
                panic!("{label}: our trace has extra output after {line_no} lines: {got:?}")
            }
            (None, None) => break,
        }
    }

    assert_eq!(
        line_no, EXPECTED_LINES,
        "{label}: expected {EXPECTED_LINES} lines in the golden file"
    );
    // `lines()` is newline-agnostic, so pin the raw bytes too. Only a trailing
    // newline difference can survive the loop above, hence the short message.
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: every line matches but the byte lengths differ \
         (missing or extra trailing newline?)"
    );
    assert!(
        actual == expected,
        "{label}: byte-for-byte comparison failed even though every line matched"
    );
}

/// Read `phc-winner-argon2/kats/<name>`.
fn read_golden(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("phc-winner-argon2")
        .join("kats")
        .join(name);
    fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "cannot read the golden KAT {}: {e}\n\
             The C reference checkout must be present at phc-winner-argon2/.",
            path.display()
        )
    })
}

/// The backends this CPU can really execute, always including `Scalar`.
///
/// Taken from [`Backend::ALL`] rather than a local list, so a backend added to
/// the crate is KAT-checked without anyone remembering to edit this file.
fn runnable_backends() -> Vec<Backend> {
    Backend::ALL
        .into_iter()
        .filter(|b| b.is_available())
        .collect()
}

/// Regenerate `kats/<file>` on every runnable backend and diff it.
fn check_golden_file(file: &str, algorithm: Algorithm, version: Version) {
    let expected = read_golden(file);

    let backends = runnable_backends();
    assert!(
        backends.contains(&Backend::Scalar),
        "Scalar must always be available"
    );

    for backend in backends {
        // SAFETY: `runnable_backends()` filtered on `Backend::is_available()`,
        // so every backend here is one this CPU advertises.
        let actual = unsafe { generate(algorithm, version, backend) };
        assert_trace_matches(&format!("kats/{file} [{backend}]"), &expected, &actual);
    }
}

// ---------------------------------------------------------------------------
// One test per golden file
// ---------------------------------------------------------------------------

#[test]
fn kat_argon2d_v0x13() {
    check_golden_file("argon2d", Algorithm::Argon2d, Version::V0x13);
}

#[test]
fn kat_argon2i_v0x13() {
    check_golden_file("argon2i", Algorithm::Argon2i, Version::V0x13);
}

#[test]
fn kat_argon2id_v0x13() {
    check_golden_file("argon2id", Algorithm::Argon2id, Version::V0x13);
}

#[test]
fn kat_argon2d_v0x10() {
    check_golden_file("argon2d_v16", Algorithm::Argon2d, Version::V0x10);
}

#[test]
fn kat_argon2i_v0x10() {
    check_golden_file("argon2i_v16", Algorithm::Argon2i, Version::V0x10);
}

#[test]
fn kat_argon2id_v0x10() {
    check_golden_file("argon2id_v16", Algorithm::Argon2id, Version::V0x10);
}

// ---------------------------------------------------------------------------
// Cross-checks on the KAT parameters
// ---------------------------------------------------------------------------

/// The public API (no `__internal`) must produce the same tag as the trace.
///
/// This is what a user actually calls, and it is the one place `secret` and
/// `ad` support is load-bearing: without `hash_into_with_ad` the KAT
/// parameters would be unreachable from outside the crate.
#[test]
fn public_api_reproduces_every_kat_tag() {
    for (file, algorithm, version) in [
        ("argon2d", Algorithm::Argon2d, Version::V0x13),
        ("argon2i", Algorithm::Argon2i, Version::V0x13),
        ("argon2id", Algorithm::Argon2id, Version::V0x13),
        ("argon2d_v16", Algorithm::Argon2d, Version::V0x10),
        ("argon2i_v16", Algorithm::Argon2i, Version::V0x10),
        ("argon2id_v16", Algorithm::Argon2id, Version::V0x10),
    ] {
        let golden = read_golden(file);
        let tag_line = golden
            .lines()
            .next_back()
            .expect("golden file is not empty")
            .strip_prefix("Tag: ")
            .expect("the last line of a golden file is the tag");
        let expected: Vec<u8> = tag_line
            .split_whitespace()
            .map(|byte| u8::from_str_radix(byte, 16).expect("hex byte"))
            .collect();
        assert_eq!(expected.len(), OUTLEN);

        let params = Params::new_with_threads(M_COST, T_COST, LANES, LANES, OUTLEN)
            .expect("genkat.c's parameters are valid");
        let mut tag = [0u8; OUTLEN];
        Argon2::new(algorithm, version, params)
            .hash_into_with_ad(&PWD, &SALT, &SECRET, &AD, &mut tag)
            .expect("hashing cannot fail here");

        assert_eq!(
            tag.as_slice(),
            expected.as_slice(),
            "kats/{file}: public-API tag differs from the golden tag"
        );
    }
}

/// The same six golden tags, through **one reused arena** instead of six fresh
/// ones.
///
/// [`public_api_reproduces_every_kat_tag`] gives each file its own
/// `alloc_zeroed` arena, which is the one situation where arena reuse cannot
/// possibly be wrong. This runs all six through a single `Argon2::hasher()`, in
/// a rotating order, four times over — so each file's hash lands on memory that
/// the *previous* file's hash filled with a different algorithm's or a
/// different version's derived material.
///
/// The parameters are identical across the six (`t=3 m=32 lanes=4`), so the
/// arena is allocated exactly once and every hash after the first runs dirty.
/// That is asserted, not assumed: without it this would silently degrade into
/// six independent hashes and prove nothing beyond the test above.
///
/// # What this cannot cover, and where that coverage lives
///
/// `Hasher` always takes the backend from runtime detection — that is what
/// makes it safe — so this reaches `Neon` natively and `Sse2` under Rosetta,
/// but never `Scalar` or a forced `Avx2`. There is no pooled equivalent of
/// [`hash_traced`], either, so this compares tags rather than the whole
/// block-level trace. Both gaps are covered inside the crate by
/// `core::tests::a_pooled_hash_reproduces_the_one_shot_arena_word_for_word`,
/// which drives the pooled path over every available backend and compares every
/// word of every block at every pass boundary.
#[test]
fn a_pooled_hasher_reproduces_every_kat_tag() {
    let files = [
        ("argon2d", Algorithm::Argon2d, Version::V0x13),
        ("argon2i", Algorithm::Argon2i, Version::V0x13),
        ("argon2id", Algorithm::Argon2id, Version::V0x13),
        ("argon2d_v16", Algorithm::Argon2d, Version::V0x10),
        ("argon2i_v16", Algorithm::Argon2i, Version::V0x10),
        ("argon2id_v16", Algorithm::Argon2id, Version::V0x10),
    ];

    let expected: Vec<Vec<u8>> = files.iter().map(|(file, _, _)| golden_tag(file)).collect();

    let params = Params::new_with_threads(M_COST, T_COST, LANES, LANES, OUTLEN)
        .expect("genkat.c's parameters are valid");
    let blocks = params.memory_blocks() as usize;

    let mut hasher = Argon2::new(Algorithm::Argon2id, Version::V0x13, params).hasher();
    assert_eq!(hasher.reserved_blocks(), 0, "nothing allocated up front");

    let mut hashes = 0usize;
    for round in 0..4 {
        // Rotate the starting point so consecutive hashes are (almost) never
        // the same file, and every file follows every other one across rounds.
        for step in 0..files.len() {
            let index = (step + round) % files.len();
            let (file, algorithm, version) = files[index];

            hasher.set_argon2(Argon2::new(algorithm, version, params));
            let mut tag = [0u8; OUTLEN];
            hasher
                .hash_into_with_ad(&PWD, &SALT, &SECRET, &AD, &mut tag)
                .expect("hashing genkat.c's parameters cannot fail");
            hashes += 1;

            assert_eq!(
                tag.as_slice(),
                expected[index].as_slice(),
                "kats/{file} [pooled, round {round}]: the tag from a reused arena \
                 differs from the golden tag.\n\
                 This is a real bug in the arena-reuse layer, not a test problem.\n\
                 \x20 golden (C): {}\n\
                 \x20 ours:       {}",
                expected[index]
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>(),
                tag.iter().map(|b| format!("{b:02x}")).collect::<String>(),
            );

            // One allocation for all 24 hashes: the parameters never change, so
            // every hash but the first must have run on a dirty arena.
            assert_eq!(
                hasher.reserved_blocks(),
                blocks,
                "kats/{file} [pooled, round {round}]: the arena was resized, so \
                 this hash did not run on a reused arena"
            );
        }
    }
    assert_eq!(hashes, 24);
}

/// The `Tag: ` line at the bottom of a golden file, as bytes.
fn golden_tag(file: &str) -> Vec<u8> {
    let golden = read_golden(file);
    let tag_line = golden
        .lines()
        .next_back()
        .expect("golden file is not empty")
        .strip_prefix("Tag: ")
        .expect("the last line of a golden file is the tag");
    let tag: Vec<u8> = tag_line
        .split_whitespace()
        .map(|byte| u8::from_str_radix(byte, 16).expect("hex byte"))
        .collect();
    assert_eq!(tag.len(), OUTLEN, "kats/{file}: unexpected tag length");
    tag
}

/// Negative control: prove the comparator actually detects a divergence.
///
/// A trace test that silently compares nothing is worse than no test at all,
/// so flip one bit of one word of one block and check that
/// `assert_trace_matches` panics *and* names the right pass, block and word.
///
/// Not on wasi: the check works by `catch_unwind`, and wasi is panic=abort.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn the_comparator_catches_a_single_flipped_bit() {
    let golden = read_golden("argon2i");

    // "Block 0017 [ 42]: ..." in pass 1 — a middle-of-the-arena word.
    let target = golden
        .lines()
        .enumerate()
        .filter(|(_, line)| line.starts_with("Block 0017 [ 42]: "))
        .nth(1)
        .map(|(index, line)| (index, line.to_string()))
        .expect("pass 1 dumps block 17 word 42");

    let (_, original) = &target;
    let (block, word, value) = parse_block_line(original).expect("a block line");
    assert_eq!((block, word), (17, 42));

    let corrupted_line = format!("Block {block:04} [{word:3}]: {:016x}", value ^ 1);
    let corrupted: String = golden
        .lines()
        .enumerate()
        .map(|(index, line)| {
            if index == target.0 {
                format!("{corrupted_line}\n")
            } else {
                format!("{line}\n")
            }
        })
        .collect();

    let report = std::panic::catch_unwind(|| {
        assert_trace_matches("negative-control", &golden, &corrupted);
    })
    .expect_err("a flipped bit must fail the comparison");

    let message = report
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| report.downcast_ref::<&str>().copied())
        .unwrap_or("");

    assert!(
        message.contains("pass 1") && message.contains("block 17") && message.contains("word 42"),
        "the failure message must localise the divergence, got:\n{message}"
    );
}

/// Every SIMD backend this CPU can run must be covered by the golden files.
///
/// Guards against `runnable_backends()` quietly shrinking to `[Scalar]` and the
/// KAT suite then proving nothing about the SIMD paths.
#[test]
fn every_runnable_backend_is_kat_checked() {
    let backends = runnable_backends();
    println!("KAT backends on this host: {backends:?}");

    assert!(backends.contains(&Backend::Scalar));
    assert!(
        backends.contains(&argon2_rust::detected_backend()),
        "the backend runtime detection picked is not in the KAT list"
    );

    #[cfg(target_arch = "aarch64")]
    assert!(
        backends.contains(&Backend::Neon),
        "NEON is mandatory on aarch64, so the KATs must exercise it"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    assert!(
        backends.contains(&Backend::Sse2),
        "SSE2 is baseline on x86_64, so the KATs must exercise it"
    );
}

/// AVX2 KATs on a host whose `cpuid` hides AVX2 — **deliberately bypasses
/// runtime detection**.
///
/// # Why this exists
///
/// Rosetta 2 on this machine *executes* AVX2 correctly but does not advertise
/// it, so `is_x86_feature_detected!("avx2")` is `false` and
/// [`runnable_backends`] never lists [`Backend::Avx2`]. That is the **correct**
/// behaviour for the library — a backend the CPU does not claim must not be
/// selected — but it means the normal KAT run proves nothing about the AVX2
/// code on this host.
///
/// This test calls [`hash_traced`] with `Backend::Avx2` regardless, which
/// resolves straight to `avx2::fill_segment`.
///
/// # Two ways to get AVX2 covered, and which to prefer
///
/// ```text
/// # 1. SOUND. `+avx2` makes is_x86_feature_detected! short-circuit to true,
/// #    so Avx2 becomes genuinely available and *every* test above covers it.
/// RUSTFLAGS="-C target-feature=+avx2" \
///   cargo test --test kat --features internal-api --release \
///   --target x86_64-apple-darwin
///
/// # 2. THIS TEST. No RUSTFLAGS; calls a #[target_feature(enable="avx2")] fn
/// #    without the feature detected. Only works because the hardware really
/// #    runs the instructions — a test-only affordance, never a library path.
/// cargo test --test kat --features internal-api --release \
///   --target x86_64-apple-darwin -- --ignored forced_avx2
/// ```
///
/// Route 1 is preferred and is what CI should run; it is UB-free and covers all
/// six files on every test. This one is `#[ignore]`d because route 2 is only
/// defensible on a host measured to execute AVX2, which is not something the
/// test can check for itself — `is_available()` returning `false` is precisely
/// the situation it is working around.
///
/// AVX-512 gets no equivalent: it traps with `SIGILL` under Rosetta even when
/// forced, so `avx512.rs` is compile-verified only.
#[test]
#[ignore = "bypasses CPU feature detection; only run on a host measured to execute AVX2"]
#[cfg(target_arch = "x86_64")]
fn forced_avx2_reproduces_every_golden_file() {
    if Backend::Avx2.is_available() {
        println!(
            "AVX2 is advertised here, so the normal KATs already cover it; \
             this test is redundant but still valid."
        );
    } else {
        println!(
            "AVX2 is NOT advertised by cpuid on this host. Forcing it anyway — \
             this is only safe because the hardware executes AVX2 (Rosetta 2)."
        );
    }

    for (file, algorithm, version) in [
        ("argon2d", Algorithm::Argon2d, Version::V0x13),
        ("argon2i", Algorithm::Argon2i, Version::V0x13),
        ("argon2id", Algorithm::Argon2id, Version::V0x13),
        ("argon2d_v16", Algorithm::Argon2d, Version::V0x10),
        ("argon2i_v16", Algorithm::Argon2i, Version::V0x10),
        ("argon2id_v16", Algorithm::Argon2id, Version::V0x10),
    ] {
        let expected = read_golden(file);
        // SAFETY: the deliberate detection bypass this test exists for. Sound
        // only because this host was *measured* to execute AVX2 while Rosetta 2
        // hides it from `cpuid` — hence the `#[ignore]`, which keeps it off any
        // machine that has not been measured. Never a library path.
        let actual = unsafe { generate(algorithm, version, Backend::Avx2) };
        assert_trace_matches(&format!("kats/{file} [forced avx2]"), &expected, &actual);
    }
}

/// `threads` is a performance knob; it must not move a single word of the
/// arena. Run the whole `argon2id` trace single-threaded and diff it against
/// the file `genkat` produced with `threads = lanes = 4`.
#[test]
fn single_threaded_trace_equals_the_four_thread_golden_file() {
    let expected = read_golden("argon2id");

    let params = Params::new_with_threads(M_COST, T_COST, LANES, /* threads */ 1, OUTLEN)
        .expect("genkat.c's parameters are valid");

    let mut tag = [0u8; OUTLEN];
    let mut passes = String::with_capacity(330 * 1024);
    let mut trace = |pass: u32, blocks: &[Block]| {
        let _ = write!(passes, "\n After pass {pass}:\n");
        for (i, block) in blocks.iter().enumerate() {
            for j in 0..QWORDS_IN_BLOCK {
                let _ = writeln!(passes, "Block {i:04} [{j:3}]: {:016x}", block[j]);
            }
        }
    };

    // SAFETY: `detected_backend()` is the cached result of runtime CPU feature
    // detection, so this CPU can execute it by construction.
    let h0 = unsafe {
        hash_traced(
            argon2_rust::detected_backend(),
            Algorithm::Argon2id,
            Version::V0x13,
            &params,
            &PWD,
            &SALT,
            &SECRET,
            &AD,
            &mut tag,
            Some(&mut trace as PassTrace),
        )
    }
    .expect("hashing cannot fail here");

    let mut actual = String::with_capacity(passes.len() + 1024);
    actual.push_str("=======================================\n");
    let _ = writeln!(actual, "Argon2id version number 19");
    actual.push_str("=======================================\n");
    let _ = writeln!(
        actual,
        "Memory: {M_COST} KiB, Iterations: {T_COST}, Parallelism: {LANES} lanes, \
         Tag length: {OUTLEN} bytes"
    );
    let _ = write!(actual, "Password[{}]: ", PWD.len());
    push_hex_bytes(&mut actual, &PWD);
    let _ = write!(actual, "Salt[{}]: ", SALT.len());
    push_hex_bytes(&mut actual, &SALT);
    let _ = write!(actual, "Secret[{}]: ", SECRET.len());
    push_hex_bytes(&mut actual, &SECRET);
    let _ = write!(actual, "Associated data[{}]: ", AD.len());
    push_hex_bytes(&mut actual, &AD);
    actual.push_str("Pre-hashing digest: ");
    push_hex_bytes(&mut actual, &h0);
    actual.push_str(&passes);
    actual.push_str("Tag: ");
    push_hex_bytes(&mut actual, &tag);

    assert_trace_matches("kats/argon2id [threads=1]", &expected, &actual);
}
