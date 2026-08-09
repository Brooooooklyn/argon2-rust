//! CodSpeed benchmarks for `argon2-rust`.
//!
//! ```text
//! cargo codspeed build --bench codspeed
//! codspeed run --mode simulation -- cargo codspeed run --bench codspeed
//! ```
//!
//! # Why this file exists next to `benches/argon2.rs`
//!
//! `benches/argon2.rs` is the wall-clock harness: it sweeps 64 MiB and 256 MiB
//! working sets, spawns `lanes` OS threads, prints hand-rolled interleaved
//! ratio tables, and compares against a locally built `phc-winner-argon2`. All
//! of that is deliberate, and none of it survives the move to CI:
//!
//! * CodSpeed's simulation instrument counts instructions, cache accesses and
//!   branches inside a CPU model, so a benchmark's cost is its *work*, not its
//!   wall-clock time. A 256 MiB arena buys no extra signal, only minutes.
//! * The instrument serialises threads, so a spin barrier between workers
//!   measures the scheduler rather than the algorithm. Every configuration here
//!   therefore keeps `threads = 1` while still varying `lanes`, which exercises
//!   the multi-lane addressing and segment ordering without any thread ever
//!   being spawned (`fill_pooled` takes the single-worker path when
//!   `min(threads, lanes) == 1`).
//! * The `vs_c` group needs a C toolchain and a vendored checkout. A ratio
//!   against an external build is not something CI can track over time.
//!
//! So this file is the regression net: small, hermetic, dependency-free, and
//! covering every layer that a change to this crate can plausibly move — the
//! fill kernel, the runtime dispatch, the BLAKE2b primitive, PHC encoding, the
//! arena reuse path, and the public one-shot API.
//!
//! # Groups
//!
//! | Group | What it covers |
//! |---|---|
//! | `hash` | `Argon2::hash_into` over the m_cost x t_cost x lanes sweep. |
//! | `variants` | Argon2id vs Argon2i vs Argon2d at one matched point. |
//! | `versions` | v0x13 vs the legacy v0x10 indexing. |
//! | `lanes` | Multi-lane addressing at fixed total memory. |
//! | `backends` | Every fill backend this CPU advertises, at one matched point. |
//! | `reuse` | A fresh arena per hash vs a pooled one kept in a `Hasher`. |
//! | `encoded` | `hash_encoded` / `verify_encoded`, the PHC round trip. |
//! | `encoding` | The string layer alone: base64, encode, decode. |
//! | `blake2b` | The hash primitive and Argon2's variable-length `H'`. |
//! | `dispatch` | The cached backend load on the hot path. |

use argon2_rust::__internal::{
    Backend, backend, blake2b, blake2b_long, decode_string, encode_string_alloc, from_base64,
    hash_with_backend, to_base64,
};
use argon2_rust::{Algorithm, Argon2, Params, Version};
use codspeed_criterion_compat::{
    BenchmarkId, Criterion, black_box, criterion_group, criterion_main,
};

/// Tag length, the RFC 9106 recommendation and this crate's default.
const OUTLEN: usize = 32;

const PWD: &[u8] = b"correct horse battery staple";
/// 16 bytes, the RFC 9106 recommended salt length.
const SALT: &[u8] = b"codspeed-salt-16";

/// Build parameters, pinning `threads` to 1 — see the module docs.
///
/// `lanes` still varies: it changes the tag and the memory layout, and is the
/// knob worth tracking. The thread count is a pure wall-clock knob that the
/// simulation instrument cannot observe honestly.
fn params(m_cost: u32, t_cost: u32, lanes: u32) -> Params {
    match Params::new_with_threads(m_cost, t_cost, lanes, 1, OUTLEN) {
        Ok(p) => p,
        Err(e) => panic!("bad bench params m={m_cost} t={t_cost} p={lanes}: {e:?}"),
    }
}

fn argon2id(m_cost: u32, t_cost: u32, lanes: u32) -> Argon2 {
    Argon2::new(
        Algorithm::Argon2id,
        Version::V0x13,
        params(m_cost, t_cost, lanes),
    )
}

// ---------------------------------------------------------------------------
// 1. The cost-parameter sweep
// ---------------------------------------------------------------------------

/// `(m_cost KiB, t_cost, lanes)`, ordered cheapest first.
///
/// `m8_t1_p1` is the smallest hash Argon2 admits and is almost entirely
/// BLAKE2b and fixed cost; `m4096_t3_p4` is dominated by the fill kernel. A
/// change that only moves one end of that range shows up here as exactly that.
const GRID: &[(u32, u32, u32)] = &[
    (8, 1, 1),
    (64, 1, 1),
    (64, 3, 1),
    (512, 1, 1),
    (512, 1, 4),
    (512, 3, 1),
    (4096, 1, 1),
    (4096, 1, 4),
    (4096, 3, 4),
];

fn bench_hash(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash");

    for &(m_cost, t_cost, lanes) in GRID {
        let argon2 = argon2id(m_cost, t_cost, lanes);
        let mut out = [0u8; OUTLEN];
        group.bench_function(format!("m{m_cost}_t{t_cost}_p{lanes}"), |b| {
            b.iter(|| {
                argon2
                    .hash_into(black_box(PWD), black_box(SALT), &mut out)
                    .expect("hash");
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. Variants and versions
// ---------------------------------------------------------------------------

/// Argon2i pays for `next_addresses` on every pass, Argon2id only on the first
/// half of pass 0, Argon2d never. This group is the data-independent addressing
/// overhead, isolated at one matched point.
fn bench_variants(c: &mut Criterion) {
    let mut group = c.benchmark_group("variants");
    let p = params(1024, 2, 1);

    for algorithm in [Algorithm::Argon2id, Algorithm::Argon2i, Algorithm::Argon2d] {
        let argon2 = Argon2::new(algorithm, Version::V0x13, p);
        let mut out = [0u8; OUTLEN];
        group.bench_function(algorithm.as_str(), |b| {
            b.iter(|| {
                argon2
                    .hash_into(black_box(PWD), black_box(SALT), &mut out)
                    .expect("hash");
            });
        });
    }

    group.finish();
}

/// v0x10 and v0x13 differ in the XOR-vs-overwrite rule on passes after the
/// first, so this pair needs `t_cost > 1` to mean anything.
fn bench_versions(c: &mut Criterion) {
    let mut group = c.benchmark_group("versions");
    let p = params(1024, 2, 1);

    for version in [Version::V0x13, Version::V0x10] {
        let argon2 = Argon2::new(Algorithm::Argon2id, version, p);
        let mut out = [0u8; OUTLEN];
        group.bench_function(format!("v{:#x}", version.as_u32()), |b| {
            b.iter(|| {
                argon2
                    .hash_into(black_box(PWD), black_box(SALT), &mut out)
                    .expect("hash");
            });
        });
    }

    group.finish();
}

/// Lanes at *fixed total memory*: the fill does the same amount of block work
/// in every row, so what moves is the segment ordering and `index_alpha`'s
/// cross-lane reference window, not the size of the arena.
fn bench_lanes(c: &mut Criterion) {
    let mut group = c.benchmark_group("lanes");

    for lanes in [1u32, 2, 4, 8] {
        let argon2 = argon2id(2048, 1, lanes);
        let mut out = [0u8; OUTLEN];
        group.bench_function(format!("p{lanes}"), |b| {
            b.iter(|| {
                argon2
                    .hash_into(black_box(PWD), black_box(SALT), &mut out)
                    .expect("hash");
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. The fill backends
// ---------------------------------------------------------------------------

/// One row per backend this CPU advertises, so a change to a single SIMD kernel
/// is attributable instead of being averaged into the dispatched hash.
///
/// The list is `Backend::ALL` filtered by [`Backend::is_available`] — the same
/// runtime cascade the library uses. Under the simulation instrument the CPU
/// model decides what `cpuid` reports, so a backend it cannot execute is never
/// advertised and never reaches `hash_with_backend`. Which rows exist therefore
/// follows the runner, and the runner is pinned in CI.
fn bench_backends(c: &mut Criterion) {
    let mut group = c.benchmark_group("backends");
    let p = params(1024, 1, 1);

    for backend in Backend::ALL.into_iter().filter(|b| b.is_available()) {
        let mut out = [0u8; OUTLEN];
        group.bench_function(BenchmarkId::new(backend.name(), "m1024_t1_p1"), |b| {
            b.iter(|| {
                // SAFETY: `is_available()` above is the runtime CPU cascade, so
                // this backend's `#[target_feature]` requirements are met on
                // this machine. That is exactly `hash_with_backend`'s contract.
                unsafe {
                    hash_with_backend(
                        black_box(backend),
                        Algorithm::Argon2id,
                        Version::V0x13,
                        &p,
                        PWD,
                        SALT,
                        &[],
                        &[],
                        &mut out,
                    )
                }
                .expect("hash");
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Arena reuse
// ---------------------------------------------------------------------------

/// One-shot `Argon2` (map, fault, wipe, unmap every call) against a `Hasher`
/// that parks the arena between calls.
///
/// The simulation instrument does not charge page faults or `mmap`, which is
/// where most of the wall-clock win lives; what it *does* show is the wipe and
/// the recycle path, so this group guards the reuse bookkeeping rather than
/// reproducing the wall-clock delta from the crate docs.
fn bench_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("reuse");

    for m_cost in [1024u32, 8192] {
        let argon2 = argon2id(m_cost, 1, 1);
        let mut out = [0u8; OUTLEN];

        group.bench_function(BenchmarkId::new("oneshot", format!("m{m_cost}")), |b| {
            b.iter(|| {
                argon2
                    .hash_into(black_box(PWD), black_box(SALT), &mut out)
                    .expect("hash");
            });
        });

        let mut hasher = argon2.hasher();
        hasher.reserve().expect("reserve");
        group.bench_function(BenchmarkId::new("pooled", format!("m{m_cost}")), |b| {
            b.iter(|| {
                hasher
                    .hash_into(black_box(PWD), black_box(SALT), &mut out)
                    .expect("hash");
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 5. The PHC string API
// ---------------------------------------------------------------------------

/// The two calls an application actually makes, at a deliberately small
/// `m_cost` so the encoding and decoding work is a visible share of the total
/// rather than rounding error on top of the fill.
fn bench_encoded(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoded");
    let argon2 = argon2id(64, 1, 1);
    let encoded = argon2.hash_encoded(PWD, SALT).expect("hash_encoded");

    group.bench_function("hash_encoded", |b| {
        b.iter(|| {
            black_box(
                argon2
                    .hash_encoded(black_box(PWD), black_box(SALT))
                    .expect("hash_encoded"),
            );
        });
    });

    group.bench_function("verify_encoded", |b| {
        b.iter(|| {
            Argon2::verify_encoded(black_box(&encoded), black_box(PWD), Algorithm::Argon2id)
                .expect("verify_encoded");
        });
    });

    group.bench_function("verify_encoded_mismatch", |b| {
        b.iter(|| {
            let _ = black_box(Argon2::verify_encoded(
                black_box(&encoded),
                black_box(b"wrong password".as_slice()),
                Algorithm::Argon2id,
            ));
        });
    });

    group.finish();
}

/// The string layer with no hashing at all. These are nanosecond-scale, which
/// is precisely the range wall-clock benchmarking cannot resolve and the
/// simulation instrument can.
fn bench_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoding");
    let p = params(65536, 3, 4);
    let tag = [0x5au8; OUTLEN];
    let encoded = encode_string_alloc(Algorithm::Argon2id, Version::V0x13, &p, SALT, &tag)
        .expect("encode_string_alloc");

    group.bench_function("encode_string_alloc", |b| {
        b.iter(|| {
            black_box(
                encode_string_alloc(
                    Algorithm::Argon2id,
                    Version::V0x13,
                    black_box(&p),
                    black_box(SALT),
                    black_box(&tag),
                )
                .expect("encode_string_alloc"),
            );
        });
    });

    group.bench_function("decode_string", |b| {
        b.iter(|| {
            black_box(decode_string(black_box(&encoded), Algorithm::Argon2id).expect("decode"));
        });
    });

    // 1 KiB in, so the inner loop dominates the call overhead.
    let raw = [0xa5u8; 1024];
    let mut b64 = [0u8; 1408];
    let b64_len = to_base64(&mut b64, &raw).expect("to_base64");
    let b64 = &b64[..b64_len];

    group.bench_function("to_base64_1kib", |b| {
        let mut dst = [0u8; 1408];
        b.iter(|| {
            black_box(to_base64(&mut dst, black_box(&raw)).expect("to_base64"));
        });
    });

    group.bench_function("from_base64_1kib", |b| {
        let mut dst = [0u8; 1024];
        b.iter(|| {
            black_box(from_base64(&mut dst, black_box(b64)).expect("from_base64"));
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 6. The BLAKE2b primitive
// ---------------------------------------------------------------------------

/// Argon2 spends its whole fixed cost in BLAKE2b: the initial hash, the two
/// 1024-byte `H'` expansions that seed each lane, and the final tag.
///
/// `blake2b_long` with a 1024-byte output is the exact shape `fill_first_blocks`
/// calls, and the 64-byte case is the plain single-call path.
fn bench_blake2b(c: &mut Criterion) {
    let mut group = c.benchmark_group("blake2b");

    for &len in &[64usize, 1024, 65536] {
        let input = vec![0x3cu8; len];
        let mut out = [0u8; 64];
        group.bench_function(BenchmarkId::new("blake2b", len), |b| {
            b.iter(|| {
                blake2b(&mut out, black_box(&input)).expect("blake2b");
            });
        });
    }

    let input = [0x3cu8; 72];
    let mut block = vec![0u8; 1024];
    group.bench_function("blake2b_long_1024", |b| {
        b.iter(|| {
            blake2b_long(&mut block, black_box(&input)).expect("blake2b_long");
        });
    });

    let mut short = [0u8; 32];
    group.bench_function("blake2b_long_32", |b| {
        b.iter(|| {
            blake2b_long(&mut short, black_box(&input)).expect("blake2b_long");
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 7. Runtime dispatch
// ---------------------------------------------------------------------------

/// The per-call cost of asking which backend to use, once the cascade has run
/// and cached its answer. This is one relaxed atomic load on the hot path, and
/// it is here so that it stays one.
fn bench_dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("dispatch");

    // Warm the cache so the first sample is not the whole cascade.
    black_box(backend());

    group.bench_function("cached_backend", |b| {
        b.iter(|| black_box(backend()));
    });

    group.bench_function("detected_backend", |b| {
        b.iter(|| black_box(argon2_rust::detected_backend()));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_hash,
    bench_variants,
    bench_versions,
    bench_lanes,
    bench_backends,
    bench_reuse,
    bench_encoded,
    bench_encoding,
    bench_blake2b,
    bench_dispatch,
);
criterion_main!(benches);
