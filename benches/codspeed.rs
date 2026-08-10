//! Stable CodSpeed benchmarks for the crate's major public APIs.
//!
//! ```text
//! cargo codspeed build --bench codspeed
//! cargo codspeed run --bench codspeed
//! ```
//!
//! Keep this suite deliberately small. CodSpeed runs it on every pull request,
//! and each benchmark id becomes part of the long-term performance history.
//! Internal kernels, dispatch, and BLAKE2b are therefore measured through the
//! public methods that use them instead of receiving separate
//! implementation-specific benchmarks. PHC parsing and serialization are the
//! exception: the public methods also run Argon2, whose memory fill hides codec
//! regressions, so the production PHC codec is isolated as a whole below.
//!
//! Every configuration uses one worker thread. CodSpeed's CPU simulation does
//! not model wall-clock parallel speedup, and this crate's multi-threaded fill
//! uses a spin barrier; benchmarking that path under a serialized simulator
//! would measure scheduling artifacts. The existing Criterion suite remains
//! the source for native multi-threaded and backend-specific measurements.

use std::time::Duration;

use argon2_rust::__internal::{decode_string, encode_string_alloc};
use argon2_rust::{Algorithm, Argon2, Params, Version};
use codspeed_criterion_compat::{
    BenchmarkId, Criterion, black_box, criterion_group, criterion_main,
};

const PASSWORD: &[u8] = b"correct horse battery staple";
const SALT: &[u8] = b"codspeed-salt-16";
const SECRET: &[u8] = b"server-side pepper";
const AD: &[u8] = b"argon2-rust benchmark";
const OUT_LEN: usize = 32;
const TAG: [u8; OUT_LEN] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
];

fn params(m_cost: u32) -> Params {
    Params::new_with_threads(m_cost, 1, 1, 1, OUT_LEN).expect("benchmark parameters must be valid")
}

fn argon2id(m_cost: u32) -> Argon2 {
    Argon2::new(Algorithm::Argon2id, Version::V0x13, params(m_cost))
}

/// Exercise all three public algorithms through the caller-owned-output API.
/// A 1 MiB arena is large enough for the fill schedule to dominate without
/// making one simulated CI run needlessly expensive.
fn bench_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("public/algorithm/hash_into");
    let p = params(1024);

    for algorithm in [Algorithm::Argon2d, Algorithm::Argon2i, Algorithm::Argon2id] {
        let argon2 = Argon2::new(algorithm, Version::V0x13, p);
        let mut out = [0u8; OUT_LEN];

        // Resolve runtime dispatch before CodSpeed enters the measured closure.
        argon2
            .hash_into(PASSWORD, SALT, &mut out)
            .expect("warm-up hash");

        group.bench_function(BenchmarkId::from_parameter(algorithm.as_str()), |b| {
            b.iter(|| {
                argon2
                    .hash_into(black_box(PASSWORD), black_box(SALT), &mut out)
                    .expect("hash_into");
                black_box(&out);
            });
        });
    }

    group.finish();
}

/// The PHC codec behind the public encoded/password APIs, without an Argon2
/// fill masking its cost. This is one realistic record: the crate defaults, a
/// 16-byte salt, and a 32-byte tag. Direct Base64 throughput remains covered by
/// `benches/base64.rs`; these cases measure the complete production format.
fn bench_phc_codec(c: &mut Criterion) {
    let mut group = c.benchmark_group("public/phc_codec");
    let p = Params::default();
    let encoded = encode_string_alloc(Algorithm::Argon2id, Version::V0x13, &p, SALT, &TAG)
        .expect("PHC string");

    // Resolve runtime dispatch for both directions before measurement.
    decode_string(&encoded, Algorithm::Argon2id).expect("PHC parse warm-up");

    group.bench_function("serialize", |b| {
        b.iter(|| {
            black_box(
                encode_string_alloc(
                    Algorithm::Argon2id,
                    Version::V0x13,
                    black_box(&p),
                    black_box(SALT),
                    black_box(&TAG),
                )
                .expect("PHC serialization"),
            );
        });
    });

    group.bench_function("parse", |b| {
        b.iter(|| {
            black_box(
                decode_string(black_box(&encoded), Algorithm::Argon2id).expect("PHC parsing"),
            );
        });
    });

    group.finish();
}

/// Major one-shot `Argon2` entry points at a small valid memory cost. Keeping
/// the fill short makes allocation, BLAKE2b, PHC encoding, and bounded-decoder
/// changes visible instead of rounding them away under a production-size fill.
fn bench_one_shot(c: &mut Criterion) {
    let mut group = c.benchmark_group("public/one_shot");
    let p = params(64);
    let argon2 = argon2id(64);
    let expected = argon2.hash(PASSWORD, SALT).expect("raw tag");
    let encoded = argon2.hash_encoded(PASSWORD, SALT).expect("PHC string");

    group.bench_function("hash_allocating", |b| {
        b.iter(|| {
            black_box(
                argon2
                    .hash(black_box(PASSWORD), black_box(SALT))
                    .expect("hash"),
            );
        });
    });

    let mut out = [0u8; OUT_LEN];
    group.bench_function("hash_into_with_secret_and_ad", |b| {
        b.iter(|| {
            argon2
                .hash_into_with_ad(
                    black_box(PASSWORD),
                    black_box(SALT),
                    black_box(SECRET),
                    black_box(AD),
                    &mut out,
                )
                .expect("keyed hash");
            black_box(&out);
        });
    });

    group.bench_function("verify_raw", |b| {
        b.iter(|| {
            argon2
                .verify(black_box(PASSWORD), black_box(SALT), black_box(&expected))
                .expect("verify raw tag");
        });
    });

    group.bench_function("hash_encoded", |b| {
        b.iter(|| {
            black_box(
                argon2
                    .hash_encoded(black_box(PASSWORD), black_box(SALT))
                    .expect("hash encoded"),
            );
        });
    });

    group.bench_function("verify_encoded", |b| {
        b.iter(|| {
            Argon2::verify_encoded(
                black_box(&encoded),
                black_box(PASSWORD),
                Algorithm::Argon2id,
            )
            .expect("verify encoded");
        });
    });

    group.bench_function("verify_encoded_bounded", |b| {
        b.iter(|| {
            Argon2::verify_encoded_bounded(
                black_box(&encoded),
                black_box(PASSWORD),
                Algorithm::Argon2id,
                black_box(&p),
            )
            .expect("bounded verify");
        });
    });

    group.finish();
}

/// Steady-state `Hasher` entry points. Each hasher reserves its arena before
/// measurement so the benchmark records reuse rather than the first allocation.
fn bench_pooled(c: &mut Criterion) {
    let mut group = c.benchmark_group("public/pooled");
    let p = params(64);
    let argon2 = argon2id(64);
    let encoded = argon2.hash_encoded(PASSWORD, SALT).expect("PHC string");

    let mut raw = argon2.hasher();
    raw.reserve().expect("reserve");
    let mut out = [0u8; OUT_LEN];
    group.bench_function("hash_into", |b| {
        b.iter(|| {
            raw.hash_into(black_box(PASSWORD), black_box(SALT), &mut out)
                .expect("pooled hash_into");
            black_box(&out);
        });
    });

    let mut keyed = argon2.hasher();
    keyed.reserve().expect("reserve");
    let mut keyed_out = [0u8; OUT_LEN];
    group.bench_function("hash_into_with_secret_and_ad", |b| {
        b.iter(|| {
            keyed
                .hash_into_with_ad(
                    black_box(PASSWORD),
                    black_box(SALT),
                    black_box(SECRET),
                    black_box(AD),
                    &mut keyed_out,
                )
                .expect("pooled keyed hash");
            black_box(&keyed_out);
        });
    });

    let mut encode = argon2.hasher();
    encode.reserve().expect("reserve");
    group.bench_function("hash_encoded", |b| {
        b.iter(|| {
            black_box(
                encode
                    .hash_encoded(black_box(PASSWORD), black_box(SALT))
                    .expect("pooled hash encoded"),
            );
        });
    });

    let mut verify = argon2.hasher();
    verify.reserve().expect("reserve");
    group.bench_function("verify_encoded", |b| {
        b.iter(|| {
            verify
                .verify_encoded(
                    black_box(&encoded),
                    black_box(PASSWORD),
                    Algorithm::Argon2id,
                )
                .expect("pooled verify encoded");
        });
    });

    let mut bounded = argon2.hasher();
    bounded.reserve().expect("reserve");
    group.bench_function("verify_encoded_bounded", |b| {
        b.iter(|| {
            bounded
                .verify_encoded_bounded(
                    black_box(&encoded),
                    black_box(PASSWORD),
                    Algorithm::Argon2id,
                    black_box(&p),
                )
                .expect("pooled bounded verify");
        });
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets = bench_algorithms, bench_phc_codec, bench_one_shot, bench_pooled
}
criterion_main!(benches);
