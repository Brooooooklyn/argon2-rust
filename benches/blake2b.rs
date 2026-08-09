//! Matched-process BLAKE2b backend benchmarks.
//!
//! These are the three shapes on Argon2's real path:
//!
//! * one short BLAKE2b digest (the H0 pre-hash shape);
//! * `blake2b_long(72 -> 1024)` for each of the first two blocks per lane; and
//! * `blake2b_long(1024 -> 32)` for the final tag.
//!
//! Every backend is forced explicitly after `is_available()` proves it safe,
//! so the scalar/SIMD comparison shares the same binary and process.

use std::hint::black_box;
use std::time::Duration;

use argon2_rust::__internal::{Blake2bBackend, blake2b_long_with_backend, blake2b_with_backend};
use criterion::{BenchmarkId, Criterion, Throughput};

const SHORT_INPUT: [u8; 72] = sequence();
const BLOCK_INPUT: [u8; 1024] = block_sequence();

const fn sequence() -> [u8; 72] {
    let mut out = [0u8; 72];
    let mut i = 0;
    while i < out.len() {
        out[i] = i as u8;
        i += 1;
    }
    out
}

const fn block_sequence() -> [u8; 1024] {
    let mut out = [0u8; 1024];
    let mut i = 0;
    while i < out.len() {
        out[i] = i as u8;
        i += 1;
    }
    out
}

fn main() {
    let mut criterion = Criterion::default()
        .sample_size(100)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
        .configure_from_args();

    let mut short = criterion.benchmark_group("blake2b/short_72_to_64");
    short.throughput(Throughput::Bytes(SHORT_INPUT.len() as u64));
    for &backend in Blake2bBackend::ALL {
        if !backend.is_available() {
            continue;
        }
        let mut out = [0u8; 64];
        short.bench_function(BenchmarkId::from_parameter(backend.name()), |b| {
            b.iter(|| {
                // SAFETY: unavailable backends were skipped above.
                unsafe {
                    blake2b_with_backend(black_box(&mut out), black_box(&SHORT_INPUT), backend)
                }
                .expect("valid lengths");
                black_box(out[0]);
            });
        });
    }
    short.finish();

    let mut expand = criterion.benchmark_group("blake2b/expand_72_to_1024");
    expand.throughput(Throughput::Bytes(SHORT_INPUT.len() as u64));
    for &backend in Blake2bBackend::ALL {
        if !backend.is_available() {
            continue;
        }
        let mut out = [0u8; 1024];
        expand.bench_function(BenchmarkId::from_parameter(backend.name()), |b| {
            b.iter(|| {
                // SAFETY: unavailable backends were skipped above.
                unsafe {
                    blake2b_long_with_backend(black_box(&mut out), black_box(&SHORT_INPUT), backend)
                }
                .expect("valid lengths");
                black_box(out[0]);
            });
        });
    }
    expand.finish();

    let mut finalize = criterion.benchmark_group("blake2b/finalize_1024_to_32");
    finalize.throughput(Throughput::Bytes(BLOCK_INPUT.len() as u64));
    for &backend in Blake2bBackend::ALL {
        if !backend.is_available() {
            continue;
        }
        let mut out = [0u8; 32];
        finalize.bench_function(BenchmarkId::from_parameter(backend.name()), |b| {
            b.iter(|| {
                // SAFETY: unavailable backends were skipped above.
                unsafe {
                    blake2b_long_with_backend(black_box(&mut out), black_box(&BLOCK_INPUT), backend)
                }
                .expect("valid lengths");
                black_box(out[0]);
            });
        });
    }
    finalize.finish();

    criterion.final_summary();
}
