//! Base64 codec microbenchmarks.
//!
//! ```text
//! cargo bench --bench base64
//! ```
//!
//! The 16-byte salt and 32-byte tag are the important shapes: they become the
//! two Base64 fields in a typical PHC string. Larger inputs show whether the
//! block loop scales without letting bulk throughput hide a small-input
//! regression.

use std::hint::black_box;

use argon2_rust::__internal::{b64_len, from_base64, to_base64};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

const LENGTHS: [usize; 5] = [8, 16, 32, 256, 4096];

fn input(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(37) ^ 0x5a)
        .collect()
}

fn bench_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("base64/encode");
    for len in LENGTHS {
        let src = input(len);
        let mut dst = vec![0u8; b64_len(len as u32) + 1];
        group.throughput(Throughput::Bytes(len as u64));
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, _| {
            b.iter(|| {
                black_box(to_base64(black_box(&mut dst), black_box(&src)).unwrap());
            });
        });
    }
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("base64/decode");
    for len in LENGTHS {
        let raw = input(len);
        let mut src = vec![0u8; b64_len(len as u32) + 1];
        let written = to_base64(&mut src, &raw).unwrap();
        src.truncate(written);
        let mut dst = vec![0u8; len];
        group.throughput(Throughput::Bytes(src.len() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(len), &len, |b, _| {
            b.iter(|| {
                black_box(from_base64(black_box(&mut dst), black_box(&src)).unwrap());
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode);
criterion_main!(benches);
