# argon2-rust

[![CI](https://github.com/Brooooooklyn/argon2-rust/actions/workflows/ci.yml/badge.svg)](https://github.com/Brooooooklyn/argon2-rust/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/argon2-rust.svg)](https://crates.io/crates/argon2-rust)
[![docs.rs](https://docs.rs/argon2-rust/badge.svg)](https://docs.rs/argon2-rust)
[![license](https://img.shields.io/badge/license-CC0--1.0%20OR%20Apache--2.0-blue.svg)](#license)

A pure-Rust port of the reference [Argon2](https://github.com/P-H-C/phc-winner-argon2)
implementation (RFC 9106), with **runtime-dispatched SIMD** — faster than the
C reference, OpenSSL, and the popular Rust crates, on both x86-64 and aarch64.

- **Zero mandatory dependencies**, `#![no_std]` + `alloc`
- **Bit-exact with the C reference** — verified against the official KAT
  traces (12,304 lines of internal state per file), the official `test.c`
  vectors, and a live differential harness comparing tags *and* C error codes
- **Runtime CPU dispatch**: AVX-512 → AVX2 → SSE2(+SSSE3) → NEON → scalar,
  cached in a single atomic; safe code can never reach an instruction set the
  CPU lacks. On wasm32 a **SIMD128** backend is selected at compile time
  (`-C target-feature=+simd128`), the only sound choice for a wasm module
- **Persistent worker pool** for `lanes > 1` (3 thread spawns per hash instead
  of the C's 48), with a hand-rolled 0.6 µs barrier
- **OS-native memory**: `mmap` + `MADV_HUGEPAGE` arena on Linux, secure
  wipe with a compiler barrier that survives `-O3`, optional pooled arena
  reuse across hashes
- **Full API**: raw hash, PHC encode/decode, verify, d/i/id × v0x10/v0x13,
  password-flavored aliases, error codes identical to the C (`-1..-35`)

## Performance

All numbers are wall-clock medians, interleaved rep-by-rep so machine drift
hits both arms equally, with **tag equality asserted on every repetition**.
Lower is better for the ms columns; bigger is better for speedups.

### vs the C reference (`phc-winner-argon2`)

Sapphire Rapids (`c3-standard-4`, 4 vCPU, 105 MiB L3), C built with its own
`-march=native` — its `fill_block` contains 503 AVX-512 EVEX instructions, so
this is the best the C can do on this machine. Argon2id:

| config | C (native AVX-512) | argon2-rust | speedup |
|---|---:|---:|---:|
| 64 MiB, t=1, p=1 | 50.0 ms | 24.9 ms | **2.01x** |
| 64 MiB, t=1, p=4 | 22.0 ms | 13.1 ms | **1.69x** |
| 64 MiB, t=3, p=1 | 98.3 ms | 65.2 ms | **1.51x** |
| 64 MiB, t=3, p=4 | 40.9 ms | 28.1 ms | **1.45x** |
| 256 MiB, t=1, p=1 | 216.4 ms | 105.8 ms | **2.05x** |
| 256 MiB, t=1, p=4 | 91.6 ms | 49.7 ms | **1.84x** |
| 256 MiB, t=3, p=1 | 471.0 ms | 309.9 ms | **1.52x** |
| 256 MiB, t=3, p=4 | 173.0 ms | 120.0 ms | **1.44x** |

Every reachable path wins — each Rust backend was also compared against the C
built for the *same* ISA tier (60+ cells, all three variants):

| Rust backend | C build | whole-hash range | fill kernel only |
|---|---|---|---|
| scalar | `ref.c` (gcc auto-vectorized to AVX2) | 1.11x – 1.52x | 1.00x – 1.10x |
| sse2 | `opt.c` SSE2 / SSSE3 | 1.14x – 1.54x | 1.03x – 1.19x |
| avx2 | `opt.c` AVX2 | 1.23x – 1.56x | 1.09x – 1.13x |
| avx512 | `opt.c` AVX-512 | 1.44x – 2.05x | 1.16x – 1.28x |

The fill-kernel win comes from a fully-unrolled round schedule (823 → 574
dynamic instructions per block) plus software prefetch of the reference block
on the data-independent path. The fixed-cost win (2.6x – 7.6x) comes from the
`mmap`+hugepage arena and the persistent worker pool — the C pays malloc,
page faults, and 48 `pthread_create`/`join` cycles per hash at t=3, p=4;
this crate pays one `mmap` and 3 spawns.

### vs OpenSSL 3.5 (EVP_KDF Argon2, thread pool enabled)

Same machine, in-process `EVP_KDF_derive` timing, tags verified identical:

| config | OpenSSL 3.5.5 | argon2-rust | speedup |
|---|---:|---:|---:|
| 64 MiB, t=1, p=1 | 85.4 ms | 24.1 ms | **3.5x** |
| 64 MiB, t=3, p=4 | 94.3 ms | 25.6 ms | **3.7x** |
| 256 MiB, t=1, p=1 | 356.7 ms | 103.8 ms | **3.4x** |
| 256 MiB, t=3, p=4 | 384.8 ms | 111.7 ms | **3.4x** |

(OpenSSL's Argon2 has no SIMD fill at all.)

### vs the popular Rust crates

The two most-downloaded Argon2 crates on crates.io —
[`argon2`](https://crates.io/crates/argon2) (RustCrypto, 41.6M downloads) and
[`rust-argon2`](https://crates.io/crates/rust-argon2) (19.7M downloads) —
interleaved, tags asserted identical, Argon2id:

**x86-64 (Sapphire Rapids, AVX-512 backend)**

| config | argon2-rust | RustCrypto | rust-argon2 | vs RustCrypto | vs rust-argon2 |
|---|---:|---:|---:|---:|---:|
| 64 MiB, t=1, p=1 | 23.6 ms | 57.0 | 85.8 | **2.4x** | **3.6x** |
| 64 MiB, t=3, p=4 | 23.7 ms | 133.7 | 199.1 | **5.6x** | **8.4x** |
| 256 MiB, t=1, p=1 | 99.3 ms | 253.0 | 367.3 | **2.5x** | **3.7x** |
| 256 MiB, t=3, p=4 | 115.7 ms | 600.9 | 874.3 | **5.2x** | **7.6x** |

**aarch64 (Apple Silicon, NEON backend)**

| config | argon2-rust | RustCrypto | rust-argon2 | vs RustCrypto | vs rust-argon2 |
|---|---:|---:|---:|---:|---:|
| 64 MiB, t=1, p=1 | 14.5 ms | 22.4 | 26.7 | **1.6x** | **1.9x** |
| 64 MiB, t=3, p=4 | 13.7 ms | 65.0 | 83.5 | **4.8x** | **6.1x** |
| 256 MiB, t=1, p=1 | 66.9 ms | 93.2 | 112.2 | **1.4x** | **1.7x** |
| 256 MiB, t=3, p=4 | 62.0 ms | 292.0 | 351.2 | **4.7x** | **5.7x** |

Both crates compute lanes sequentially even at `p > 1`, so the margin grows
with parallelism; the single-thread rows are the honest kernel-vs-kernel
comparison.

## Quick start

```toml
[dependencies]
argon2-rust = "0.1"
```

```rust
use argon2_rust::{Algorithm, Argon2, Params, Version};

let params = Params::new(65536, 3, 4, 32)?;          // m=64 MiB, t=3, p=4, out=32 B
let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);

let mut tag = [0u8; 32];
argon2.hash_into(b"password", b"random salt 16B!", &mut tag)?;

// PHC string format
let encoded = argon2.hash_encoded(b"password", b"random salt 16B!")?;
assert!(argon2.verify_encoded(b"password", &encoded).is_ok());
```

Hashing many passwords? Pool the arena — one allocation total instead of one
per hash:

```rust
let mut hasher = argon2.hasher();
for (pwd, salt) in &credentials {
    let mut tag = [0u8; 32];
    hasher.hash_into(pwd, salt, &mut tag)?;
}
```

## How the dispatch works

```
                     first hash in the process
                              │
                    is_*_feature_detected! cascade
        ┌──────────┬────────┬─────────┬────────┬────────┐
        ▼          ▼        ▼         ▼        ▼        ▼
      AVX-512 →  AVX2  →  SSE2  →   NEON  → scalar   (cached in one AtomicU8)
        │    (SSSE3 probed separately at runtime)
        │    aarch64: NEON is compile-time on Apple/Windows (measured
        │    fastest there); other aarch64 hosts run a one-time ~4 ms
        │    shootout — Neoverse N1 gets scalar, Apple-class cores NEON
        │    wasm32: SIMD128 selected at compile time (1.5-1.6x over
        │    scalar under wasmtime); on wasm32-wasip1-threads the worker
        │    pool runs for real — 3.9x from 4 lanes
        ▼
  one fn-pointer resolve per hash, one indirect call per *segment*
  (thousands of blocks) — dispatch cost is one relaxed atomic load
```

Safe API never names a backend, so safe code can never execute an instruction
the CPU lacks. Explicit-backend entry points exist for testing and are
`unsafe fn` (with `compile_fail` doctests proving the boundary).

## Feature flags

| feature | default | what it does |
|---|:---:|---|
| `std` | ✓ | runtime CPU detection (falls back to compile-time cfgs without it) |
| `parallel` | ✓ | multi-lane fill on the persistent worker pool |
| `zeroize-memory` | ✓ | wipe internal buffers (the C's `FLAG_clear_internal_memory`) |
| `bump-alloc` | | opt-in bump allocator for small scratch buffers inside a reusable `Workspace` |
| `internal-api` | | test/bench hooks (`__internal`); never enable in production |

`--no-default-features` builds for `no_std` (with `alloc`), including e.g.
`thumbv7em-none-eabi`.

## Verification

The test suite proves equivalence with the C rather than assuming it:

- **Golden traces**: all six official KAT files replayed block-by-block
  (12,304 lines of internal memory state each), per runnable backend
- **Official vectors**: every `hashtest()` call from `test.c`, plus
  error-state parity (same numeric codes, same messages)
- **Live differential**: ~1,300 randomized parameter sets hashed by both this
  crate and the compiled C reference in one process; tags *and* error codes
  must match exactly
- **Reuse/audit**: pooled-arena byte-identity with one-shot hashing, an
  allocator spy proving wipe-before-free, and RSS isolation checks

## Reproducing the benchmarks

```console
# Full criterion suite (vs C via dlopen, per-backend grid)
cargo bench --features internal-api

# Seconds-scale iteration harness
cargo bench --bench micro --features internal-api -- \
    --backend avx512 --m 262144 --t 3 --p 4 --vs-c
```

`--vs-c` loads `phc-winner-argon2/libargon2.so.1` (built with its own
`make`) at runtime, prints the ISA genuinely inside it, and asserts tag
equality every repetition.

## License

Licensed under either of [CC0-1.0](LICENSE-CC0) or
[Apache-2.0](LICENSE-APACHE) at your option.
