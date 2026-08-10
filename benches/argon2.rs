//! Criterion benchmarks for `argon2-rust`.
//!
//! Run everything:
//!
//! ```text
//! cargo bench --bench argon2
//! ```
//!
//! Run one group (criterion filters on the whole benchmark id, so the group
//! name is a prefix):
//!
//! ```text
//! cargo bench --bench argon2 -- grid
//! cargo bench --bench argon2 -- dispatch
//! cargo bench --bench argon2 -- vs_c
//! ```
//!
//! Two summary tables print after the groups: arena reuse, and Rust vs C. They
//! are hand-rolled rather than criterion because both are *ratios* between
//! benchmarks, and criterion times each benchmark as a separate block — on a
//! machine that drifts, a block design charges the drift to whichever arm ran
//! second. Both tables interleave instead. Environment:
//!
//! ```text
//! ARGON2_BENCH_SUMMARY=0            skip both tables
//! ARGON2_BENCH_REUSE_REPS=101       interleaved rounds per reuse row (default 15)
//! ARGON2_BENCH_REUSE_ONLY=m262144   restrict the reuse sweep to matching configs
//! ARGON2_BENCH_REUSE_BIG=1          add the two 1 GiB reuse rows
//! ARGON2_BENCH_DIST=m,t,p           dump every sample instead of benchmarking
//! ARGON2_BENCH_FORCE=avx2           run a backend detection rejected (see below)
//! ARGON2_BENCH_SUMMARY_BACKEND=avx2 time this backend in the ratio table
//! ```
//!
//! `ARGON2_BENCH_SUMMARY_BACKEND` exists for matched Rust-vs-C rows. Detection
//! always returns the *best* backend, so on an AVX-512 host the ratio table
//! would otherwise only ever compare `avx512` against whichever C is installed.
//! Point it at `avx2` with an AVX2 C library and the row becomes like-for-like.
//! It selects among backends this CPU can execute and refuses anything else.
//!
//! # Groups
//!
//! | Group | What it answers |
//! |---|---|
//! | `grid` | Every backend this CPU can execute, over the m_cost × t_cost × lanes sweep. |
//! | `variants` | Argon2id vs Argon2i vs Argon2d at one matched point. |
//! | `parallel` | `threads = 1` vs `threads = lanes` at fixed `lanes`. |
//! | `reuse` | A fresh arena per hash (`Argon2`) vs one kept between hashes (`Hasher`). |
//! | `reuse_alloc` | The three memory stages in isolation: allocate, wipe, recycle. |
//! | `encoded` | `hash_encoded` / `verify_encoded`, one-shot vs pooled. |
//! | `bump` | What wiring `bumpalo` into the encoding buffers *could* have bought. |
//! | `dispatch` | The cost of runtime backend detection, end to end and in isolation. |
//! | `vs_c` | This crate vs `phc-winner-argon2` at matched parameters. |
//!
//! `bump` only exists with `--features bump-alloc`; the rest always do.
//!
//! # Which backends run here
//!
//! Backends come from [`Backend::ALL`] filtered by `Backend::is_available()`,
//! which is runtime CPU detection — the same cascade the library itself uses.
//! On `aarch64-apple-darwin` that yields `scalar` and `neon`. Under
//! `--target x86_64-apple-darwin` it yields `scalar` and `sse2`: Rosetta 2
//! executes AVX2 but does not advertise it in `cpuid`, so `is_available()`
//! says `false` and the filter drops it. That is correct behaviour, so to
//! measure AVX2 anyway, force it past detection:
//!
//! ```text
//! ARGON2_BENCH_FORCE=avx2 cargo bench --target x86_64-apple-darwin -- grid
//! ```
//!
//! Use that rather than `RUSTFLAGS="-C target-feature=+avx2"`. Both make AVX2
//! appear, but the flag also lets LLVM auto-vectorise the *scalar* backend into
//! AVX2 that Rosetta then emulates, which wrecks the baseline — see the table
//! below. `ARGON2_BENCH_FORCE` changes dispatch only, so `scalar` stays
//! compiled at the x86-64 baseline and `scalar` vs `avx2` stays a fair pair.
//!
//! AVX-512 is never benchmarked here. Rosetta traps it with `SIGILL`, so
//! `is_available()` drops it and `ARGON2_BENCH_FORCE` explicitly refuses it.
//! It is compile-verified only. Do not add `+avx512f` to `RUSTFLAGS` either —
//! `src/fill_block/mod.rs` documents why that turns the check into a lie.
//!
//! ## Rosetta timings prove execution, not speed
//!
//! Run the x86 backends under Rosetta to check they *work*, and then throw the
//! numbers away. Measured on this host at `grid/*/m4096_t1_p1`. Rows 1 and 3
//! were taken back to back so the load they saw was the same; row 2 needs a
//! different `RUSTFLAGS` and so is a separate build and a separate run, which is
//! why it is quoted as a ratio against its own baseline rather than compared
//! absolutely to the other two.
//!
//! | build | scalar | sse2 | avx2 |
//! |---|---|---|---|
//! | `x86_64` baseline | 1.755 ms | 1.461 ms | not available |
//! | `x86_64` + `RUSTFLAGS=+avx2` | 10.11 ms | 2.026 ms | 5.497 ms |
//! | `x86_64` + `ARGON2_BENCH_FORCE=avx2` | 1.762 ms | 1.436 ms | 5.328 ms |
//!
//! **ROSETTA — EXECUTION PROOF, NOT SPEED. Every number in this table is
//! emulated x86 running on an arm64 CPU.**
//!
//! Row 2 is why row 3 exists. `RUSTFLAGS=+avx2` regresses `scalar` **5.8x** and
//! `sse2` **1.4x**, because LLVM auto-vectorises them into 256-bit code that
//! Rosetta then emulates — the baseline moves, so nothing can be compared
//! against it. `ARGON2_BENCH_FORCE` changes dispatch only, so rows 1 and 3 agree
//! on `scalar` to 0.4% and on `sse2` to 1.7%, and only `avx2` is added.
//!
//! AVX2 still lands slower than SSE2 in row 3, and that is also Rosetta: it
//! emulates 256-bit instructions worse than 128-bit ones. None of these numbers
//! say anything about how these backends perform on real x86 silicon. Only the
//! native `aarch64` numbers in this file are performance data.
//!
//! Rosetta distorts the *memory* stages too, and unevenly, which is why the
//! reuse percentage measured under it must not be quoted (`reuse_alloc`, 4 MiB):
//!
//! | stage | native aarch64 | Rosetta x86_64 | ratio |
//! |---|---|---|---|
//! | `alloc_free` (`posix_memalign` + memset + `free`) | 16.05 µs | 84.50 µs | **5.3x** |
//! | `secure_wipe` (volatile stores) | 74.80 µs | 82.75 µs | 1.1x |
//!
//! The memset is a bulk-store loop Rosetta translates badly; the volatile wipe
//! is already store-bound, so it barely moves. Reuse saves the first and not the
//! second, so under Rosetta it looks worth **-6.2%** at `m4096_t1_p1` against
//! **-2.7%** natively. Same code, same saving, 2.3x the apparent value.
//!
//! # Results on this host
//!
//! **NATIVE `aarch64-apple-darwin`, backend `neon` — this section is real speed.**
//! Apple M5 Max (12P + 6E), 128 GiB, 16 KiB pages, macOS, rustc 1.97.1, bench
//! profile (`opt-level = 3`, `lto = "thin"`, `codegen-units = 1`). The machine
//! was **not** idle: load average ran 3.1 to 9.0 (a browser, a window server and
//! sibling build jobs). That is why the reuse table below is a paired
//! interleaved measurement with a sign test rather than a pair of criterion
//! block estimates.
//!
//! ## Arena reuse — the headline
//!
//! `Argon2::hash_into` (fresh arena per hash) vs `Hasher::hash_into` (arena
//! kept). 101 interleaved rounds, order flipped every round; the delta is the
//! median per-round paired difference, `alloc` is `Arena::new` + drop measured
//! in the same run, `got` is delta / alloc. **Two independent runs are shown**,
//! because a single run of an effect this small proves nothing. The 1 GiB rows
//! are one run of 31.
//!
//! | config | one-shot | pooled | delta A | delta B | wins A | wins B | alloc | got A/B |
//! |---|---|---|---|---|---|---|---|---|
//! | `m8_t1_p1` | 10.9 µs | 10.9 µs | -0.76% | -0.77% | 71/101 | 86/101 | 0.1 µs | 100 / 101% |
//! | `m64_t1_p1` | 20.4 µs | 19.5 µs | **-4.55%** | **-4.33%** | 95/101 | 97/101 | 0.8 µs | 116 / 111% |
//! | `m1024_t1_p1` | 174 µs | 169 µs | **-3.25%** | **-2.83%** | 91/101 | 93/101 | 5.1 µs | 112 / 95% |
//! | `m4096_t1_p1` | 674 µs | 655 µs | **-2.66%** | **-2.64%** | 94/101 | 90/101 | 16 µs | 112 / 109% |
//! | `m4096_t3_p1` | 1.874 ms | 1.853 ms | **-1.07%** | **-0.97%** | 71/101 | 70/101 | 17 µs | 117 / 106% |
//! | `m4096_t1_p4` | 424 µs | 403 µs | **-5.06%** | **-4.29%** | 97/101 | 84/101 | 17 µs | 124 / 107% |
//! | `m65536_t1_p1` | 14.84 ms | 14.71 ms | -0.76% | -1.90% | *59/101* | *65/101* | 237 µs | 47 / 124% |
//! | `m65536_t1_p4` | 5.562 ms | 5.314 ms | **-4.07%** | **-3.82%** | 76/101 | 79/101 | 238 µs | 95 / 92% |
//! | `m65536_t3_p4` | 13.99 ms | 13.86 ms | -1.02% | -1.73% | *52/101* | *58/101* | 243 µs | 57 / 106% |
//! | `m262144_t1_p1` | 69.10 ms | 68.06 ms | **-1.31%** | **-1.51%** | 76/101 | 71/101 | 949 µs | 96 / 112% |
//! | `m262144_t1_p4` | 23.50 ms | 22.41 ms | **-4.79%** | **-4.31%** | 101/101 | 101/101 | 984 µs | 116 / 104% |
//! | `m262144_t3_p4` | 63.94 ms | 62.95 ms | **-1.77%** | **-1.84%** | 89/101 | 87/101 | 1.00 ms | 113 / 118% |
//! | `m1048576_t1_p1` | 329.5 ms | 322.3 ms | **-2.41%** | — | 25/31 | — | 4.08 ms | 195% |
//! | `m1048576_t1_p4` | 108.3 ms | 104.8 ms | **-3.00%** | — | 27/31 | — | 4.11 ms | 79% |
//!
//! `wins` is a sign test — the count of rounds in which the one-shot arm was
//! slower. 50% means "no difference", so 101/101 is conclusive and *52/101 is
//! nothing at all*. The two italicised rows are the honest non-results here: at
//! `m = 64` MiB the saving is 1.6% of a 15 ms hash whose per-round spread is
//! 4.5%, and 101 rounds cannot resolve that. Note that the sign test flagged
//! them *before* the second run disagreed with the first on those exact two
//! rows and no others — which is the point of quoting it.
//!
//! Read the rest as: **-1% to -5%, and it is exactly the allocation and zeroing
//! cost, never more.** Across run B's twelve rows `got` is 92% to 124%, mean
//! 107%: the saving *is* the `alloc` column. Rows above 100% are noise in the
//! numerator, not a bonus.
//!
//! Reuse does **not** remove allocator calls (there was only ever one), page
//! faults (a steady-state process has none) or the secure wipe (both arms pay
//! it, unchanged). It removes one memset.
//!
//! ### The row where the naive measurement said "slower"
//!
//! The criterion `reuse` group, which times the two arms as separate blocks,
//! reported `m65536_t3_p4` as **+3.80% for the pooled arm** — a regression. It
//! is not one: the same configuration interleaved is -1.0% and -1.7% across the
//! two runs above, with a sign test of 52/101 and 58/101, i.e. no resolvable
//! difference in either direction. A block design cannot tell a 0.24 ms saving
//! apart from a 0.5 ms thermal step that happened to land between the blocks.
//!
//! That is the whole reason the summary table exists, and it is why the
//! criterion `reuse` group should be read as a sanity check on the absolute
//! times and not as the source of the percentages.
//!
//! ### Where these numbers disagree with what is already written down
//!
//! * `src/core.rs`'s `Hasher` docs quote `256 MiB, t=1, p=1` at **-2.7%**. Both
//!   runs here give -1.3% and -1.5%, and the stage arithmetic agrees with the
//!   smaller figure: `alloc_free` at 256 MiB is 1.03 ms out of a 69 ms hash,
//!   which is 1.5%, so -2.7% is above what is physically available at that
//!   configuration. Everything else in that table reproduces.
//! * `src/lib.rs` describes the win as "1.4-2.4% at `lanes = 1` and 3.0-5.1% at
//!   `lanes = 4`". The `lanes = 1` spread measured here is wider on both sides
//!   (-0.8% at `m = 8` KiB to -4.6% at 64 KiB), and the `lanes = 4` floor is
//!   lower than 3.0% whenever `t_cost > 1` (-1.7% to -1.8% at `t = 3`), because
//!   extra passes multiply the fill and not the one saved memset.
//!
//! ## Where the memory time actually goes (`reuse_alloc`, native)
//!
//! | arena | `alloc_free` | `secure_wipe` | `recycle` |
//! |---|---|---|---|
//! | 64 KiB | 0.55 µs | 0.92 µs | 0.92 µs |
//! | 1 MiB | 4.22 µs | 18.52 µs | 18.65 µs |
//! | 4 MiB | 16.05 µs | 74.80 µs | 74.85 µs |
//! | 64 MiB | 257.5 µs | 1.231 ms | 1.234 ms |
//! | 256 MiB | 1.032 ms | 4.980 ms | 4.980 ms |
//!
//! Two things to read off it. `recycle` equals `secure_wipe` to within 0.7% at
//! every size, which is the proof that the pooled path adds nothing beyond the
//! wipe both paths already pay. And the wipe is **4.4x to 4.8x** the whole
//! allocation cost, because `write_volatile` per `u64` cannot lower to `dc zva`:
//! 50 GB/s against 240 GB/s. The thing this exercise optimised is the smaller of
//! the two by a factor of five.
//!
//! ## Runtime dispatch did not move
//!
//! What the `dispatch` group measures is `cached - forced`: two byte-identical
//! calls that differ only by one relaxed atomic load. Against the criterion
//! baseline this repository already had on disk, taken before any of the reuse
//! work existed:
//!
//! | | before | after |
//! |---|---|---|
//! | `cached - forced` at `m8` | -0.034 µs (-0.31%) | -0.041 µs (-0.37%) |
//! | `cached - forced` at `m4096` | -2.62 µs (-0.36%) | -0.15 µs (-0.02%) |
//! | `dispatch_isolated/cached_load` | 0.304 ns | 0.305 ns |
//! | `dispatch_isolated/uncached_detect` | 0.345 ns | 0.351 ns |
//! | `dispatch_isolated/resolve_fn_ptr` | 0.457 ns | 0.508 ns |
//!
//! Negative before and negative after: one relaxed load is below this host's
//! noise floor in both trees, which is the claim `src/fill_block/mod.rs` makes.
//! `public_api_pooled` is the new arm; at `m8`, the most adversarial size that
//! exists, it is 10.83 µs against `public_api`'s 10.95 µs, so the `Workspace`
//! acquire/guard/release round trip costs nothing measurable either.
//!
//! One caveat on that stored baseline: all three `m4096` arms also moved
//! together by -8.2% to -9.0%, including `forced`, which never touches a
//! `Workspace`. A uniform shift of every arm is machine state or the unrelated
//! `src/fill_block/` edits that predate it — it cannot be the reuse work, and it
//! cancels out of the `cached - forced` difference, which is the statistic the
//! group exists to produce.
//!
//! ## The encoded API, and the bump that was not wired
//!
//! `bumpalo` is **not** used by `hash_encoded` or `verify_encoded`. There are
//! zero call sites of `Workspace::bump` in `src/` outside `memory.rs` itself, so
//! `--features bump-alloc` cannot change those paths, and measuring with the
//! feature on and off produced +1.5%, +0.3%, +2.4% and -0.3% across four
//! benchmarks: no consistent sign, i.e. drift.
//!
//! What the feature would have been worth, measured (`bump`, native):
//!
//! | three buffers of 98 B + 51 B + 33 B | time | vs `Vec` |
//! |---|---|---|
//! | `Vec::try_reserve` + `resize` (what ships) | 32.38 ns | — |
//! | reused `Workspace` bump + wiped `reset_bump` | 14.70 ns | **-17.7 ns** |
//! | a fresh `Bump::new()` per call | 23.87 ns | -8.5 ns |
//!
//! 17.7 ns. The `encoded` group puts that in proportion: at `m = 8` KiB — the
//! smallest hash Argon2 permits, where encoding is the largest share it will
//! ever be — `hash_encoded` is 11.14 µs against `hash_into`'s 10.98 µs, so the
//! whole of encoding is 158 ns and a bump could remove at most a ninth of it. At
//! `m = 64` MiB the same difference is 127 µs, which is not the encoding at all;
//! it is noise three orders of magnitude larger than the effect.
//!
//! Note also that the fresh-`Bump` row *beats* three `Vec`s, because its single
//! chunk allocation amortises over three buffers. A fresh bump is only a
//! regression for a *single* buffer.
//!
//! # The C reference
//!
//! `vs_c` loads `phc-winner-argon2/libargon2.1.dylib` with `dlopen` at runtime
//! rather than linking it, so this bench needs no `build.rs` and no extra
//! dependency. Build the library first:
//!
//! ```text
//! make -C phc-winner-argon2 OPTTARGET=none
//! ```
//!
//! If the library is missing, `vs_c` prints a note and skips instead of
//! failing.
//!
//! **The comparison is Rust-SIMD vs C-SCALAR, not Rust vs C-SIMD.** `src/opt.c`
//! is x86-only, so on this arm64 host the reference Makefile's `OPTTARGET`
//! probe fails and it compiles `src/ref.c`, the portable scalar path. It is
//! still `-O3` (`cc -std=c89 -O3 -Wall -g -Iinclude -Isrc -pthread`); the
//! Makefile's "Building without optimizations" message refers to SIMD, not to
//! the optimiser. The honest like-for-like row is therefore `rust_scalar` vs
//! `c_ref`; `rust_neon` vs `c_ref` shows what this port's SIMD buys over the
//! only C build that exists on this machine.
//!
//! Both sides wipe the whole arena before freeing it, so neither is being
//! given a discount: `free_memory()` in `src/core.c` calls
//! `clear_internal_memory()` unconditionally and `FLAG_clear_internal_memory`
//! defaults to `1`, which is what the `zeroize-memory` feature does in
//! `Arena::drop`.

use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, SamplingMode, Throughput};

use argon2_rust::__internal::{Backend, fill_segment_fn, hash_with_backend};
use argon2_rust::{Algorithm, Argon2, Params, Version};

/// Reads the ISA out of the compiled C library instead of assuming one. Shared
/// verbatim with `benches/micro.rs`; see that file for the invocation.
#[path = "support/cref_isa.rs"]
mod cref_isa;

// ---------------------------------------------------------------------------
// Fixed inputs
// ---------------------------------------------------------------------------

const PWD: &[u8] = b"correct horse battery staple";
/// 16 bytes, the length RFC 9106 recommends and `argon2_hash` expects.
const SALT: &[u8] = b"0123456789abcdef";
const OUTLEN: usize = 32;

/// `m_cost` values in KiB: 4 MiB, 64 MiB, 256 MiB.
const M_COSTS: [u32; 3] = [4096, 65536, 262144];
const T_COSTS: [u32; 2] = [1, 3];
const LANES: [u32; 4] = [1, 2, 4, 8];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build params, or die loudly — a bench that silently skips is worse than one
/// that stops.
fn params(m_cost: u32, t_cost: u32, lanes: u32, threads: u32) -> Params {
    match Params::new_with_threads(m_cost, t_cost, lanes, threads, OUTLEN) {
        Ok(p) => p,
        Err(e) => panic!("bad bench params m={m_cost} t={t_cost} p={lanes}/{threads}: {e:?}"),
    }
}

/// Backends this CPU can actually execute, in ascending preference order.
///
/// Normally this is `Backend::ALL` filtered by `is_available()`, i.e. the same
/// runtime cascade the library uses.
///
/// `ARGON2_BENCH_FORCE=avx2[,sse2,...]` **adds** backends that detection
/// rejected. It exists for exactly one situation: Rosetta 2 on this host
/// *executes* AVX2 but does not advertise it in `cpuid`, so `is_available()`
/// correctly returns `false` and the filter correctly drops it — yet the code
/// does run, and it is worth measuring.
///
/// This is the right way to get that measurement, and
/// `RUSTFLAGS="-C target-feature=+avx2"` is the wrong way. That flag makes
/// `is_x86_feature_detected!("avx2")` short-circuit to `true`, so AVX2 appears
/// — but it also lets LLVM auto-vectorise the *scalar* backend into AVX2, which
/// Rosetta then emulates. The module docs record the damage: scalar regresses
/// 5.5x, which destroys the baseline the comparison is against. Forcing the
/// backend here instead touches only dispatch. Scalar stays compiled at the
/// x86-64 baseline, so `scalar` vs `avx2` remains a fair pair.
///
/// # Safety
///
/// A `#[target_feature]` function whose feature is absent is UB. This is sound
/// on this host *only* because Rosetta genuinely executes SSE2/SSSE3/AVX2 — a
/// measured fact, not an assumption. Never force `avx512`: Rosetta traps it
/// with `SIGILL`. The check below refuses it for that reason. This is a
/// bench-only affordance and never a library path.
/// Which backend the interleaved summary tables time as the "SIMD" arm.
///
/// Defaults to [`argon2_rust::detected_backend`], which is the right answer for
/// a headline number: it is what a caller actually gets.
///
/// It is overridable because a *matched* Rust-vs-C row needs the Rust backend to
/// be the same instruction set the C was built at, and detection always returns
/// the best one. On a host that detects `avx512` there is otherwise no way to
/// time `rust_avx2` against a C built at AVX2 using this drift-robust method —
/// the only alternative is criterion's block design, which is the exact bias
/// [`interleaved_samples`] exists to remove. Measured on a noisy 4-vCPU box, the
/// two disagreed by up to 17%.
///
/// This is deliberately NOT `ARGON2_BENCH_FORCE`. That knob *adds* a backend
/// detection rejected, so it can name something this CPU cannot execute; this
/// one selects among backends that are already available, and refuses anything
/// else rather than crash or silently fall back.
fn summary_backend() -> Backend {
    let detected = argon2_rust::detected_backend();
    let Ok(spec) = std::env::var("ARGON2_BENCH_SUMMARY_BACKEND") else {
        return detected;
    };
    let name = spec.trim();
    match Backend::ALL.iter().copied().find(|b| b.name() == name) {
        Some(b) if b.is_available() => b,
        Some(b) => {
            eprintln!(
                "ARGON2_BENCH_SUMMARY_BACKEND: {} is not executable on this CPU, \
                 using {detected}",
                b.name()
            );
            detected
        }
        None => {
            eprintln!("ARGON2_BENCH_SUMMARY_BACKEND: unknown backend {name:?}, using {detected}");
            detected
        }
    }
}

fn runnable_backends() -> Vec<Backend> {
    let mut backends: Vec<Backend> = Backend::ALL
        .iter()
        .copied()
        .filter(|b| b.is_available())
        .collect();

    let Ok(spec) = std::env::var("ARGON2_BENCH_FORCE") else {
        return backends;
    };

    for name in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let Some(forced) = Backend::ALL.iter().copied().find(|b| b.name() == name) else {
            eprintln!("ARGON2_BENCH_FORCE: unknown backend {name:?}, ignoring");
            continue;
        };
        if forced == Backend::Avx512 {
            eprintln!(
                "ARGON2_BENCH_FORCE: refusing to force avx512 — it traps with SIGILL \
                 under Rosetta and cannot be executed on this host at all"
            );
            continue;
        }
        if backends.contains(&forced) {
            continue;
        }
        eprintln!(
            "ARGON2_BENCH_FORCE: adding {name} even though detection rejected it. \
             Sound here only because Rosetta executes these instructions."
        );
        backends.push(forced);
    }

    // Keep `Backend::ALL`'s ascending preference order, so bench ids stay in a
    // stable sequence no matter what order the env var listed.
    backends.sort_unstable();
    backends
}

/// The backends [`runnable_backends`] blessed, as a bitmask over `Backend as u8`,
/// resolved once.
///
/// This is what lets [`rust_hash`] stay a **safe** function while
/// `__internal::hash_with_backend` is `unsafe`: every backend it forwards has
/// been through the cascade above, so it is either advertised by `cpuid` /
/// `sysctl` or was deliberately opted in through `ARGON2_BENCH_FORCE` (which
/// refuses `avx512`). A `Backend` that reached `rust_hash` any other way is a
/// bug in this file and now fails loudly instead of executing an instruction
/// the CPU may not have.
fn blessed_mask() -> u8 {
    static MASK: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MASK.get_or_init(|| {
        runnable_backends()
            .into_iter()
            .fold(0u8, |mask, b| mask | (1u8 << (b as u8)))
    })
}

/// Run a few large hashes on every runnable backend before any group is timed.
///
/// Criterion warms up each *benchmark*, but not the *process*, so whichever
/// benchmark happens to run first absorbs all the one-time costs: first-touch
/// page faults on a fresh 64 MiB arena, the allocator growing its heap, the
/// thread pool's first `std::thread::scope`, and the CPU still at an idle clock.
///
/// That is not a hypothetical. Without this, `variants/scalar/argon2id` — the
/// first benchmark in that group — measured 21.73 ms while the `grid` group
/// measured the identical configuration at 19.04 ms, a 14% inflation. It made
/// Argon2id look *slower* than Argon2i, which is impossible: Argon2id runs
/// `next_addresses` only on the first half of pass 0, Argon2i runs it on every
/// segment, so Argon2id can never do more work.
///
/// Both `m_cost` values used here are large enough to fault in a realistic
/// arena, and both thread counts are exercised so the parallel path is warm too.
fn warm_up_process() {
    let mut out = [0u8; OUTLEN];
    for backend in runnable_backends() {
        for &(m_cost, lanes) in &[(65536u32, 1u32), (65536, 8)] {
            let p = params(m_cost, 1, lanes, lanes);
            rust_hash(backend, Algorithm::Argon2id, &p, &mut out);
        }
    }
    black_box(out);
}

/// One full hash through the forced-backend entry point.
///
/// Safe, and [`blessed_mask`] is what makes it safe: it refuses any backend that
/// neither runtime detection nor `ARGON2_BENCH_FORCE` sanctioned. The check is a
/// cached load and a bit test against a hash that takes milliseconds, so it does
/// not perturb a measurement.
#[inline]
fn rust_hash(backend: Backend, algorithm: Algorithm, p: &Params, out: &mut [u8; OUTLEN]) {
    assert!(
        blessed_mask() & (1u8 << (backend as u8)) != 0,
        "{backend} was neither detected on this CPU nor listed in \
         ARGON2_BENCH_FORCE; running it would be undefined behaviour"
    );
    // SAFETY: the assertion above establishes exactly what
    // `hash_with_backend` requires — `backend` came out of
    // `runnable_backends()`, so it is either advertised by runtime detection or
    // was explicitly opted in on a host measured to execute it (see that
    // function's `# Safety`).
    unsafe {
        hash_with_backend(
            backend,
            algorithm,
            Version::V0x13,
            p,
            PWD,
            SALT,
            &[],
            &[],
            out,
        )
    }
    .expect("hash");
}

/// Bytes of memory the algorithm touches per hash: the *aligned* block count,
/// not the requested `m_cost`, times the number of passes.
fn throughput_of(p: &Params) -> Throughput {
    Throughput::Bytes(u64::from(p.memory_blocks()) * 1024 * u64::from(p.passes()))
}

/// Very rough wall-clock estimate, used only to pick sample counts so the sweep
/// finishes in minutes rather than hours. Calibrated on this host at roughly
/// 0.4 µs per KiB per pass single-threaded. Getting it wrong costs runtime, not
/// accuracy: criterion re-derives its real iteration count from the warm-up.
fn est_millis(p: &Params) -> f64 {
    const MS_PER_KIB_PER_PASS: f64 = 0.000_40;
    let cpus = std::thread::available_parallelism().map_or(1, std::num::NonZero::get) as f64;
    let parallel = f64::from(p.effective_threads()).min(cpus);
    let serial = f64::from(p.memory_blocks()) * f64::from(p.passes()) * MS_PER_KIB_PER_PASS;
    // 1.3 covers imperfect scaling; floor at 1 so tiny cases stay sane.
    (serial / parallel * 1.3).max(0.001)
}

/// Size a group for a benchmark expected to take `est_ms`, so that no single
/// benchmark runs for more than a few seconds.
///
/// The sample floors matter more than they look. At `m_cost = 262144` one
/// iteration allocates, fills, wipes and frees 256 MiB, so a single descheduled
/// run — this host is a laptop with a browser on it — can be 2-3x the honest
/// time. Criterion's headline figure is a *mean*, so with the criterion default
/// of 10 samples one such run moves the estimate by 20%. Twenty samples plus a
/// warm-up of roughly three iterations keeps the big configurations stable; the
/// `Found N outliers` line criterion prints is the check that it worked.
fn tune(group: &mut BenchmarkGroup<'_, WallTime>, est_ms: f64) {
    let samples = if est_ms >= 100.0 {
        20
    } else if est_ms >= 20.0 {
        25
    } else if est_ms >= 2.0 {
        50
    } else {
        100
    };
    if est_ms >= 20.0 {
        // Linear sampling would run 1+2+...+n iterations. For anything this
        // slow that is a waste; Flat runs exactly `samples` batches.
        group.sampling_mode(SamplingMode::Flat);
    }
    let measurement = (est_ms * f64::from(samples) * 1.2).max(1000.0);
    let warm_up = (est_ms * 3.0).clamp(300.0, 3000.0);
    group
        .sample_size(samples as usize)
        .warm_up_time(Duration::from_millis(warm_up as u64))
        .measurement_time(Duration::from_millis(measurement as u64));
}

// ---------------------------------------------------------------------------
// The C reference, loaded at runtime
// ---------------------------------------------------------------------------

/// `dlopen`/`dlsym` bindings to `phc-winner-argon2`.
///
/// Loading the reference at runtime keeps the whole C comparison inside the one
/// file this bench owns: no `build.rs`, no `.cargo/config.toml`, no new
/// dependency. `dlopen` lives in libSystem on macOS and libc on modern glibc,
/// both of which every Rust binary already links.
mod cref {
    use std::ffi::{c_char, c_int, c_void};

    /// `argon2_hash()` from `include/argon2.h`.
    pub type Argon2Hash = unsafe extern "C" fn(
        t_cost: u32,
        m_cost: u32,
        parallelism: u32,
        pwd: *const c_void,
        pwdlen: usize,
        salt: *const c_void,
        saltlen: usize,
        hash: *mut c_void,
        hashlen: usize,
        encoded: *mut c_char,
        encodedlen: usize,
        ty: c_int,
        version: u32,
    ) -> c_int;

    unsafe extern "C" {
        fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    /// `RTLD_NOW`: bind every symbol up front, so no lazy-binding stub cost
    /// lands inside the first timed call.
    const RTLD_NOW: c_int = 2;

    const CANDIDATES: &[&str] = &[
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/phc-winner-argon2/libargon2.1.dylib\0"
        ),
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/phc-winner-argon2/libargon2.so.1\0"
        ),
    ];

    /// The reference `argon2_hash`, or `None` if the library was never built.
    pub fn argon2_hash() -> Option<Argon2Hash> {
        for path in CANDIDATES {
            // SAFETY: `path` is a `\0`-terminated string literal. `dlopen`
            // returns null on failure, which is checked. The handle is
            // deliberately leaked: the library stays loaded for the process.
            let handle = unsafe { dlopen(path.as_ptr().cast::<c_char>(), RTLD_NOW) };
            if handle.is_null() {
                continue;
            }
            // SAFETY: `handle` is a live `dlopen` handle and the symbol name is
            // `\0`-terminated. Null is checked before use.
            let symbol = unsafe { dlsym(handle, c"argon2_hash".as_ptr()) };
            if symbol.is_null() {
                continue;
            }
            // SAFETY: `argon2_hash` in `include/argon2.h` has exactly the
            // signature `Argon2Hash` spells out, and the loaded library is the
            // reference implementation this repo vendors.
            return Some(unsafe { std::mem::transmute::<*mut c_void, Argon2Hash>(symbol) });
        }
        None
    }

    /// Which file actually loaded, for the report header.
    pub fn library_path() -> Option<&'static str> {
        for path in CANDIDATES {
            // SAFETY: as above.
            let handle = unsafe { dlopen(path.as_ptr().cast::<c_char>(), RTLD_NOW) };
            if !handle.is_null() {
                return Some(path.trim_end_matches('\0'));
            }
        }
        None
    }

    /// `Algorithm` as the C's `argon2_type` discriminant.
    pub const fn type_of(algorithm: super::Algorithm) -> c_int {
        match algorithm {
            super::Algorithm::Argon2d => 0,
            super::Algorithm::Argon2i => 1,
            super::Algorithm::Argon2id => 2,
        }
    }

    /// One reference hash. Panics on a non-`ARGON2_OK` return, because that
    /// means the two sides were not asked the same question.
    pub fn hash(
        f: Argon2Hash,
        algorithm: super::Algorithm,
        m_cost: u32,
        t_cost: u32,
        parallelism: u32,
        out: &mut [u8; super::OUTLEN],
    ) {
        // SAFETY: every pointer is to a live local or to a `const` slice with
        // the length passed alongside it; `encoded` is null with length 0,
        // which `argon2_hash` documents as "raw hash only".
        let rc = unsafe {
            f(
                t_cost,
                m_cost,
                parallelism,
                super::PWD.as_ptr().cast::<c_void>(),
                super::PWD.len(),
                super::SALT.as_ptr().cast::<c_void>(),
                super::SALT.len(),
                out.as_mut_ptr().cast::<c_void>(),
                super::OUTLEN,
                std::ptr::null_mut(),
                0,
                type_of(algorithm),
                0x13,
            )
        };
        assert_eq!(rc, 0, "C argon2_hash failed with code {rc}");
    }

    /// The header line, describing the library that is **actually loaded**.
    ///
    /// This used to be the constant string `"(src/ref.c, scalar, -O3)"`. That
    /// was a lie whenever the tree had been rebuilt with `make OPTTARGET=...`,
    /// and it lied at the worst possible moment: it printed "scalar" beside
    /// timings taken against an AVX-512 `src/opt.c`, and a conclusion was drawn
    /// from the resulting mismatched comparison. Nothing here is assumed now —
    /// [`crate::cref_isa::probe`] reads the `fill_block` bytes out of the
    /// binary and reports the encoding markers it finds alongside the label, so
    /// the classification can be checked rather than trusted.
    pub fn describe() -> String {
        library_path().map_or_else(
            || "not built".to_string(),
            |p| {
                super::cref_isa::probe(p).map_or_else(
                    || format!("{p} (ISA probe failed: could not read the file)"),
                    |isa| isa.describe(),
                )
            },
        )
    }

    /// Just the ISA word (`scalar` when the probe found nothing vector-shaped),
    /// for the footnote under the ratio table.
    pub fn isa_word() -> String {
        library_path()
            .and_then(super::cref_isa::probe)
            .map_or_else(|| "unknown".to_string(), |i| i.isa().to_string())
    }
}

// ---------------------------------------------------------------------------
// 1. The parameter grid, per backend
// ---------------------------------------------------------------------------

/// Every runnable backend across m_cost × t_cost × lanes, Argon2id.
///
/// `threads == lanes` here, which is what `argon2_hash()` in the C does and
/// what `Params::new` does, so this measures the configuration a caller
/// actually gets.
fn bench_grid(c: &mut Criterion) {
    let backends = runnable_backends();
    let mut group = c.benchmark_group("grid");

    for &m_cost in &M_COSTS {
        for &t_cost in &T_COSTS {
            for &lanes in &LANES {
                let p = params(m_cost, t_cost, lanes, lanes);
                group.throughput(throughput_of(&p));
                tune(&mut group, est_millis(&p));

                let label = format!("m{m_cost}_t{t_cost}_p{lanes}");
                for &backend in &backends {
                    let mut out = [0u8; OUTLEN];
                    group.bench_function(BenchmarkId::new(backend.name(), &label), |b| {
                        b.iter(|| {
                            rust_hash(
                                black_box(backend),
                                Algorithm::Argon2id,
                                black_box(&p),
                                &mut out,
                            );
                        });
                    });
                }
            }
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. The three variants
// ---------------------------------------------------------------------------

/// Argon2id (the primary) against one Argon2i and one Argon2d point.
///
/// Argon2i pays for `next_addresses` on every pass; Argon2id pays for it only
/// on the first half of pass 0; Argon2d never does. The gap between them is the
/// data-independent addressing overhead.
fn bench_variants(c: &mut Criterion) {
    let backends = runnable_backends();
    let p = params(65536, 3, 4, 4);
    let mut group = c.benchmark_group("variants");
    group.throughput(throughput_of(&p));
    tune(&mut group, est_millis(&p));

    for &backend in &backends {
        for algorithm in [Algorithm::Argon2id, Algorithm::Argon2i, Algorithm::Argon2d] {
            let mut out = [0u8; OUTLEN];
            group.bench_function(BenchmarkId::new(backend.name(), algorithm.as_str()), |b| {
                b.iter(|| {
                    rust_hash(black_box(backend), algorithm, black_box(&p), &mut out);
                });
            });
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. Parallel scaling
// ---------------------------------------------------------------------------

/// `threads = 1` vs `threads = lanes`, at identical `lanes`.
///
/// `lanes` is held fixed inside each pair, so both arms compute the **same
/// tag** — `threads` is a pure performance knob (`core.c` reduces it to
/// `min(threads, lanes)` and it never reaches `index_alpha`). The bench asserts
/// that equality before measuring, which is also a live check on the threaded
/// path's correctness.
fn bench_parallel(c: &mut Criterion) {
    let backend = argon2_rust::detected_backend();
    let mut group = c.benchmark_group("parallel");

    for &m_cost in &[65536u32, 262144] {
        for &lanes in &[2u32, 4, 8] {
            // Same lanes, different thread counts => identical output.
            let single = params(m_cost, 1, lanes, 1);
            let multi = params(m_cost, 1, lanes, lanes);

            let mut tag_st = [0u8; OUTLEN];
            let mut tag_mt = [0u8; OUTLEN];
            rust_hash(backend, Algorithm::Argon2id, &single, &mut tag_st);
            rust_hash(backend, Algorithm::Argon2id, &multi, &mut tag_mt);
            assert_eq!(
                tag_st, tag_mt,
                "threads changed the tag at m={m_cost} lanes={lanes}; \
                 threads must never affect output, only lanes"
            );

            group.throughput(throughput_of(&multi));

            for (name, p) in [("threads1", &single), ("threadsN", &multi)] {
                tune(&mut group, est_millis(p));
                let mut out = [0u8; OUTLEN];
                let label = format!("m{m_cost}_lanes{lanes}");
                group.bench_function(BenchmarkId::new(name, &label), |b| {
                    b.iter(|| {
                        rust_hash(
                            black_box(backend),
                            Algorithm::Argon2id,
                            black_box(p),
                            &mut out,
                        );
                    });
                });
            }
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Dispatch overhead
// ---------------------------------------------------------------------------

/// Measure the runtime-detection overhead instead of asserting it is small.
///
/// The claim under test, from `src/fill_block/mod.rs`: *one relaxed atomic load
/// per hash call, one indirect call per segment.*
///
/// Two halves:
///
/// **End to end.** Three arms that differ only in how the backend reaches
/// `hash_inner`:
///
/// * `forced` — `hash_with_backend(black_box(backend), ..)`. No atomic load.
///   `black_box` stops LLVM const-folding the enum and devirtualising
///   `fill_segment_fn`, so the indirect call survives.
/// * `cached` — `hash_with_backend(detected_backend(), ..)`. Byte-identical to
///   `forced` except for the cached relaxed load. This is the clean A/B: the
///   delta is *exactly* the dispatch. The load really does happen once per
///   iteration — `Ordering::Relaxed` lowers to LLVM `monotonic`, and LICM will
///   only hoist `unordered` atomic loads out of a loop, never `monotonic` ones.
/// * `public_api` — `Argon2::hash_into`, the path a real caller takes.
///
/// They run at `m_cost = 8` KiB, the smallest `ARGON2_MIN_MEMORY` allows. That
/// is the most adversarial case that exists: 8 blocks total, `segment_length`
/// of 2, so the fixed per-call cost is amortised over 8 blocks and the indirect
/// call over 2. At the realistic `m_cost = 4096` the same call is amortised
/// over 1024 blocks per segment, so both points are measured.
///
/// **In isolation.** `detected_backend` (the cached relaxed load), `detect`
/// (the uncached feature cascade — what caching avoids), and `fill_segment_fn`
/// (resolving the pointer). Caveat, stated because it matters: a relaxed atomic
/// load in a tight loop is something LLVM may legally CSE, so the isolated
/// `detected_backend` figure is a *lower* bound on one load. The end-to-end
/// delta above is the number that cannot be optimised away, and it is the one
/// to trust.
fn bench_dispatch(c: &mut Criterion) {
    let backend = argon2_rust::detected_backend();

    // -- end to end -----------------------------------------------------
    //
    // Deliberately *not* `tune`d. This group has to resolve a difference of
    // nanoseconds inside a call of microseconds, so it buys statistical power
    // with a big sample count and a long measurement. Flat sampling keeps that
    // affordable: 200 batches instead of the 1+2+..+200 that linear sampling
    // would run.
    let mut group = c.benchmark_group("dispatch");
    group
        .sample_size(200)
        .sampling_mode(SamplingMode::Flat)
        .warm_up_time(Duration::from_secs(1));

    for &m_cost in &[8u32, 4096] {
        let p = params(m_cost, 1, 1, 1);
        let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, p);
        group.throughput(throughput_of(&p));
        group.measurement_time(Duration::from_secs(if m_cost <= 8 { 5 } else { 8 }));
        let label = format!("m{m_cost}");

        let mut out = [0u8; OUTLEN];
        group.bench_function(BenchmarkId::new("forced", &label), |b| {
            b.iter(|| {
                rust_hash(
                    black_box(backend),
                    Algorithm::Argon2id,
                    black_box(&p),
                    &mut out,
                );
            });
        });

        let mut out = [0u8; OUTLEN];
        group.bench_function(BenchmarkId::new("cached", &label), |b| {
            b.iter(|| {
                rust_hash(
                    argon2_rust::detected_backend(),
                    Algorithm::Argon2id,
                    black_box(&p),
                    &mut out,
                );
            });
        });

        let mut out = [0u8; OUTLEN];
        group.bench_function(BenchmarkId::new("public_api", &label), |b| {
            b.iter(|| {
                black_box(&argon2)
                    .hash_into(PWD, SALT, &mut out)
                    .expect("hash");
            });
        });

        // The pooled path at the same two points. At `m = 8` KiB there is
        // essentially no memset to save (42 ns of a 11 us hash), so this arm is
        // not here to show a win — it is here to catch a *loss*. If `Hasher`'s
        // extra per-call work (a `Workspace::acquire`, a guard, a `release`) cost
        // anything measurable, the smallest hash Argon2 permits is where it would
        // show, and it would show as a regression against `public_api`.
        let mut hasher = argon2.hasher();
        hasher.hash_into(PWD, SALT, &mut out).expect("hash");
        let mut out = [0u8; OUTLEN];
        group.bench_function(BenchmarkId::new("public_api_pooled", &label), |b| {
            b.iter(|| {
                hasher.hash_into(PWD, SALT, &mut out).expect("hash");
            });
        });
    }
    group.finish();

    // -- in isolation ---------------------------------------------------
    let mut group = c.benchmark_group("dispatch_isolated");
    group.sample_size(200);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(3));

    group.bench_function("cached_load", |b| {
        b.iter(|| black_box(argon2_rust::detected_backend()));
    });
    group.bench_function("uncached_detect", |b| {
        b.iter(|| black_box(argon2_rust::__internal::detect()));
    });
    group.bench_function("resolve_fn_ptr", |b| {
        b.iter(|| black_box(fill_segment_fn(black_box(backend))));
    });
    group.finish();
}

// ---------------------------------------------------------------------------
// 5. Rust vs the C reference
// ---------------------------------------------------------------------------

/// Matched-parameter comparison against `phc-winner-argon2`.
///
/// Every point is cross-checked for tag equality before it is timed, so the two
/// sides are provably doing the same work. See the module docs: the C here is
/// its **scalar** `src/ref.c` at `-O3`, because `src/opt.c` is x86-only.
fn bench_vs_c(c: &mut Criterion) {
    let Some(c_hash) = cref::argon2_hash() else {
        eprintln!(
            "vs_c: skipped, no reference library. Build it with:\n  \
             make -C phc-winner-argon2 OPTTARGET=none"
        );
        return;
    };

    let detected = argon2_rust::detected_backend();
    let mut group = c.benchmark_group("vs_c");

    for &(m_cost, t_cost, lanes) in &[
        (65536u32, 1u32, 1u32),
        (65536, 3, 1),
        (65536, 1, 4),
        (262144, 1, 1),
        (262144, 3, 4),
    ] {
        let p = params(m_cost, t_cost, lanes, lanes);

        // Prove the two sides agree before timing them.
        let mut tag_rust = [0u8; OUTLEN];
        let mut tag_c = [0u8; OUTLEN];
        rust_hash(detected, Algorithm::Argon2id, &p, &mut tag_rust);
        cref::hash(
            c_hash,
            Algorithm::Argon2id,
            m_cost,
            t_cost,
            lanes,
            &mut tag_c,
        );
        assert_eq!(
            tag_rust, tag_c,
            "rust and C disagree at m={m_cost} t={t_cost} p={lanes}"
        );

        group.throughput(throughput_of(&p));
        tune(&mut group, est_millis(&p));
        let label = format!("m{m_cost}_t{t_cost}_p{lanes}");

        // Rust, every runnable backend.
        for &backend in &runnable_backends() {
            let mut out = [0u8; OUTLEN];
            group.bench_function(
                BenchmarkId::new(format!("rust_{}", backend.name()), &label),
                |b| {
                    b.iter(|| {
                        rust_hash(
                            black_box(backend),
                            Algorithm::Argon2id,
                            black_box(&p),
                            &mut out,
                        );
                    });
                },
            );
        }

        // C.
        let mut out = [0u8; OUTLEN];
        group.bench_function(BenchmarkId::new("c_ref", &label), |b| {
            b.iter(|| {
                cref::hash(
                    c_hash,
                    Algorithm::Argon2id,
                    black_box(m_cost),
                    black_box(t_cost),
                    black_box(lanes),
                    &mut out,
                );
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 6. Arena reuse: a fresh arena per hash vs one kept between hashes
// ---------------------------------------------------------------------------

/// The reuse sweep: `(m_cost KiB, t_cost, lanes)`.
///
/// Deliberately wider than the rest of this file. Reuse saves one fixed thing
/// per hash — the `alloc_zeroed` memset — so the *fraction* it saves is that
/// constant over the whole hash, and the whole hash moves with all three knobs.
/// The sweep has to span all three or the answer is a single cherry-picked
/// percentage:
///
/// * `m_cost` scales the saved memset and the fill together, so the ratio is
///   roughly flat in `m_cost` — but only above ~1 MiB. Below that a hash is
///   mostly BLAKE2b (74.6% of an `m = 8` KiB hash is the pre-hash), which
///   dilutes it to nothing. Both ends are in here.
/// * `lanes` shrinks the fill by running it on several cores and does **not**
///   shrink the memset, so it roughly triples the share. This is the *best* case
///   and it is the realistic server one.
/// * `t_cost` multiplies the fill and does **not** multiply the memset, so it
///   roughly halves the share at `t = 3`. This is the *worst* case, and leaving
///   it out would be picking the flattering configuration.
const REUSE_GRID: &[(u32, u32, u32)] = &[
    (8, 1, 1),      // ARGON2_MIN_MEMORY: the smallest hash that exists
    (64, 1, 1),     // 64 KiB
    (1024, 1, 1),   // 1 MiB
    (4096, 1, 1),   // 4 MiB
    (4096, 3, 1),   // t_cost dilutes it
    (4096, 1, 4),   // lanes concentrate it
    (65536, 1, 1),  // RFC 9106 second recommendation
    (65536, 1, 4),  //   ... with 4 lanes
    (65536, 3, 4),  //   ... and 3 passes
    (262144, 1, 1), // 256 MiB
    (262144, 1, 4), //   ... with 4 lanes
    (262144, 3, 4), //   ... and 3 passes
];

/// `REUSE_GRID`, plus the 1 GiB rows when `ARGON2_BENCH_REUSE_BIG` is set.
///
/// 1 GiB is off by default because one `oneshot` sample allocates, zeroes,
/// fills, wipes and frees a gibibyte — around 306 ms at `lanes = 1` — and
/// criterion wants twenty of them per arm.
/// `ARGON2_BENCH_REUSE_ONLY=<substr>[,<substr>...]` keeps only the configurations
/// whose `mM_tT_pP` label contains one of the substrings, so a single row can be
/// re-run at a high round count without paying for the whole sweep.
fn reuse_grid() -> Vec<(u32, u32, u32)> {
    let mut grid = REUSE_GRID.to_vec();
    if std::env::var_os("ARGON2_BENCH_REUSE_BIG").is_some() {
        grid.push((1_048_576, 1, 1));
        grid.push((1_048_576, 1, 4));
    }
    if let Ok(spec) = std::env::var("ARGON2_BENCH_REUSE_ONLY") {
        let wanted: Vec<&str> = spec
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        grid.retain(|&(m, t, p)| {
            let label = format!("m{m}_t{t}_p{p}");
            wanted.iter().any(|w| label.contains(w))
        });
    }
    grid
}

/// Set up one reuse configuration: a one-shot `Argon2`, a warmed pooled hasher,
/// and proof that the two agree.
///
/// The warm-up is not politeness, it is the whole measurement. The *first*
/// pooled hash allocates exactly like the one-shot one; only the second and
/// later ones reuse. Timing a cold hasher would measure the two paths doing
/// identical work and report — correctly, and uselessly — that reuse is worth
/// nothing.
///
/// Returns the `Argon2` and the tag both paths produce.
macro_rules! reuse_pair {
    ($argon2:ident, $hasher:ident, $p:expr) => {
        let $argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, $p);
        let mut $hasher = $argon2.hasher();

        let mut tag_oneshot = [0u8; OUTLEN];
        let mut tag_pooled = [0u8; OUTLEN];
        $argon2
            .hash_into(PWD, SALT, &mut tag_oneshot)
            .expect("one-shot hash");
        // First: allocates. Second: the arena is a reused one, which is the
        // state every timed pooled iteration will be in.
        $hasher
            .hash_into(PWD, SALT, &mut tag_pooled)
            .expect("pooled hash");
        $hasher
            .hash_into(PWD, SALT, &mut tag_pooled)
            .expect("pooled hash");

        assert_eq!(
            tag_oneshot, tag_pooled,
            "pooled and one-shot disagree at {:?} — reuse must never change the tag",
            $p
        );
        assert_eq!(
            $hasher.reserved_blocks(),
            $p.memory_blocks() as usize,
            "the hasher is not holding on to its arena, so nothing is being reused"
        );
    };
}

/// `Argon2::hash_into` (a fresh arena every call) against `Hasher::hash_into`
/// (one arena, reused).
///
/// This is the headline A/B for the whole allocation exercise, and it is an A/B
/// between two paths that share every line of the actual computation:
/// `hash_in_arena` in `src/core.rs` is called by both. The only difference is
/// where the arena came from and what happens to it afterwards.
///
/// * one-shot: `alloc_zeroed` (posix_memalign + memset) → hash → secure wipe →
///   `free`
/// * pooled: retarget the parked arena → hash → secure wipe → park it
///
/// So the delta is exactly `posix_memalign + memset + free`, and nothing else.
/// The secure wipe is paid by both, identically. `reuse_alloc` below measures
/// that delta directly, on its own, which is the cross-check on this group.
///
/// **Read the interleaved table printed at the end of the run in preference to
/// this group at large `m_cost`.** Criterion times each benchmark as a block, so
/// a machine that drifts between the `oneshot` block and the `pooled` block
/// charges the whole drift to one of them. The summary table interleaves.
fn bench_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("reuse");

    for &(m_cost, t_cost, lanes) in &reuse_grid() {
        let p = params(m_cost, t_cost, lanes, lanes);
        reuse_pair!(argon2, hasher, p);

        group.throughput(throughput_of(&p));
        tune(&mut group, est_millis(&p));
        let label = format!("m{m_cost}_t{t_cost}_p{lanes}");

        let mut out = [0u8; OUTLEN];
        group.bench_function(BenchmarkId::new("oneshot", &label), |b| {
            b.iter(|| {
                argon2
                    .hash_into(black_box(PWD), black_box(SALT), &mut out)
                    .expect("hash");
            });
        });

        let mut out = [0u8; OUTLEN];
        group.bench_function(BenchmarkId::new("pooled", &label), |b| {
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
// 7. The memory stages, isolated
// ---------------------------------------------------------------------------

/// Block counts for `reuse_alloc`, in KiB of arena.
const ALLOC_SIZES: [usize; 5] = [64, 1024, 4096, 65536, 262_144];

/// The three things a hash does to its arena, timed one at a time.
///
/// This exists because the whole-hash A/B above has to resolve a few percent
/// inside a measurement whose run-to-run spread is a few percent. These stages
/// do not: each one is a bulk memory operation with nothing else in it, and the
/// arithmetic between them predicts the A/B result. If the two disagree, this
/// group is the one to believe.
///
/// * `alloc_free` — `Arena::new(n)` then drop. That is `posix_memalign` + the
///   zeroing memset + `free`, and **nothing else**: a freshly allocated arena is
///   flagged known-zero, so its `Drop` skips the secure wipe. This is precisely
///   and only the work the pooled path skips.
/// * `secure_wipe` — `secure_wipe_blocks` over the same bytes. **Both** paths
///   pay this, every hash, unchanged. It is here for proportion, because it is
///   several times larger than the thing being saved.
/// * `recycle` — `Workspace::acquire` then release on a warm workspace: the
///   pooled path's entire per-hash arena management, wipe included.
///
/// The identity to check is `alloc_free + secure_wipe ≈ recycle + alloc_free`,
/// i.e. `recycle ≈ secure_wipe`: recycling *is* the wipe plus a pointer
/// retarget.
fn bench_reuse_alloc(c: &mut Criterion) {
    use argon2_rust::__internal::{Arena, Workspace, secure_wipe_blocks};

    let mut group = c.benchmark_group("reuse_alloc");
    group.sampling_mode(SamplingMode::Flat);

    for &kib in &ALLOC_SIZES {
        group.throughput(Throughput::Bytes(kib as u64 * 1024));
        // A 256 MiB memset is ~1 ms and a 256 MiB volatile wipe ~5 ms, so the
        // small sizes need many samples and the big ones need few.
        let est_ms = kib as f64 * 0.000_02;
        group
            .sample_size(if est_ms >= 2.0 { 50 } else { 100 })
            .warm_up_time(Duration::from_millis(500))
            .measurement_time(Duration::from_millis((est_ms * 200.0).max(2000.0) as u64));

        let label = format!("{kib}KiB");

        group.bench_function(BenchmarkId::new("alloc_free", &label), |b| {
            b.iter(|| {
                let arena = Arena::new(black_box(kib)).expect("arena");
                black_box(arena.as_ptr());
                drop(arena);
            });
        });

        let mut arena = Arena::new(kib).expect("arena");
        group.bench_function(BenchmarkId::new("secure_wipe", &label), |b| {
            b.iter(|| secure_wipe_blocks(black_box(arena.as_mut_slice())));
        });
        drop(arena);

        let mut ws = Workspace::new();
        ws.reserve(kib).expect("reserve");
        group.bench_function(BenchmarkId::new("recycle", &label), |b| {
            b.iter(|| {
                let mut guard = ws.acquire(black_box(kib)).expect("acquire");
                // Dirty the arena, so the release wipe is a real wipe rather
                // than a branch a hash would never get to skip.
                black_box(guard.as_mut_ptr());
                drop(guard);
            });
        });
        assert_eq!(ws.capacity(), kib, "the workspace stopped reusing");
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 8. The encoded API, one-shot vs pooled
// ---------------------------------------------------------------------------

/// `hash_encoded` and `verify_encoded` on both paths, plus the raw `hash_into`
/// underneath them so the encoding cost can be read off by subtraction.
///
/// `m = 8` KiB is here because it is the smallest hash Argon2 permits, and
/// therefore the configuration in which the encoding work is the largest
/// fraction it will ever be. `m = 65536` is the RFC 9106 second recommendation,
/// i.e. what the number actually looks like in practice.
///
/// Note what this group is **not**. It is not a bump-allocator A/B, because
/// there is no bump on this path to turn off: neither `hash_encoded` nor
/// `verify_encoded` was wired to `Workspace::bump`. See `bench_bump` for the
/// counterfactual and the module docs for why it was left alone.
fn bench_encoded(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoded");

    for &(m_cost, t_cost, lanes) in &[(8u32, 1u32, 1u32), (65536, 1, 1)] {
        let p = params(m_cost, t_cost, lanes, lanes);
        reuse_pair!(argon2, hasher, p);

        let encoded = argon2.hash_encoded(PWD, SALT).expect("encode");
        Argon2::verify_encoded(&encoded, PWD, Algorithm::Argon2id).expect("verify");
        assert_eq!(
            encoded,
            hasher.hash_encoded(PWD, SALT).expect("encode"),
            "pooled and one-shot produced different PHC strings"
        );

        group.throughput(throughput_of(&p));
        tune(&mut group, est_millis(&p));
        let label = format!("m{m_cost}_t{t_cost}_p{lanes}");

        let mut out = [0u8; OUTLEN];
        group.bench_function(BenchmarkId::new("hash_into_oneshot", &label), |b| {
            b.iter(|| {
                argon2
                    .hash_into(black_box(PWD), black_box(SALT), &mut out)
                    .expect("hash");
            });
        });
        group.bench_function(BenchmarkId::new("hash_encoded_oneshot", &label), |b| {
            b.iter(|| black_box(argon2.hash_encoded(black_box(PWD), black_box(SALT))));
        });
        group.bench_function(BenchmarkId::new("verify_encoded_oneshot", &label), |b| {
            b.iter(|| {
                black_box(Argon2::verify_encoded(
                    black_box(&encoded),
                    black_box(PWD),
                    Algorithm::Argon2id,
                ))
            });
        });

        let mut out = [0u8; OUTLEN];
        group.bench_function(BenchmarkId::new("hash_into_pooled", &label), |b| {
            b.iter(|| {
                hasher
                    .hash_into(black_box(PWD), black_box(SALT), &mut out)
                    .expect("hash");
            });
        });
        group.bench_function(BenchmarkId::new("hash_encoded_pooled", &label), |b| {
            b.iter(|| black_box(hasher.hash_encoded(black_box(PWD), black_box(SALT))));
        });
        group.bench_function(BenchmarkId::new("verify_encoded_pooled", &label), |b| {
            b.iter(|| {
                black_box(hasher.verify_encoded(
                    black_box(&encoded),
                    black_box(PWD),
                    Algorithm::Argon2id,
                ))
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 9. The bumpalo counterfactual
// ---------------------------------------------------------------------------

/// The three scratch buffers one `hash_encoded` + `verify_encoded` pair
/// allocates, done with `Vec` (what the crate ships) and with the `Workspace`
/// bump (what it could have done).
///
/// Sizes are the real ones for `$argon2id$v=19$m=65536,t=2,p=1$` with a 16-byte
/// salt and a 32-byte tag: `encoded_len()` is 98, and `decode_bin` sizes its two
/// buffers `tail.len() / 4 * 3 + 3`, which is 51 for the salt (66 characters
/// left in the string) and 33 for the tag (43 left).
///
/// This measures a **counterfactual, and an optimistic one**. It prices
/// *deleting* the three allocations. Two of the three cannot be deleted at any
/// price without changing a public signature: `encode_string_alloc` hands its
/// 98-byte `Vec` straight to `String::from_utf8`, so that allocation *is* the
/// returned `String`, and `decode_bin`'s two buffers become `Decoded::salt` and
/// `Decoded::hash`. A bump in front of an owned return value adds an allocation
/// and a copy rather than removing one. So whatever number this group prints is
/// an upper bound on a saving that is mostly unavailable.
///
/// Only compiled with `--features bump-alloc`.
#[cfg(feature = "bump-alloc")]
fn bench_bump(c: &mut Criterion) {
    use argon2_rust::__internal::{Workspace, bumpalo::Bump};

    /// 98 B to encode, 51 B + 33 B to decode.
    const SCRATCH: [usize; 3] = [98, 51, 33];

    let mut group = c.benchmark_group("bump");
    group
        .sample_size(200)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(3));

    // What the crate does today.
    group.bench_function("three_vecs", |b| {
        b.iter(|| {
            for len in SCRATCH {
                let mut v: Vec<u8> = Vec::new();
                v.try_reserve(black_box(len)).expect("reserve");
                v.resize(len, 0);
                black_box(v.as_ptr());
            }
        });
    });

    // What a reused `Workspace` bump would do, wipe included — `reset_bump`
    // secure-wipes the handed-out region before resetting, so this is the
    // shippable shape and not a discount.
    let mut ws = Workspace::new();
    group.bench_function("three_bump_reset", |b| {
        b.iter(|| {
            for len in SCRATCH {
                let v = ws
                    .bump()
                    .try_alloc_slice_fill_copy(black_box(len), 0u8)
                    .expect("bump");
                black_box(v.as_ptr());
            }
            ws.reset_bump();
        });
    });

    // A fresh bump per call — the shape that looks obvious and is not.
    group.bench_function("three_fresh_bump", |b| {
        b.iter(|| {
            let bump = Bump::new();
            for len in SCRATCH {
                let v = bump
                    .try_alloc_slice_fill_copy(black_box(len), 0u8)
                    .expect("bump");
                black_box(v.as_ptr());
            }
            drop(bump);
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// Ratio summary
// ---------------------------------------------------------------------------

/// Median of `reps` runs of `f`.
fn time_millis(mut f: impl FnMut()) -> f64 {
    let start = Instant::now();
    f();
    start.elapsed().as_secs_f64() * 1000.0
}

fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn min(values: &[f64]) -> f64 {
    values.iter().copied().fold(f64::MAX, f64::min)
}

/// Relative standard deviation, in percent. The honesty column: a delta smaller
/// than this is not a result.
fn rsd_percent(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if mean <= 0.0 {
        return 0.0;
    }
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    var.sqrt() / mean * 100.0
}

/// Interleaved one-shot vs pooled at one configuration.
///
/// Two things make this trustworthy on a machine that is not idle, and both are
/// deliberate:
///
/// * **The order flips every round** (`A B` / `B A`). A monotonic drift — this
///   host warms up under sustained 256 MiB work — cancels to first order instead
///   of landing entirely on whichever arm ran second.
/// * **Warm-up rounds are untimed and run on both arms.** The first pooled hash
///   allocates; only later ones reuse. A cold hasher would measure the two paths
///   doing identical work.
///
/// Returns `(oneshot, pooled)` in round order.
fn reuse_samples(m_cost: u32, t_cost: u32, lanes: u32, reps: usize) -> (Vec<f64>, Vec<f64>) {
    let p = params(m_cost, t_cost, lanes, lanes);
    reuse_pair!(argon2, hasher, p);
    let mut out = [0u8; OUTLEN];

    let mut a = Vec::with_capacity(reps);
    let mut b = Vec::with_capacity(reps);

    // Two untimed rounds on top of the two `reuse_pair!` already ran.
    for _ in 0..2 {
        argon2.hash_into(PWD, SALT, &mut out).expect("hash");
        hasher.hash_into(PWD, SALT, &mut out).expect("hash");
    }

    for rep in 0..reps {
        // `A B` on even rounds, `B A` on odd ones, so a monotonic drift lands
        // half on each arm instead of all of it on whichever ran second.
        for arm in if rep % 2 == 0 { [0, 1] } else { [1, 0] } {
            if arm == 0 {
                a.push(time_millis(|| {
                    argon2.hash_into(PWD, SALT, &mut out).expect("hash");
                }));
            } else {
                b.push(time_millis(|| {
                    hasher.hash_into(PWD, SALT, &mut out).expect("hash");
                }));
            }
        }
    }

    assert_eq!(
        hasher.reserved_blocks(),
        p.memory_blocks() as usize,
        "the hasher stopped reusing part-way through the measurement"
    );
    (a, b)
}

/// Best of `reps` of `Arena::new(blocks)` + drop, in ms.
///
/// The whole of what the pooled path skips — `posix_memalign`, the zeroing
/// memset, `free` — and nothing more: a fresh arena is flagged known-zero, so
/// its `Drop` does not secure-wipe. Measured in the same process and the same
/// run as the A/B it is being compared against, so it is not a number quoted
/// from somewhere else.
fn alloc_free_millis(blocks: usize, reps: usize) -> f64 {
    use argon2_rust::__internal::Arena;
    let mut best = f64::MAX;
    for _ in 0..reps {
        let t = time_millis(|| {
            let arena = Arena::new(blocks).expect("arena");
            black_box(arena.as_ptr());
            drop(arena);
        });
        best = best.min(t);
    }
    best
}

/// The reuse table: one-shot vs pooled, interleaved, across the whole sweep.
///
/// Reported on the **minimum** as well as the median, because contention can
/// only ever add time — a minimum is the closest thing to the unloaded number
/// this host can produce while something else is running. The `rsd` column is
/// there so a reader can throw out any row whose delta is inside its own noise,
/// and several rows are.
///
/// The last three columns are the accountability check, and they are why this
/// table is worth more than the criterion group. `alloc` is what the one-shot
/// arm spends on allocating and zeroing, measured in this same run; `saved` is
/// what the A/B actually recovered. `got` is the ratio. If reuse worked exactly
/// as advertised, `got` would be 100% everywhere. Where it is not, the number to
/// publish is `saved`, not `alloc`.
///
/// `ARGON2_BENCH_REUSE_REPS=<n>` changes the round count (default 15).
/// `ARGON2_BENCH_REUSE_ONLY=<substr>` restricts the sweep.
/// `ARGON2_BENCH_SUMMARY=0` skips this and the ratio table.
fn reuse_summary() {
    if std::env::var("ARGON2_BENCH_SUMMARY").is_ok_and(|v| v == "0") {
        return;
    }
    let reps: usize = std::env::var("ARGON2_BENCH_REUSE_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(15);

    println!("\n=== arena reuse: fresh arena per hash vs one kept, interleaved, {reps} rounds ===");
    // NOT `summary_backend()`: `reuse_samples` drives the public `Argon2` /
    // `Hasher` API, which always dispatches to the detected backend. Printing an
    // overridden name here would label the row with a backend it did not run.
    println!("backend      : {}", argon2_rust::detected_backend());
    println!(
        "\n{:<16} {:>9} {:>9} {:>9} {:>7} {:>7} {:>8} {:>5} {:>8} {:>5}",
        "config", "1shot_med", "pool_med", "paired", "pair%", "min%", "wins", "rsd", "alloc", "got"
    );

    for &(m_cost, t_cost, lanes) in &reuse_grid() {
        let p = params(m_cost, t_cost, lanes, lanes);
        let (a, b) = reuse_samples(m_cost, t_cost, lanes, reps);
        let (a_med, b_med) = (median(&a), median(&b));
        let worst_rsd = rsd_percent(&a).max(rsd_percent(&b));
        let alloc = alloc_free_millis(p.memory_blocks() as usize, reps.max(9));

        // Per-round paired difference. `a[i]` and `b[i]` were measured
        // back-to-back, so a drift that moves both cancels in the difference —
        // which `min(a) - min(b)` does not do, because the two minima can come
        // from opposite ends of the run.
        let diffs: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
        let paired = median(&diffs);
        // Distribution-free sign test. Under "reuse changes nothing" this is a
        // fair coin, so 39/41 is not a coincidence and 22/41 is not a result.
        let wins = diffs.iter().filter(|d| **d > 0.0).count();

        println!(
            "{:<16} {a_med:>9.3} {b_med:>9.3} {paired:>9.4} {:>6.2}% {:>6.2}% {:>8} {worst_rsd:>4.1}% {alloc:>8.4} {:>4.0}%",
            format!("m{m_cost}_t{t_cost}_p{lanes}"),
            -paired / a_med * 100.0,
            (min(&b) - min(&a)) / min(&a) * 100.0,
            format!("{wins}/{reps}"),
            paired / alloc * 100.0,
        );
    }

    println!(
        "\nBoth arms run the identical computation (`hash_in_arena`). The only\n\
         difference is that the one-shot arm calls `posix_memalign`, memsets the\n\
         arena and `free`s it, and the pooled arm does not. Both secure-wipe.\n\
         `paired` is the median per-round (oneshot - pooled) in ms and is the\n\
         statistic to quote; `min%` is the same thing computed from the two\n\
         minima, which does NOT cancel drift and is here only to show how much\n\
         noisier it is. `wins` is a sign test: 50% is 'no difference'. `alloc` is\n\
         Arena::new+drop measured in this same run, `got` is paired/alloc."
    );
}

/// One interleaved sampling pass: `reps` rounds of scalar, then the detected
/// backend, then C, in that order every round.
///
/// **Round-robin, not three blocks.** This is the whole point of the function.
/// Under sustained 256 MiB work this host drifts upward by around 5% over a few
/// seconds — thermal and DVFS, confirmed by `ARGON2_BENCH_DIST`, where all
/// three implementations rise together. A block design (criterion's, and the
/// obvious hand-rolled one) charges that entire drift to whichever
/// implementation happens to run last, which is how you get a "SIMD backend is
/// slower than scalar" result. Interleaving spreads the drift evenly across all
/// three, so the *ratios* stay sound even while the absolute numbers wander.
///
/// Returns `(scalar_samples, simd_samples, c_samples)`.
#[allow(clippy::type_complexity)]
fn interleaved_samples(
    m_cost: u32,
    t_cost: u32,
    lanes: u32,
    reps: usize,
) -> (Vec<f64>, Vec<f64>, Option<Vec<f64>>) {
    let p = params(m_cost, t_cost, lanes, lanes);
    let detected = summary_backend();
    let c_hash = cref::argon2_hash();
    let mut out = [0u8; OUTLEN];

    let mut scalar = Vec::with_capacity(reps);
    let mut simd = Vec::with_capacity(reps);
    let mut c = Vec::with_capacity(reps);

    for _ in 0..reps {
        scalar.push(time_millis(|| {
            rust_hash(Backend::Scalar, Algorithm::Argon2id, &p, &mut out);
        }));
        simd.push(time_millis(|| {
            rust_hash(detected, Algorithm::Argon2id, &p, &mut out);
        }));
        if let Some(f) = c_hash {
            c.push(time_millis(|| {
                cref::hash(f, Algorithm::Argon2id, m_cost, t_cost, lanes, &mut out);
            }));
        }
    }

    let c = if c.is_empty() { None } else { Some(c) };
    (scalar, simd, c)
}

/// `ARGON2_BENCH_DIST=<m_cost>,<t_cost>,<lanes>` prints every individual sample
/// for that configuration, interleaved across scalar / detected / C.
///
/// This exists because a 256 MiB benchmark on a laptop can produce a criterion
/// confidence interval wide enough to be meaningless, and the shape of the
/// samples says which cause it was: a monotonic rise is thermal or DVFS drift,
/// two clusters is P-core/E-core migration, and isolated spikes on an otherwise
/// flat run are just another process. Interleaving the three implementations
/// rather than running them in blocks means any drift hits all three equally,
/// so the *ratio* stays honest even when the absolute numbers wander.
fn distribution_dump() {
    let Ok(spec) = std::env::var("ARGON2_BENCH_DIST") else {
        return;
    };
    let nums: Vec<u32> = spec
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let &[m_cost, t_cost, lanes] = nums.as_slice() else {
        eprintln!("ARGON2_BENCH_DIST wants '<m_cost>,<t_cost>,<lanes>', got {spec:?}");
        return;
    };

    let reps = std::env::var("ARGON2_BENCH_DIST_REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(15usize);

    let detected = summary_backend();
    let (scalar, simd, c) = interleaved_samples(m_cost, t_cost, lanes, reps);

    println!("\n=== sample distribution, m{m_cost}_t{t_cost}_p{lanes}, {reps} reps (ms) ===");
    println!(
        "{:>4} {:>10} {:>10} {:>10}",
        "rep",
        "scalar",
        detected.name(),
        "c_ref"
    );
    for rep in 0..reps {
        match c.as_ref() {
            Some(c) => println!(
                "{rep:>4} {:>10.2} {:>10.2} {:>10.2}",
                scalar[rep], simd[rep], c[rep]
            ),
            None => println!(
                "{rep:>4} {:>10.2} {:>10.2} {:>10}",
                scalar[rep], simd[rep], "-"
            ),
        }
    }

    let stat = |v: &[f64]| {
        let mut v = v.to_vec();
        v.sort_by(f64::total_cmp);
        (v[0], v[v.len() / 2], v[v.len() - 1])
    };
    let (s_min, s_med, s_max) = stat(&scalar);
    let (d_min, d_med, d_max) = stat(&simd);
    let (c_min, c_med, c_max) = c.as_ref().map_or((0.0, 0.0, 0.0), |c| stat(c));

    println!(
        "{:>4} {:>10} {:>10} {:>10}",
        "---", "------", "------", "------"
    );
    println!("{:>4} {s_min:>10.2} {d_min:>10.2} {c_min:>10.2}", "min");
    println!("{:>4} {s_med:>10.2} {d_med:>10.2} {c_med:>10.2}", "med");
    println!("{:>4} {s_max:>10.2} {d_max:>10.2} {c_max:>10.2}", "max");
    println!(
        "spread (max/min): scalar {:.2}x, {} {:.2}x, c_ref {:.2}x",
        s_max / s_min,
        detected.name(),
        d_max / d_min,
        if c_min > 0.0 { c_max / c_min } else { 0.0 }
    );
    println!(
        "A rising column is thermal/DVFS drift; two clusters is P/E-core \n\
         migration; isolated spikes are another process. Drift that hits all \n\
         three columns equally leaves the ratios intact."
    );
}

/// Criterion reports each benchmark on its own; the ratios between them are
/// what the report needs, so compute them here.
///
/// This is not a replacement for the criterion numbers — it is the companion
/// they need. Criterion gives a per-benchmark confidence interval, which is the
/// right tool for "how fast is this backend". It measures each benchmark as a
/// block, though, so on a drifting machine it is the wrong tool for "how much
/// faster is A than B". This table answers that by interleaving, and its
/// ratios should be preferred wherever the two disagree at large `m_cost`.
///
/// Set `ARGON2_BENCH_SUMMARY=0` to skip.
fn ratio_summary() {
    if std::env::var("ARGON2_BENCH_SUMMARY").is_ok_and(|v| v == "0") {
        return;
    }
    const REPS: usize = 9;

    let backends = runnable_backends();
    let detected = summary_backend();

    println!("\n=== ratio summary: interleaved, median of {REPS} (ms) ===");
    println!("host backends: {backends:?}");
    println!("detected     : {detected}");
    println!("C reference  : {}", cref::describe());
    println!(
        "\n{:<20} {:>9} {:>9} {:>9} {:>12} {:>12}",
        "config",
        "scalar",
        detected.name(),
        "c_ref",
        "simd/scalar",
        "c_ref/simd"
    );

    for &(m_cost, t_cost, lanes) in &[
        (65536u32, 1u32, 1u32),
        (65536, 3, 1),
        (65536, 1, 4),
        (262144, 1, 1),
        (262144, 3, 4),
    ] {
        let (scalar, simd, c) = interleaved_samples(m_cost, t_cost, lanes, REPS);
        let scalar = median(&scalar);
        let simd = median(&simd);
        let config = format!("m{m_cost}_t{t_cost}_p{lanes}");

        match c {
            Some(c) => {
                let c = median(&c);
                println!(
                    "{config:<20} {scalar:>9.2} {simd:>9.2} {c:>9.2} {:>11.2}x {:>11.2}x",
                    scalar / simd,
                    c / simd
                );
            }
            None => println!(
                "{config:<20} {scalar:>9.2} {simd:>9.2} {:>9} {:>11.2}x {:>12}",
                "-",
                scalar / simd,
                "-"
            ),
        }
    }
    // The footnote has to name the ISA the loaded library was really built at,
    // for the same reason `cref::describe` does: a Rust backend timed against a
    // C built at a different ISA is not a measurement of anything, and the only
    // defence is to print the truth next to the numbers.
    let c_isa = cref::isa_word();
    println!(
        "\nNOTE: c_ref is the library described above — ISA `{c_isa}`, NOT assumed.\n\
         A row is like-for-like only when the Rust backend and `{c_isa}` are the\n\
         same instruction set. `{}` vs `{c_isa}` is {}.\n\
         Rebuild the C at a matched level before quoting a ratio:\n\
         `make -C phc-winner-argon2 clean && make -C phc-winner-argon2 OPTTARGET=... libs`\n\
         (none = ref.c, x86-64 = SSE2, 'x86-64 -mssse3', 'x86-64 -mavx2', native = AVX-512).\n\
         CAUTION: with OPTTARGET=none the Makefile adds no -march at all, so ref.c is\n\
         compiled at this compiler's DEFAULT -march. If that default is above the\n\
         x86-64 baseline (Ubuntu's gcc 15 defaults to x86-64-v3, i.e. AVX2), the\n\
         result is an auto-vectorised ref.c and NOT a scalar build. The probe line\n\
         above reports what actually landed; believe it, not the OPTTARGET you typed.",
        detected.name(),
        if detected.name() == c_isa {
            "matched"
        } else {
            "NOT a like-for-like comparison"
        }
    );
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    // Diagnosing noise is a different job from measuring, so it takes over the
    // whole run instead of appending to it.
    if std::env::var_os("ARGON2_BENCH_DIST").is_some() {
        distribution_dump();
        return;
    }

    // Hand-rolled rather than `criterion_main!` so the ratio table can run
    // after the groups. `configure_from_args` keeps `--test`, `--list`,
    // `--save-baseline` and filters working exactly as usual.
    let mut criterion = Criterion::default().configure_from_args();

    // `--test` runs every benchmark once for its exit status and `--list` only
    // enumerates them; neither is a measurement, so neither should pay several
    // seconds to warm a process that is about to exit.
    let args: Vec<String> = std::env::args().collect();
    let probing = args.iter().any(|a| a == "--test" || a == "--list");
    if !probing {
        warm_up_process();
    }

    bench_grid(&mut criterion);
    bench_variants(&mut criterion);
    bench_parallel(&mut criterion);
    bench_reuse(&mut criterion);
    bench_reuse_alloc(&mut criterion);
    bench_encoded(&mut criterion);
    #[cfg(feature = "bump-alloc")]
    bench_bump(&mut criterion);
    bench_dispatch(&mut criterion);
    bench_vs_c(&mut criterion);

    criterion.final_summary();

    // `cargo test` runs bench targets with `--test`; `--list` enumerates them.
    // Neither should pay for the summary.
    if !probing {
        reuse_summary();
        ratio_summary();
    }
}
