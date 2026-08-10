//! `micro` — a seconds-scale measurement harness for the Argon2 fill loop.
//!
//! Criterion (`benches/argon2.rs`) is the right tool for a publishable number:
//! it warms up, bootstraps a confidence interval and reports outliers. It also
//! costs minutes. When you are iterating on `fill_segment` you need an answer
//! in seconds, on one backend, with a spread you can see. That is this.
//!
//! ```text
//! cargo bench --bench micro --features internal-api -- --backend avx512 --m 65536 --t 1 --p 1
//! ```
//!
//! Everything it prints is wall-clock over a whole `fill_memory_blocks`, so
//! there is no per-iteration instrument in the hot loop to distort it.
//!
//! # Modes
//!
//! | `--mode` | what is timed | use it for |
//! |---|---|---|
//! | `fill` (default) | `fill_memory_blocks` only, on a warm pre-seeded arena | per-block throughput of the fill loop |
//! | `hash` | the whole `hash_with_backend`, allocation included | comparing against criterion or against the C |
//! | `seg` | each `fill_segment` call separately, single-threaded | where inside a pass the time goes |
//! | `decompose` | `t = 1, 2, 3`, least-squares → ms/pass and fixed ms | **is a difference in the kernel or around it** |
//! | `stages` | alloc / first blocks / fill / finalize / wipe+free | which stage a fixed cost is in |
//! | `memory` | faults, memset and wipe over the arena, on their own | pricing the memory operations |
//!
//! `fill` is the one to iterate against: it excludes allocation, the two
//! BLAKE2b end caps and the final wipe, so a 1% change in the compression loop
//! shows up as very nearly 1%.
//!
//! **Start with `decompose`.** A hash is `fixed + t * fill`, and on this target
//! the two terms are the same order of magnitude — at `m = 64 MiB, t = 1` the
//! fill loop is only 27% of the wall clock and memory management is 73%. A
//! ratio taken on whole hashes therefore says almost nothing about the kernel.
//! `decompose` separates them, needs no instrumentation inside either
//! implementation, and works on the C as a black box.
//!
//! `seg` splits the same work by `(pass, slice)`. Note that for Argon2id the
//! data-independent slices are also the cache-hottest ones, so that split does
//! **not** isolate address generation; run `--alg i` against `--alg d` for
//! that (they differ only in where `pseudo_rand` comes from).
//!
//! # Reading the output
//!
//! ```text
//! ns/block  med 866.4   min 864.9   max 871.2   spread 0.73%
//! ```
//!
//! `med` is the number to compare. `spread` is `(max - min) / min` over the
//! reps and is the honesty check: if it is above a couple of percent the box
//! was busy and the median is not worth quoting. Check with
//! `uptime; pgrep -a -f 'cargo|argon2|bench'` and re-run.
//!
//! # How much of a difference this can actually resolve
//!
//! Measured on the Sapphire Rapids target (4 vCPU = 2 physical cores + SMT,
//! 105 MiB L3), six independent process invocations of the same command:
//!
//! | working set | within-run spread | run-to-run range of the median |
//! |---|---|---|
//! | `-m 262144` (256 MiB, DRAM-bound) | 0.3 – 1.1% | **0.5%**, one 3% excursion in six |
//! | `-m 65536` (64 MiB, fits in L3) | 1.5 – 6.2% | 3.1% |
//!
//! **Use `-m 262144` for anything you intend to quote.** 64 MiB fits inside
//! this box's L3, so the number moves with whatever else is resident, and it is
//! also sensitive to code layout: the *same* source rebuilt after unrelated
//! edits elsewhere in this file moved `-m 65536 -t 1` by 16% (16.7 → 14.4 ms)
//! and left `-m 262144` alone. That is I-cache and branch-predictor aliasing,
//! not a real change, and it will fool you.
//!
//! So, to compare a kernel change against the code before it:
//!
//! ```text
//! # 1. before touching anything, keep the binary
//! cargo build --release --benches --features internal-api
//! cp target/release/deps/micro-* /tmp/micro-before
//!
//! # 2. make the change, rebuild, then alternate the two binaries IN ONE LOOP
//! for i in 1 2 3; do
//!   printf 'before '; /tmp/micro-before -b avx512 -m 262144 -t 3 -p 1 -r 5 -w 1 | grep ns/block
//!   printf 'after  '; target/release/deps/micro-XXXX -b avx512 -m 262144 -t 3 -p 1 -r 5 -w 1 | grep ns/block
//! done
//! ```
//!
//! Alternating is what makes it trustworthy: a machine that drifts over a
//! minute hits both arms equally. Two numbers taken minutes apart do not
//! compare, whatever their reported spread says. Running under `cargo bench -q`
//! instead of invoking the binary directly makes no measurable difference
//! (checked: 14.31–14.58 ms either way), so use whichever is convenient.
//!
//! Nothing here needs `taskset`. Pinning to one vCPU was measured and did not
//! reduce either spread.
//!
//! # Comparing against the C
//!
//! `--vs-c` additionally times `argon2_hash` from the compiled
//! `phc-winner-argon2` at the same parameters, **interleaved** rep by rep so
//! that any drift on the box hits both arms equally, and prints the ratio. It
//! forces `--mode hash`, because that is the only thing the C exposes.
//!
//! The header line reports the ISA that is genuinely in the loaded library,
//! read out of its `fill_block` bytes (see `support/cref_isa.rs`). Check it.
//! A Rust backend timed against a C built at a different ISA is not a
//! measurement of anything.
//!
//! `--sweep` runs the standard 12-cell parameter matrix
//! (m ∈ {64 MiB, 256 MiB} × t ∈ {1, 3} × p ∈ {1, 2, 4}) and prints it as a
//! table; combine with `--vs-c` for the matched-ISA comparison.
//!
//! # Safety
//!
//! `--backend` is checked against `Backend::is_available()` before anything is
//! called, so this cannot dispatch to an instruction set the CPU lacks.

use std::ffi::c_void;
use std::hint::black_box;
use std::time::Instant;

use argon2_rust::__internal::{
    Arena, Instance, Position, fill_first_blocks, fill_memory_blocks_traced, fill_segment_fn,
    finalize, hash_with_backend, initial_hash,
};
use argon2_rust::{Algorithm, Backend, Params, Version};

#[path = "support/cref_isa.rs"]
mod cref_isa;

const PWD: &[u8] = b"password";
const SALT: &[u8] = b"somesalt";
const OUTLEN: usize = 32;
const SYNC_POINTS: u32 = 4;

// ---------------------------------------------------------------------------
// Arguments
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Fill,
    Hash,
    Seg,
    Decompose,
    Stages,
    Memory,
}

struct Args {
    backend: Backend,
    alg: Algorithm,
    m_cost: u32,
    t_cost: u32,
    lanes: u32,
    threads: Option<u32>,
    reps: usize,
    warmup: usize,
    mode: Mode,
    vs_c: bool,
    sweep: bool,
    csv: bool,
}

impl Default for Args {
    fn default() -> Args {
        Args {
            backend: argon2_rust::detected_backend(),
            alg: Algorithm::Argon2id,
            m_cost: 65536,
            t_cost: 1,
            lanes: 1,
            threads: None,
            reps: 9,
            warmup: 2,
            mode: Mode::Fill,
            vs_c: false,
            sweep: false,
            csv: false,
        }
    }
}

fn parse_backend(s: &str) -> Backend {
    match s {
        "scalar" => Backend::Scalar,
        "neon" => Backend::Neon,
        "sse2" => Backend::Sse2,
        "avx2" => Backend::Avx2,
        "avx512" => Backend::Avx512,
        "detected" | "auto" => argon2_rust::detected_backend(),
        other => panic!("unknown backend `{other}` (scalar|sse2|avx2|avx512|neon|detected)"),
    }
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    // `cargo bench` hands the harness a couple of flags of its own; ignore the
    // ones that are not ours rather than failing, so `--bench` still works.
    while i < argv.len() {
        let arg = argv[i].as_str();
        let next = |i: &mut usize| -> String {
            *i += 1;
            argv.get(*i)
                .unwrap_or_else(|| panic!("{arg} needs a value"))
                .clone()
        };
        match arg {
            "-b" | "--backend" => a.backend = parse_backend(&next(&mut i)),
            "-m" | "--m" | "--m-cost" => a.m_cost = next(&mut i).parse().expect("m"),
            "-t" | "--t" | "--t-cost" => a.t_cost = next(&mut i).parse().expect("t"),
            "-p" | "--p" | "--lanes" => a.lanes = next(&mut i).parse().expect("p"),
            "--threads" => a.threads = Some(next(&mut i).parse().expect("threads")),
            "-r" | "--reps" => a.reps = next(&mut i).parse().expect("reps"),
            "-w" | "--warmup" => a.warmup = next(&mut i).parse().expect("warmup"),
            "--mode" => {
                a.mode = match next(&mut i).as_str() {
                    "fill" => Mode::Fill,
                    "hash" => Mode::Hash,
                    "seg" => Mode::Seg,
                    "decompose" => Mode::Decompose,
                    "stages" => Mode::Stages,
                    "memory" => Mode::Memory,
                    other => {
                        panic!("unknown mode `{other}` (fill|hash|seg|decompose|stages|memory)")
                    }
                }
            }
            "-a" | "--alg" => {
                a.alg = match next(&mut i).as_str() {
                    "id" | "argon2id" => Algorithm::Argon2id,
                    "i" | "argon2i" => Algorithm::Argon2i,
                    "d" | "argon2d" => Algorithm::Argon2d,
                    other => panic!("unknown alg `{other}` (id|i|d)"),
                }
            }
            "--vs-c" => a.vs_c = true,
            "--sweep" => a.sweep = true,
            "--csv" => a.csv = true,
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            // `cargo bench` passes `--bench`; ignore it.
            "--bench" | "--nocapture" => {}
            // `cargo test` runs bench targets with `--test`, and `--list` only
            // enumerates them. Neither is a measurement, so neither should pay
            // for one -- exit clean, exactly as criterion does.
            "--test" | "--list" => {
                println!(
                    "micro: --{} is a probe, not a measurement; nothing to run",
                    &arg[2..]
                );
                std::process::exit(0);
            }
            other => panic!("unknown flag `{other}` (try --help)"),
        }
        i += 1;
    }
    // `decompose` has its own C arm. Everything else can only be compared
    // through `argon2_hash`, the one entry point the C library exports, so
    // `--vs-c` implies `--mode hash` there. Resolved after the loop so flag
    // order does not matter.
    if a.vs_c && a.mode != Mode::Decompose {
        a.mode = Mode::Hash;
    }
    assert!(
        a.backend.is_available(),
        "backend {} is not available on this CPU; running it would be undefined behaviour",
        a.backend
    );
    a
}

fn print_help() {
    println!(
        "micro — seconds-scale Argon2 fill-loop timing\n\
         \n\
         cargo bench --bench micro --features internal-api -- [flags]\n\
         \n\
           -b --backend  scalar|sse2|avx2|avx512|neon|detected   (default: detected)\n\
           -m --m        m_cost in KiB                          (default: 65536)\n\
           -t --t        t_cost                                 (default: 1)\n\
           -p --lanes    lanes                                  (default: 1)\n\
              --threads  threads                                (default: = lanes)\n\
           -a --alg      id|i|d                                 (default: id)\n\
           -r --reps     timed repetitions                      (default: 9)\n\
           -w --warmup   untimed repetitions first              (default: 2)\n\
              --mode     fill|hash|seg|decompose|stages|memory  (default: fill)\n\
              --vs-c     also time the compiled C, interleaved  (forces --mode hash)\n\
              --sweep    run the 12-cell m/t/p matrix\n\
              --csv      machine-readable output\n"
    );
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Median of a sample, by sorting a copy. `n` is always tiny here.
fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("no NaN"));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn min_of(xs: &[f64]) -> f64 {
    xs.iter().copied().fold(f64::INFINITY, f64::min)
}
fn max_of(xs: &[f64]) -> f64 {
    xs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
}
/// `(max - min) / min`, as a percentage. The honesty check on a run.
fn spread_pct(xs: &[f64]) -> f64 {
    let lo = min_of(xs);
    if lo <= 0.0 {
        return f64::NAN;
    }
    (max_of(xs) - lo) / lo * 100.0
}

// ---------------------------------------------------------------------------
// Work
// ---------------------------------------------------------------------------

fn make_params(m_cost: u32, t_cost: u32, lanes: u32, threads: u32) -> Params {
    Params::new_with_threads(m_cost, t_cost, lanes, threads, OUTLEN).expect("params")
}

/// Blocks a full `fill_memory_blocks` actually writes: every block of every
/// lane on every pass, less the two per lane that `fill_first_blocks` already
/// produced and pass 0 / slice 0 therefore skips.
fn blocks_per_fill(p: &Params) -> u64 {
    u64::from(p.memory_blocks()) * u64::from(p.passes()) - 2 * u64::from(p.lanes())
}

/// A seeded arena plus the `Instance` over it, rebuilt between reps.
struct Bed {
    arena: Arena,
    blockhash: [u8; 72],
    alg: Algorithm,
    params: Params,
}

impl Bed {
    fn new(alg: Algorithm, params: &Params) -> Bed {
        let arena = Arena::new(params.memory_blocks() as usize).expect("arena");
        let blockhash =
            initial_hash(alg, Version::V0x13, params, PWD, SALT, &[], &[]).expect("initial_hash");
        Bed {
            arena,
            blockhash,
            alg,
            params: *params,
        }
    }

    /// Put the arena back into the state `fill_memory_blocks` expects: blocks 0
    /// and 1 of every lane hold `H(H0 || i || lane)`.
    ///
    /// The rest of the arena is *not* reset between reps. That is deliberate:
    /// pass 0 overwrites every block unconditionally (`with_xor` is false), so
    /// the work performed is identical either way, and re-zeroing 256 MiB
    /// between reps would cost more than the measurement. `--warmup` exists so
    /// that the first-touch page faults are not in the sample.
    fn seed(&mut self) -> Instance {
        let lane_length = self.params.lane_length();
        let lanes = self.params.lanes();
        fill_first_blocks(
            &mut self.blockhash,
            self.arena.as_mut_slice(),
            lanes,
            lane_length,
        )
        .expect("fill_first_blocks");
        let n = self.params.memory_blocks() as usize;
        // SAFETY: the arena was allocated for exactly `n` blocks by
        // `Arena::new(params.memory_blocks())` above, is `ARENA_ALIGN`-aligned
        // by construction, and outlives every use of the returned `Instance`
        // (it is a field of `self`, and the `Instance` never escapes the
        // caller's stack frame). Re-deriving the pointer here, after the
        // `&mut [Block]` above has been dropped, keeps the provenance clean.
        unsafe {
            Instance::new(
                self.arena.as_mut_ptr(),
                n,
                self.alg,
                Version::V0x13,
                &self.params,
            )
        }
    }
}

/// `--mode fill`: milliseconds per `fill_memory_blocks`, one entry per rep.
fn time_fill(backend: Backend, alg: Algorithm, p: &Params, reps: usize, warmup: usize) -> Vec<f64> {
    let mut bed = Bed::new(alg, p);
    let mut out = Vec::with_capacity(reps);
    for r in 0..(warmup + reps) {
        let inst = bed.seed();
        let t0 = Instant::now();
        // SAFETY: `parse_args` asserted `backend.is_available()`, and `inst`
        // upholds `Instance::new`'s contract (see `Bed::seed`). Nothing else is
        // touching this arena.
        unsafe { fill_memory_blocks_traced(&inst, backend, None) }.expect("fill");
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        // Keep the result live so nothing above can be dead-code eliminated.
        // SAFETY: block 0 is in bounds and the fill has been joined.
        black_box(unsafe { inst.block(0) }.0[0]);
        if r >= warmup {
            out.push(dt);
        }
    }
    out
}

/// `--mode hash`: milliseconds per full `hash_with_backend`, allocation and
/// BLAKE2b end caps included. This is the mode that is comparable to criterion
/// and to the C's `argon2_hash`.
fn time_hash(backend: Backend, alg: Algorithm, p: &Params, reps: usize, warmup: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(reps);
    let mut tag = [0u8; OUTLEN];
    for r in 0..(warmup + reps) {
        let t0 = Instant::now();
        // SAFETY: `parse_args` asserted `backend.is_available()`.
        unsafe {
            hash_with_backend(
                backend,
                alg,
                Version::V0x13,
                p,
                PWD,
                SALT,
                &[],
                &[],
                &mut tag,
            )
        }
        .expect("hash");
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        black_box(tag[0]);
        if r >= warmup {
            out.push(dt);
        }
    }
    out
}

/// One `(pass, slice)` bucket of `--mode seg`.
struct SegBucket {
    pass: u32,
    slice: u32,
    /// Nanoseconds per block, one entry per rep.
    ns_per_block: Vec<f64>,
    blocks: u64,
    data_independent: bool,
}

/// `--mode seg`: time every `fill_segment` call on its own, single-threaded.
///
/// Single-threaded on purpose — with several lanes in flight, a per-segment
/// wall-clock number measures contention, not the segment.
fn time_segments(
    backend: Backend,
    alg: Algorithm,
    p: &Params,
    reps: usize,
    warmup: usize,
) -> Vec<SegBucket> {
    let st = make_params(p.memory_kib(), p.passes(), p.lanes(), 1);
    let mut bed = Bed::new(alg, &st);
    let fill = fill_segment_fn(backend);
    let seg_len = u64::from(st.segment_length());
    let lanes = u64::from(st.lanes());

    let mut buckets: Vec<SegBucket> = Vec::new();
    for pass in 0..st.passes() {
        for slice in 0..SYNC_POINTS {
            let first = pass == 0 && slice == 0;
            buckets.push(SegBucket {
                pass,
                slice,
                ns_per_block: Vec::with_capacity(reps),
                blocks: lanes * (seg_len - u64::from(first) * 2),
                data_independent: matches!(alg, Algorithm::Argon2i)
                    || (matches!(alg, Algorithm::Argon2id) && pass == 0 && slice < SYNC_POINTS / 2),
            });
        }
    }

    for r in 0..(warmup + reps) {
        let inst = bed.seed();
        for (b, bucket) in buckets.iter_mut().enumerate() {
            let pass = (b as u32) / SYNC_POINTS;
            let slice = (b as u32) % SYNC_POINTS;
            let t0 = Instant::now();
            for lane in 0..st.lanes() {
                // SAFETY: `backend.is_available()` was asserted; `inst` upholds
                // `Instance::new`; this thread is the only one filling.
                unsafe { fill(&inst, Position::new(pass, lane, slice, 0)) };
            }
            let ns = t0.elapsed().as_secs_f64() * 1e9;
            if r >= warmup {
                bucket.ns_per_block.push(ns / bucket.blocks as f64);
            }
        }
        // SAFETY: block 0 is in bounds, and every segment above has returned.
        black_box(unsafe { inst.block(0) }.0[0]);
    }
    buckets
}

// ---------------------------------------------------------------------------
// Decomposition: per-pass fill vs fixed per-hash cost
// ---------------------------------------------------------------------------

/// `t_cost` values the linear fit uses. Argon2 does exactly `t_cost` passes
/// over the arena and everything else — allocation, first-touch page faults,
/// the two BLAKE2b end caps, the wipe, the free — happens once regardless, so
/// `ms(t)` is a straight line whose slope is one pass of the fill loop and
/// whose intercept is everything that is not the fill loop.
const T_FIT: [u32; 3] = [1, 2, 3];

/// Ordinary least squares through `(x, y)`. Returns `(intercept, slope, r2)`.
fn fit(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    let n = xs.len() as f64;
    let mx = xs.iter().sum::<f64>() / n;
    let my = ys.iter().sum::<f64>() / n;
    let sxy: f64 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let sxx: f64 = xs.iter().map(|x| (x - mx) * (x - mx)).sum();
    let slope = if sxx == 0.0 { 0.0 } else { sxy / sxx };
    let intercept = my - slope * mx;
    let ss_tot: f64 = ys.iter().map(|y| (y - my) * (y - my)).sum();
    let ss_res: f64 = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| {
            let e = y - (intercept + slope * x);
            e * e
        })
        .sum();
    let r2 = if ss_tot == 0.0 {
        1.0
    } else {
        1.0 - ss_res / ss_tot
    };
    (intercept, slope, r2)
}

/// `--mode decompose`: split a hash into "one pass of fill" and "everything
/// else", for us and optionally for the C, at matched parameters.
///
/// This is the measurement that says whether a deficit lives in the
/// compression loop or in the memory management around it, and it needs no
/// instrumentation inside either implementation — only that both be timed at
/// several `t_cost` values. The C is a black box through `argon2_hash`, and
/// this works on a black box.
fn run_decompose(a: &Args) {
    let threads = a.threads.unwrap_or(a.lanes);
    let cf = if a.vs_c { cref::argon2_hash() } else { None };
    assert!(
        !a.vs_c || cf.is_some(),
        "--vs-c asked for, but the C reference library is not built"
    );

    let xs: Vec<f64> = T_FIT.iter().map(|&t| f64::from(t)).collect();
    let mut ours = Vec::new();
    let mut theirs = Vec::new();

    println!(
        "  m={} p={} threads={} {:?}, backend {}",
        a.m_cost, a.lanes, threads, a.alg, a.backend
    );
    println!(
        "  {:>3} {:>12} {:>9} {:>12} {:>9}",
        "t", "ours_ms", "spread", "c_ms", "spread"
    );
    for &t in &T_FIT {
        let p = make_params(a.m_cost, t, a.lanes, threads);
        match cf {
            Some(f) => {
                let (o, c) = interleaved(a.backend, a.alg, &p, a.reps, a.warmup, f);
                println!(
                    "  {t:>3} {:>12.3} {:>8.2}% {:>12.3} {:>8.2}%",
                    median(&o),
                    spread_pct(&o),
                    median(&c),
                    spread_pct(&c)
                );
                ours.push(median(&o));
                theirs.push(median(&c));
            }
            None => {
                let o = time_hash(a.backend, a.alg, &p, a.reps, a.warmup);
                println!(
                    "  {t:>3} {:>12.3} {:>8.2}% {:>12} {:>9}",
                    median(&o),
                    spread_pct(&o),
                    "-",
                    "-"
                );
                ours.push(median(&o));
            }
        }
    }

    let (o_fix, o_slope, o_r2) = fit(&xs, &ours);
    println!("\n  ours    fill {o_slope:>8.3} ms/pass   fixed {o_fix:>8.3} ms   (r2 {o_r2:.5})");
    if !theirs.is_empty() {
        let (c_fix, c_slope, c_r2) = fit(&xs, &theirs);
        println!("  c       fill {c_slope:>8.3} ms/pass   fixed {c_fix:>8.3} ms   (r2 {c_r2:.5})");
        println!(
            "  ours/c  fill {:>8.3}x            fixed {:>8.3}x   \
             (fixed gap {:+.2} ms, fill gap {:+.2} ms/pass)",
            o_slope / c_slope,
            o_fix / c_fix,
            o_fix - c_fix,
            o_slope - c_slope
        );
    }

    // Cross-check. `--mode fill` on a warm, reused arena, fitted the same way,
    // must reproduce the same slope: both are the marginal cost of one more
    // pass, and a pass costs the same whether or not the arena was cold. If the
    // two slopes disagree, one of the two measurements is wrong and neither
    // should be quoted.
    //
    // Fitted rather than read off `t = 1`, because `t = 1` is not a
    // representative pass: pass 0 has `with_xor == false` and never reads the
    // block it is about to overwrite, so it moves 2 KiB per block where every
    // later pass moves 3 KiB. The slope is the with-XOR pass either way.
    let warm: Vec<f64> = T_FIT
        .iter()
        .map(|&t| {
            let p = make_params(a.m_cost, t, a.lanes, threads);
            median(&time_fill(a.backend, a.alg, &p, a.reps, a.warmup))
        })
        .collect();
    let (w_fix, w_slope, w_r2) = fit(&xs, &warm);
    println!(
        "  cross-check: warm-arena fill slope {w_slope:>8.3} ms/pass \
         ({:+.1}% vs the hash slope, r2 {w_r2:.5})",
        (w_slope / o_slope - 1.0) * 100.0
    );
    println!(
        "               warm-arena fixed      {w_fix:>8.3} ms   \
         -> pass-0 discount, no allocation, no wipe;\n\
         \x20              the hash fixed cost is {:.3} ms more than that, and that \
         difference is\n\x20              allocation + first-touch page faults + the \
         final wipe.",
        o_fix - w_fix
    );
}

// ---------------------------------------------------------------------------
// Stages: where the fixed cost goes on our side
// ---------------------------------------------------------------------------

/// `--mode stages`: time the four things a hash does, separately.
///
/// Only our side — the C exposes no seam to time through. Together with
/// `--mode decompose`, which prices the C's fixed cost as a black box, this
/// says which of our stages the difference is in.
fn run_stages(a: &Args) {
    let threads = a.threads.unwrap_or(a.lanes);
    let p = make_params(a.m_cost, a.t_cost, a.lanes, threads);
    let n = p.memory_blocks() as usize;
    let lane_length = p.lane_length();
    let lanes = p.lanes();

    let mut alloc = Vec::new();
    let mut first = Vec::new();
    let mut fill = Vec::new();
    let mut fin = Vec::new();
    let mut drop_ = Vec::new();
    let mut tag = [0u8; OUTLEN];

    for r in 0..(a.warmup + a.reps) {
        let mut blockhash =
            initial_hash(a.alg, Version::V0x13, &p, PWD, SALT, &[], &[]).expect("initial_hash");

        // Allocation. On this crate's path that is `posix_memalign` plus an
        // explicit `memset` of the whole arena, because `ARENA_ALIGN` is 64 and
        // `System::alloc_zeroed` only reaches `calloc` when `align <= 16`. The
        // memset is what first-touches every page, so the page-fault cost of
        // the arena lands here rather than in the fill.
        let t0 = Instant::now();
        let mut arena = Arena::new(n).expect("arena");
        let d_alloc = t0.elapsed().as_secs_f64() * 1e3;

        let t1 = Instant::now();
        fill_first_blocks(&mut blockhash, arena.as_mut_slice(), lanes, lane_length)
            .expect("first blocks");
        let d_first = t1.elapsed().as_secs_f64() * 1e3;

        // SAFETY: the arena holds exactly `n` blocks and outlives `inst`.
        let inst = unsafe { Instance::new(arena.as_mut_ptr(), n, a.alg, Version::V0x13, &p) };
        let t2 = Instant::now();
        // SAFETY: `parse_args` asserted `backend.is_available()`.
        unsafe { fill_memory_blocks_traced(&inst, a.backend, None) }.expect("fill");
        let d_fill = t2.elapsed().as_secs_f64() * 1e3;

        let t3 = Instant::now();
        finalize(&inst, &mut tag).expect("finalize");
        let d_fin = t3.elapsed().as_secs_f64() * 1e3;

        let t4 = Instant::now();
        drop(arena);
        let d_drop = t4.elapsed().as_secs_f64() * 1e3;

        black_box(tag[0]);
        if r >= a.warmup {
            alloc.push(d_alloc);
            first.push(d_first);
            fill.push(d_fill);
            fin.push(d_fin);
            drop_.push(d_drop);
        }
    }

    let rows: [(&str, &Vec<f64>); 5] = [
        ("alloc+memset", &alloc),
        ("first_blocks", &first),
        ("fill", &fill),
        ("finalize", &fin),
        ("wipe+free", &drop_),
    ];
    let total: f64 = rows.iter().map(|(_, v)| median(v)).sum();
    println!(
        "  m={} t={} p={} threads={} backend={}   ({} reps + {} warmup)",
        p.memory_kib(),
        p.passes(),
        p.lanes(),
        p.threads(),
        a.backend,
        a.reps,
        a.warmup
    );
    println!(
        "  {:<14} {:>10} {:>8} {:>9}",
        "stage", "ms", "% total", "spread"
    );
    for (name, v) in rows {
        let m = median(v);
        println!(
            "  {name:<14} {m:>10.3} {:>7.1}% {:>8.2}%",
            m / total * 100.0,
            spread_pct(v)
        );
    }
    println!("  {:<14} {total:>10.3}", "TOTAL");
    println!(
        "  NOTE: this is a COLD arena every rep, like `Argon2::hash`. `--mode fill`\n  \
         reuses one arena and so reports the steady-state fill only; the\n  \
         difference between the two `fill` numbers is first-touch page faults."
    );
}

// ---------------------------------------------------------------------------
// Memory primitives
// ---------------------------------------------------------------------------

/// `--mode memory`: price the four memory operations a hash performs on the
/// arena, at the arena's real size.
///
/// `--mode stages` says allocation and the wipe dominate a hash at any `m_cost`
/// worth using; this says *why*, by separating the two costs that are bundled
/// inside `Arena::new`:
///
/// * **first touch** — one minor fault per 4 KiB page, unavoidable for any
///   implementation that gets fresh pages from the kernel. Both this crate and
///   the C pay it, and `perf stat -e page-faults` confirms both take the same
///   count.
/// * **the zeroing memset itself** — `ARENA_ALIGN` is 64 and
///   `System::alloc_zeroed` only reaches `calloc` when `align <= 16`, so this
///   crate gets `posix_memalign` plus an explicit `write_bytes` over the whole
///   arena. `malloc` in the C never writes those bytes: kernel pages arrive
///   zeroed and the C trusts that.
///
/// and to compare the wipe this crate performs against the one the C performs:
///
/// * **volatile wipe** — `secure_wipe_blocks`, one `write_volatile::<u64>` per
///   word, which no compiler may vectorise, coalesce or turn into `rep stosb`.
/// * **`memset` on resident memory** — the speed a non-volatile store loop
///   reaches, which is what the C's `explicit_bzero` compiles down to on glibc.
fn run_memory(a: &Args) {
    let p = make_params(a.m_cost, 1, a.lanes, 1);
    let n = p.memory_blocks() as usize;
    let mib = (n * 1024) as f64 / (1024.0 * 1024.0);

    let mut cold = Vec::new();
    let mut warm_memset = Vec::new();
    let mut wipe = Vec::new();
    let mut free = Vec::new();

    for r in 0..(a.warmup + a.reps) {
        // Cold: `posix_memalign` + `write_bytes` over pages the kernel has not
        // handed out yet, so this is faults + memset together.
        let t0 = Instant::now();
        let mut arena = Arena::new(n).expect("arena");
        let d_cold = t0.elapsed().as_secs_f64() * 1e3;

        // Warm: the same bytes again, now resident. Faults are gone, so this is
        // the memset alone -- and it is also the speed `explicit_bzero` runs at.
        let t1 = Instant::now();
        arena
            .as_mut_slice()
            .fill(argon2_rust::__internal::Block::ZERO);
        let d_warm = t1.elapsed().as_secs_f64() * 1e3;
        black_box(arena.as_slice()[0].0[0]);

        // The volatile wipe, on resident memory, so it is directly comparable
        // to the line above.
        let t2 = Instant::now();
        argon2_rust::__internal::secure_wipe_blocks(arena.as_mut_slice());
        let d_wipe = t2.elapsed().as_secs_f64() * 1e3;

        let t3 = Instant::now();
        drop(arena);
        let d_free = t3.elapsed().as_secs_f64() * 1e3;

        if r >= a.warmup {
            cold.push(d_cold);
            warm_memset.push(d_warm);
            wipe.push(d_wipe);
            free.push(d_free);
        }
    }

    println!(
        "  arena = {n} blocks = {mib:.0} MiB   ({} reps + {} warmup)",
        a.reps, a.warmup
    );
    println!(
        "  {:<34} {:>10} {:>10} {:>9}",
        "operation", "ms", "GiB/s", "spread"
    );
    let gibs = |ms: f64| mib / 1024.0 / (ms / 1e3);
    for (name, v) in [
        ("Arena::new (faults + memset)", &cold),
        ("memset, resident (~explicit_bzero)", &warm_memset),
        ("secure_wipe_blocks (volatile u64)", &wipe),
        ("drop: wipe-if-dirty + free", &free),
    ] {
        let m = median(v);
        println!(
            "  {name:<34} {m:>10.3} {:>10.2} {:>8.2}%",
            gibs(m),
            spread_pct(v)
        );
    }
    let (c, w) = (median(&cold), median(&warm_memset));
    println!(
        "  => first-touch faults alone ~= {:.3} ms  (cold - resident memset)",
        c - w
    );
    println!(
        "  => volatile wipe costs {:.2}x a plain memset over the same bytes",
        median(&wipe) / w
    );
}

// ---------------------------------------------------------------------------
// The C reference
// ---------------------------------------------------------------------------

mod cref {
    use std::ffi::{c_char, c_int, c_void};

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

    /// `RTLD_NOW`: bind everything up front, so no lazy-binding stub lands
    /// inside the first timed call.
    const RTLD_NOW: c_int = 2;

    pub const CANDIDATES: &[&str] = &[
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/phc-winner-argon2/libargon2.1.dylib\0"
        ),
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/phc-winner-argon2/libargon2.so.1\0"
        ),
    ];

    /// The loaded library's path, or `None` if it was never built.
    pub fn library_path() -> Option<&'static str> {
        for path in CANDIDATES {
            // SAFETY: `path` is a `\0`-terminated string literal; the returned
            // handle is checked for null and deliberately leaked (the library
            // stays loaded for the life of the process).
            let handle = unsafe { dlopen(path.as_ptr().cast::<c_char>(), RTLD_NOW) };
            if !handle.is_null() {
                return Some(path.trim_end_matches('\0'));
            }
        }
        None
    }

    /// `argon2_hash` out of the reference library.
    pub fn argon2_hash() -> Option<Argon2Hash> {
        for path in CANDIDATES {
            // SAFETY: as `library_path`.
            let handle = unsafe { dlopen(path.as_ptr().cast::<c_char>(), RTLD_NOW) };
            if handle.is_null() {
                continue;
            }
            // SAFETY: `handle` is live and the symbol name is `\0`-terminated.
            let symbol = unsafe { dlsym(handle, c"argon2_hash".as_ptr()) };
            if symbol.is_null() {
                continue;
            }
            // SAFETY: `argon2_hash` in `include/argon2.h` has exactly this
            // signature, and the library is the reference implementation this
            // repo vendors.
            return Some(unsafe { std::mem::transmute::<*mut c_void, Argon2Hash>(symbol) });
        }
        None
    }

    pub const fn type_of(algorithm: super::Algorithm) -> c_int {
        match algorithm {
            super::Algorithm::Argon2d => 0,
            super::Algorithm::Argon2i => 1,
            super::Algorithm::Argon2id => 2,
        }
    }
}

/// One C hash. Panics on a non-`ARGON2_OK` return: that means the two sides
/// were not asked the same question, which would silently corrupt the ratio.
fn c_hash(f: cref::Argon2Hash, alg: Algorithm, p: &Params, out: &mut [u8; OUTLEN]) {
    // SAFETY: every pointer is to a live local or a `const` slice whose length
    // is passed alongside it; `encoded` is null with length 0, which
    // `argon2_hash` documents as "raw hash only".
    let rc = unsafe {
        f(
            p.passes(),
            p.memory_kib(),
            p.lanes(),
            PWD.as_ptr().cast::<c_void>(),
            PWD.len(),
            SALT.as_ptr().cast::<c_void>(),
            SALT.len(),
            out.as_mut_ptr().cast::<c_void>(),
            OUTLEN,
            std::ptr::null_mut(),
            0,
            cref::type_of(alg),
            0x13,
        )
    };
    assert_eq!(rc, 0, "C argon2_hash failed with code {rc}");
}

/// Ours and the C's, interleaved rep by rep so drift hits both arms equally.
fn interleaved(
    backend: Backend,
    alg: Algorithm,
    p: &Params,
    reps: usize,
    warmup: usize,
    f: cref::Argon2Hash,
) -> (Vec<f64>, Vec<f64>) {
    let mut ours = Vec::with_capacity(reps);
    let mut theirs = Vec::with_capacity(reps);
    let mut tag = [0u8; OUTLEN];
    let mut ctag = [0u8; OUTLEN];
    for r in 0..(warmup + reps) {
        let t0 = Instant::now();
        // SAFETY: `parse_args` asserted `backend.is_available()`.
        unsafe {
            hash_with_backend(
                backend,
                alg,
                Version::V0x13,
                p,
                PWD,
                SALT,
                &[],
                &[],
                &mut tag,
            )
        }
        .expect("hash");
        let a = t0.elapsed().as_secs_f64() * 1e3;

        let t1 = Instant::now();
        c_hash(f, alg, p, &mut ctag);
        let b = t1.elapsed().as_secs_f64() * 1e3;

        // Free correctness check: the two arms must agree, every rep. A ratio
        // between two different computations is worthless.
        assert_eq!(tag, ctag, "Rust and C tags differ at {p:?}");

        if r >= warmup {
            ours.push(a);
            theirs.push(b);
        }
    }
    (ours, theirs)
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

fn header(a: &Args) {
    let isa = cref::library_path().and_then(cref_isa::probe);
    println!(
        "micro: backend={} alg={:?} host_detected={}",
        a.backend,
        a.alg,
        argon2_rust::detected_backend()
    );
    if a.vs_c {
        match isa {
            Some(i) => println!("C ref: {}", i.describe()),
            None => println!("C ref: NOT BUILT — run `vm.sh cref <OPTTARGET>` first"),
        }
    }
}

fn report_one(a: &Args, p: &Params, ms: &[f64], label: &str) {
    let blocks = blocks_per_fill(p);
    let med = median(ms);
    let ns_block = med * 1e6 / blocks as f64;
    if a.csv {
        println!(
            "{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.2},{:.2}",
            a.backend.name(),
            label,
            p.memory_kib(),
            p.passes(),
            p.lanes(),
            p.threads(),
            med,
            min_of(ms),
            max_of(ms),
            spread_pct(ms),
            ns_block
        );
        return;
    }
    println!(
        "  m={:<7} t={} p={} threads={}  blocks={}",
        p.memory_kib(),
        p.passes(),
        p.lanes(),
        p.threads(),
        blocks
    );
    println!(
        "  ms/{label:<5}  med {:>9.3}   min {:>9.3}   max {:>9.3}   spread {:>5.2}%",
        med,
        min_of(ms),
        max_of(ms),
        spread_pct(ms)
    );
    println!(
        "  ns/block   med {:>9.2}   ({} reps + {} warmup)",
        ns_block, a.reps, a.warmup
    );
}

fn run_single(a: &Args) {
    let threads = a.threads.unwrap_or(a.lanes);
    let p = make_params(a.m_cost, a.t_cost, a.lanes, threads);

    match a.mode {
        Mode::Fill => {
            let ms = time_fill(a.backend, a.alg, &p, a.reps, a.warmup);
            report_one(a, &p, &ms, "fill");
        }
        Mode::Hash => {
            if a.vs_c {
                let f = cref::argon2_hash().expect("C reference library not built");
                let (ours, theirs) = interleaved(a.backend, a.alg, &p, a.reps, a.warmup, f);
                let (o, t) = (median(&ours), median(&theirs));
                println!(
                    "  m={:<7} t={} p={}   ours {o:>8.2} ms (spread {:.2}%)   \
                     c {t:>8.2} ms (spread {:.2}%)   ours/c {:.3}x",
                    p.memory_kib(),
                    p.passes(),
                    p.lanes(),
                    spread_pct(&ours),
                    spread_pct(&theirs),
                    o / t
                );
            } else {
                let ms = time_hash(a.backend, a.alg, &p, a.reps, a.warmup);
                report_one(a, &p, &ms, "hash");
            }
        }
        Mode::Decompose => run_decompose(a),
        Mode::Memory => run_memory(a),
        Mode::Stages => run_stages(a),
        Mode::Seg => {
            let buckets = time_segments(a.backend, a.alg, &p, a.reps, a.warmup);
            println!(
                "  m={} t={} p={} (forced single-threaded), {:?}",
                p.memory_kib(),
                p.passes(),
                p.lanes(),
                a.alg
            );
            println!(
                "  {:>4} {:>5} {:>6} {:>12} {:>10} {:>8}",
                "pass", "slice", "addr", "ns/block", "blocks", "spread"
            );
            let mut di = Vec::new();
            let mut dd = Vec::new();
            for b in &buckets {
                let m = median(&b.ns_per_block);
                println!(
                    "  {:>4} {:>5} {:>6} {:>12.2} {:>10} {:>7.2}%",
                    b.pass,
                    b.slice,
                    if b.data_independent { "indep" } else { "dep" },
                    m,
                    b.blocks,
                    spread_pct(&b.ns_per_block)
                );
                if b.data_independent {
                    di.push(m)
                } else {
                    dd.push(m)
                }
            }
            if !di.is_empty() && !dd.is_empty() {
                let (a2, b2) = (median(&di), median(&dd));
                println!(
                    "  indep {a2:.2} vs dep {b2:.2} ns/block ({:+.1}%) -- CONFOUNDED, see below",
                    (a2 / b2 - 1.0) * 100.0
                );
                println!(
                    "  That difference is NOT the address-generation cost. The\n                       data-independent slices of Argon2id are pass 0 slices 0-1, whose\n                       reference area is small and cache-hot, so locality moves with the\n                       addressing mode and the two effects are not separable here.\n                       To price address generation, run the same (m, t, p) twice with\n                       `--alg i` and `--alg d`: those differ ONLY in where pseudo_rand\n                       comes from, and their reference distributions are the same."
                );
            }
        }
    }
}

/// The standard 12-cell matrix: m ∈ {64 MiB, 256 MiB} × t ∈ {1, 3} × p ∈ {1, 2, 4}.
const SWEEP: &[(u32, u32, u32)] = &[
    (65536, 1, 1),
    (65536, 1, 2),
    (65536, 1, 4),
    (65536, 3, 1),
    (65536, 3, 2),
    (65536, 3, 4),
    (262144, 1, 1),
    (262144, 1, 2),
    (262144, 1, 4),
    (262144, 3, 1),
    (262144, 3, 2),
    (262144, 3, 4),
];

fn run_sweep(a: &Args) {
    if a.vs_c {
        let f = cref::argon2_hash().expect("C reference library not built");
        println!(
            "\n{:<18} {:>10} {:>8} {:>10} {:>8} {:>9}",
            "config", "ours_ms", "spread", "c_ms", "spread", "ours/c"
        );
        // Keyed by (m, p) so the t = 1 and t = 3 rows can be paired up after
        // the sweep; see the derived table below.
        let mut by_mp: Vec<(u32, u32, u32, f64, f64)> = Vec::new();
        for &(m, t, p_) in SWEEP {
            let p = make_params(m, t, p_, a.threads.unwrap_or(p_));
            let (ours, theirs) = interleaved(a.backend, a.alg, &p, a.reps, a.warmup, f);
            let (o, c) = (median(&ours), median(&theirs));
            println!(
                "m{m}_t{t}_p{p_:<9} {o:>10.2} {:>7.2}% {c:>10.2} {:>7.2}% {:>8.3}x",
                spread_pct(&ours),
                spread_pct(&theirs),
                o / c
            );
            by_mp.push((m, t, p_, o, c));
        }

        // A hash is `fixed + t * fill`, so the t = 1 and t = 3 rows already in
        // the table determine both terms with no extra measurement:
        //     fill  = (t3 - t1) / 2
        //     fixed = t1 - fill
        // That split is the whole point. `ours/c` above is a ratio of two
        // numbers each dominated by allocation, first-touch page faults and
        // the final wipe; this table separates the Argon2 kernel from them.
        println!(
            "\n{:<14} {:>9} {:>9} {:>8}   {:>9} {:>9} {:>8}",
            "derived", "fill_ours", "fill_c", "ratio", "fix_ours", "fix_c", "ratio"
        );
        println!(
            "{:<14} {:>9} {:>9} {:>8}   {:>9} {:>9} {:>8}",
            "(ms/pass)", "", "", "", "(ms)", "", ""
        );
        for &(m, p_) in &[
            (65536u32, 1u32),
            (65536, 2),
            (65536, 4),
            (262144, 1),
            (262144, 2),
            (262144, 4),
        ] {
            let get = |t: u32| {
                by_mp
                    .iter()
                    .find(|r| r.0 == m && r.1 == t && r.2 == p_)
                    .copied()
            };
            let (Some(t1), Some(t3)) = (get(1), get(3)) else {
                continue;
            };
            let (of, cf) = ((t3.3 - t1.3) / 2.0, (t3.4 - t1.4) / 2.0);
            let (ox, cx) = (t1.3 - of, t1.4 - cf);
            println!(
                "m{m}_p{p_:<8} {of:>9.2} {cf:>9.2} {:>7.3}x   {ox:>9.2} {cx:>9.2} {:>7.3}x",
                of / cf,
                ox / cx
            );
        }
    } else {
        println!(
            "\n{:<18} {:>10} {:>8} {:>12}",
            "config", "ms", "spread", "ns/block"
        );
        for &(m, t, p_) in SWEEP {
            let p = make_params(m, t, p_, a.threads.unwrap_or(p_));
            let ms = match a.mode {
                Mode::Hash => time_hash(a.backend, a.alg, &p, a.reps, a.warmup),
                _ => time_fill(a.backend, a.alg, &p, a.reps, a.warmup),
            };
            let med = median(&ms);
            println!(
                "m{m}_t{t}_p{p_:<9} {med:>10.2} {:>7.2}% {:>12.2}",
                spread_pct(&ms),
                med * 1e6 / blocks_per_fill(&p) as f64
            );
        }
    }
}

fn main() {
    let a = parse_args();
    header(&a);
    if a.sweep {
        run_sweep(&a);
    } else {
        run_single(&a);
    }
}
