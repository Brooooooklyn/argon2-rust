//! Randomised differential testing of this crate against the reference C
//! implementation in `phc-winner-argon2/`.
//!
//! # How the C side is driven
//!
//! **This test shells out; it does not link.** The spec preferred linking
//! `libargon2.a` into the test binary, which needs a `build.rs` to emit
//! `cargo:rustc-link-search` plus a `Cargo.toml` feature to gate it. Neither
//! file belongs to this test's owner, so instead:
//!
//! 1. `libargon2.a` is built once with `make -C phc-winner-argon2 OPTTARGET=none`
//!    (skipped when the archive already exists). `OPTTARGET=none` makes
//!    `src/opt.c` fail its compile probe, so the Makefile falls back to
//!    `src/ref.c` — the *scalar reference* `fill_block`, which is the thing this
//!    port must agree with.
//! 2. [`HARNESS_C`] below is written to `target/differential-harness/harness.c`
//!    and compiled against that archive, once per `cargo test` process.
//! 3. The harness speaks a line protocol on stdin/stdout so one process serves
//!    a whole sweep. Spawning ~1000 processes would dominate the runtime;
//!    spawning one does not.
//!
//! Request, one line, fields separated by single spaces:
//!
//! ```text
//! <type> <version> <t_cost> <m_cost> <lanes> <threads> <outlen> <pwd> <salt> <secret> <ad>
//! ```
//!
//! `type` is `0|1|2` (`Argon2_d|Argon2_i|Argon2_id`), `version` is decimal
//! (`16` or `19`), and the four buffers are lowercase hex with `-` meaning
//! empty. Response is `OK <taghex>` or `ERR <c_error_code>`.
//!
//! The harness calls `argon2_ctx()`, not `argon2_hash()`, because `argon2_ctx`
//! is what this crate's [`Argon2::hash_into_with_ad`] mirrors: same
//! `validate_inputs()` order, no extra pre-checks, and it is the only entry
//! point that accepts `secret` and `ad`.
//!
//! # What is compared
//!
//! The *whole* `Result`, not just the tag. Every case asserts one of:
//!
//! * both sides succeed and the tags are byte-identical, or
//! * both sides fail with the same numeric code
//!   ([`Error::as_c_code`] against the C's `int`).
//!
//! So a case whose parameters are invalid is an error-parity check for free,
//! and [`error_code_parity`] then deliberately targets the ordering inside
//! `validate_inputs()`.
//!
//! # Every backend, not just the fast one
//!
//! Each batch is answered by the C **once**, then replayed against every
//! [`Driver`] this host can run: the public API (whatever `detect()` picked),
//! each [`Backend`] forced through `__internal::hash_with_backend`, and the
//! pooled path described below. The C side is the expensive half, so the extra
//! drivers cost almost nothing.
//!
//! This matters. `tests/kat.rs` pins every backend, but only at genkat.c's one
//! parameter set (`t=3 m=32 lanes=4`). Without the sweep below, a `Scalar` bug
//! that only shows at `lanes = 3` or at an `m_cost` that truncates would be
//! invisible: the public API never selects `Scalar` on a machine that has SIMD.
//!
//! | target | drivers per batch |
//! |---|---|
//! | `aarch64-apple-darwin` | public API, pooled, `Scalar`, `Neon` |
//! | `x86_64-apple-darwin` (Rosetta) | public API, pooled, `Scalar`, `Sse2` |
//! | …plus `RUSTFLAGS="-C target-feature=+avx2"` | and `Avx2` |
//!
//! # The pooled driver
//!
//! [`Driver::Pooled`] answers a whole batch through **one** `Argon2::hasher()`,
//! so every case after the first runs on an arena that still holds the previous
//! case's derived material. Its tags are compared against the same C answers as
//! every other driver, which makes the sweep a direct test of the claim that
//! *many hashes through a reused arena produce identical tags to the same
//! hashes each through a fresh arena* — mediated by an implementation that has
//! never heard of arena reuse.
//!
//! It costs one extra Rust pass per batch and adds coverage no other driver
//! has, because the batches are exactly the adversarial input a reused arena
//! needs: consecutive cases differ in algorithm, version, `m_cost`, `t_cost`,
//! `lanes`, `threads` and every buffer length, so the arena is grown, re-lent
//! as a narrow window over a larger allocation, and handed to both the
//! single-threaded and the `std::thread::scope` fill inside one allocation's
//! lifetime. [`tag_parity_reused_arena_worst_case_ordering`] then pins the
//! orderings most likely to break it, rather than leaving them to chance.
//!
//! # Coverage
//!
//! Counts below are per driver; multiply by `drivers().len()` (3 on this host)
//! for the number of Rust hashes actually checked against a C answer.
//!
//! | batch | cases | hashed | rejected | pooled: hashes / allocations |
//! |---|---|---|---|---|
//! | [`tag_parity_cost_grid`] | 588 | 468 | 120 | 468 / 9 |
//! | [`tag_parity_randomised`] | 320 | 301 | 19 | 301 / 4 |
//! | [`error_code_parity`] | 168 | 24 | 144 | 24 / 1 |
//! | [`tag_parity_buffer_shapes`] | 122 | 122 | 0 | 122 / 9 |
//! | [`tag_parity_reused_arena_worst_case_ordering`] | 44 | 44 | 0 | 44 / 1 |
//! | [`the_c_harness_is_really_running_argon2`] | 2 | 1 | 1 | 1 / 1 |
//! | **default run** | **1244** | **960** | **284** | **960 / 25** |
//! | [`tag_parity_large_params`] (`#[ignore]`) | 42 | 42 | 0 | 42 / 4 |
//!
//! The last column is the pooled driver's own coverage, printed by each batch:
//! 960 hashes against 25 arena allocations means the median hash ran on an
//! arena that had already served 38 others. The descending batch is the
//! extreme — 44 hashes, one allocation, so 43 of them ran over a window of a
//! larger dirty arena.
//!
//! Every batch asserts a floor on both counts, so a batch that degenerated
//! into all-errors (or all-successes) fails instead of quietly proving less
//! than its case count suggests.
//!
//! # Determinism
//!
//! Every byte of every password/salt/secret/ad comes from a SplitMix64 seeded
//! with the constant [`SEED`]. There is no `rand`, no clock, no thread-order
//! dependence. Failure messages carry the full case, so a failure is
//! reproducible by reading it.
//!
//! ```text
//! cargo test --test differential --features internal-api --release
//! cargo test --test differential --features internal-api --release -- --ignored
//! ```

use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use argon2_rust::__internal::{hash_with_backend, validate_inputs};
use argon2_rust::params::{Memory, TagLen};
use argon2_rust::{Algorithm, Argon2, Backend, Error, Hasher, Params, Version};

/// The one constant the whole sweep hangs off. Change it to explore a
/// different corner of the space; failures stay reproducible either way.
const SEED: u64 = 0x_A250_2015_C0FF_EE01;

// ---------------------------------------------------------------------------
// The C harness
// ---------------------------------------------------------------------------

/// Source of the C driver. Written to `target/` and compiled at test time.
///
/// Kept deliberately dumb: parse, call `argon2_ctx`, print. Every buffer is
/// allocated non-`NULL` even when its length is 0, because `validate_inputs()`
/// in `core.c` has `*_PTR_MISMATCH` branches keyed on `NULL`, and a Rust empty
/// slice is never null. Passing `NULL` here would compare two different
/// programs.
const HARNESS_C: &str = r#"/* GENERATED by tests/differential.rs - do not edit. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "argon2.h"

#define LINE_CAP (1u << 20)
#define NFIELDS 11

static char line[LINE_CAP];

static int hexval(int c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* Decode one hex token; "-" is the empty buffer. Never returns NULL on
 * success, so the NULL-pointer branches of validate_inputs() stay unreachable
 * and the C matches Rust's always-valid empty slices. */
static unsigned char *parse_hex(const char *tok, size_t *len) {
    size_t n, i;
    unsigned char *buf;

    if (strcmp(tok, "-") == 0) {
        *len = 0;
        return (unsigned char *)malloc(1);
    }

    n = strlen(tok);
    if (n % 2 != 0) return NULL;

    *len = n / 2;
    buf = (unsigned char *)malloc(n / 2 + 1);
    if (buf == NULL) return NULL;

    for (i = 0; i < n / 2; ++i) {
        int hi = hexval((unsigned char)tok[2 * i]);
        int lo = hexval((unsigned char)tok[2 * i + 1]);
        if (hi < 0 || lo < 0) {
            free(buf);
            return NULL;
        }
        buf[i] = (unsigned char)((hi << 4) | lo);
    }
    return buf;
}

int main(void) {
    while (fgets(line, (int)sizeof line, stdin) != NULL) {
        char *tok[NFIELDS];
        char *p;
        int i, rc;
        argon2_context ctx;
        argon2_type type;
        unsigned long outlen;
        unsigned char *pwd, *salt, *secret, *ad, *out;
        size_t pwdlen, saltlen, secretlen, adlen;

        if (strchr(line, '\n') == NULL && !feof(stdin)) {
            printf("BAD line-too-long\n");
            fflush(stdout);
            return 1;
        }

        p = strtok(line, " \t\r\n");
        if (p == NULL) continue; /* blank line */

        for (i = 0; i < NFIELDS; ++i) {
            if (p == NULL) {
                printf("BAD too-few-fields\n");
                fflush(stdout);
                return 1;
            }
            tok[i] = p;
            p = strtok(NULL, " \t\r\n");
        }

        type = (argon2_type)strtoul(tok[0], NULL, 10);
        outlen = strtoul(tok[6], NULL, 10);

        pwd = parse_hex(tok[7], &pwdlen);
        salt = parse_hex(tok[8], &saltlen);
        secret = parse_hex(tok[9], &secretlen);
        ad = parse_hex(tok[10], &adlen);
        out = (unsigned char *)malloc(outlen ? (size_t)outlen : 1);

        if (pwd == NULL || salt == NULL || secret == NULL || ad == NULL ||
            out == NULL) {
            printf("BAD allocation-or-hex\n");
            fflush(stdout);
            return 1;
        }

        ctx.out = out;
        ctx.outlen = (uint32_t)outlen;
        ctx.pwd = pwd;
        ctx.pwdlen = (uint32_t)pwdlen;
        ctx.salt = salt;
        ctx.saltlen = (uint32_t)saltlen;
        ctx.secret = secret;
        ctx.secretlen = (uint32_t)secretlen;
        ctx.ad = ad;
        ctx.adlen = (uint32_t)adlen;
        ctx.t_cost = (uint32_t)strtoul(tok[2], NULL, 10);
        ctx.m_cost = (uint32_t)strtoul(tok[3], NULL, 10);
        ctx.lanes = (uint32_t)strtoul(tok[4], NULL, 10);
        ctx.threads = (uint32_t)strtoul(tok[5], NULL, 10);
        ctx.version = (uint32_t)strtoul(tok[1], NULL, 10);
        ctx.allocate_cbk = NULL;
        ctx.free_cbk = NULL;
        ctx.flags = ARGON2_DEFAULT_FLAGS;

        rc = argon2_ctx(&ctx, type);

        if (rc == ARGON2_OK) {
            unsigned long j;
            fputs("OK ", stdout);
            for (j = 0; j < outlen; ++j) printf("%02x", out[j]);
            fputc('\n', stdout);
        } else {
            printf("ERR %d\n", rc);
        }
        fflush(stdout);

        free(pwd);
        free(salt);
        free(secret);
        free(ad);
        free(out);
    }
    return 0;
}
"#;

fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Build (once per process) and return the harness executable.
fn harness() -> &'static Path {
    static BUILT: OnceLock<Result<PathBuf, String>> = OnceLock::new();
    match BUILT.get_or_init(build_harness) {
        Ok(path) => path.as_path(),
        Err(why) => panic!("cannot build the C differential harness:\n{why}"),
    }
}

fn run(what: &str, command: &mut Command) -> Result<(), String> {
    let output = command
        .output()
        .map_err(|e| format!("{what}: failed to spawn: {e}"))?;
    if output.status.success() {
        return Ok(());
    }
    Err(format!(
        "{what}: exited with {}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    ))
}

fn build_harness() -> Result<PathBuf, String> {
    let root = manifest_dir();
    let c_ref = root.join("phc-winner-argon2");
    if !c_ref.join("include/argon2.h").is_file() {
        return Err(format!(
            "the C reference checkout is missing at {}",
            c_ref.display()
        ));
    }

    // 1. `make -C phc-winner-argon2 OPTTARGET=none`, only when needed.
    let archive = c_ref.join("libargon2.a");
    if !archive.is_file() {
        run(
            "make -C phc-winner-argon2 OPTTARGET=none libs",
            Command::new("make")
                .arg("-C")
                .arg(&c_ref)
                .arg("OPTTARGET=none")
                .arg("libs"),
        )?;
    }
    if !archive.is_file() {
        return Err(format!("{} still does not exist", archive.display()));
    }

    // 2. Drop the harness source next to the build output.
    let build_dir = root.join("target").join("differential-harness");
    fs::create_dir_all(&build_dir).map_err(|e| format!("mkdir {}: {e}", build_dir.display()))?;

    let source = build_dir.join("harness.c");
    let stale = fs::read_to_string(&source)
        .map(|s| s != HARNESS_C)
        .unwrap_or(true);
    if stale {
        fs::write(&source, HARNESS_C).map_err(|e| format!("write {}: {e}", source.display()))?;
    }

    // 3. Compile, unless the binary is already newer than both inputs.
    let exe = build_dir.join("harness");
    if !needs_rebuild(&exe, &[&source, &archive]) {
        return Ok(exe);
    }

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    run(
        "cc harness.c libargon2.a",
        Command::new(cc)
            .arg("-O2")
            .arg("-Wall")
            .arg("-I")
            .arg(c_ref.join("include"))
            .arg(&source)
            .arg(&archive)
            .arg("-pthread")
            .arg("-o")
            .arg(&exe),
    )?;

    Ok(exe)
}

fn needs_rebuild(target: &Path, inputs: &[&Path]) -> bool {
    let Ok(target_time) = fs::metadata(target).and_then(|m| m.modified()) else {
        return true;
    };
    inputs.iter().any(|input| {
        fs::metadata(input)
            .and_then(|m| m.modified())
            .map(|t| t > target_time)
            .unwrap_or(true)
    })
}

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

#[derive(Clone, PartialEq, Eq)]
struct Case {
    algorithm: Algorithm,
    version: Version,
    t_cost: u32,
    m_cost: u32,
    lanes: u32,
    threads: u32,
    outlen: usize,
    pwd: Vec<u8>,
    salt: Vec<u8>,
    secret: Vec<u8>,
    ad: Vec<u8>,
}

impl fmt::Debug for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} v={:#x} t={} m={} lanes={} threads={} outlen={} \
             pwd[{}] salt[{}] secret[{}] ad[{}]",
            self.algorithm,
            self.version.as_u32(),
            self.t_cost,
            self.m_cost,
            self.lanes,
            self.threads,
            self.outlen,
            self.pwd.len(),
            self.salt.len(),
            self.secret.len(),
            self.ad.len(),
        )
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn hex_or_dash(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        "-".to_string()
    } else {
        hex(bytes)
    }
}

impl Case {
    /// One request line for the harness.
    fn request(&self) -> String {
        format!(
            "{} {} {} {} {} {} {} {} {} {} {}\n",
            self.algorithm.as_u32(),
            self.version.as_u32(),
            self.t_cost,
            self.m_cost,
            self.lanes,
            self.threads,
            self.outlen,
            hex_or_dash(&self.pwd),
            hex_or_dash(&self.salt),
            hex_or_dash(&self.secret),
            hex_or_dash(&self.ad),
        )
    }

    /// What the C would print for the exact same inputs — the identity of a
    /// case for de-duplication purposes.
    fn key(&self) -> String {
        self.request()
    }

    /// This crate's answer, following `argon2_ctx()` step for step.
    ///
    /// `validate_inputs` runs first and separately because `ParamsBuilder::build`
    /// validates on construction with placeholder buffer lengths, so it would
    /// otherwise report a cost error where the C reports (say)
    /// `ARGON2_SALT_TOO_SHORT`. `params.rs` documents this and points at
    /// `validate_inputs` as the way to reproduce the C ordering exactly.
    ///
    /// The validation half is driver-independent — the backend cannot change
    /// which inputs are legal — so an invalid case yields the same `Err` for
    /// every [`Driver`], and only the hashing half actually varies.
    fn rust_result(&self, driver: Driver) -> Result<Vec<u8>, Error> {
        validate_inputs(
            self.outlen,
            self.pwd.len(),
            self.salt.len(),
            self.secret.len(),
            self.ad.len(),
            self.m_cost,
            self.t_cost,
            self.lanes,
            self.threads,
        )?;

        let params = Params::builder()
            .memory(Memory::kib(u64::from(self.m_cost)))
            .passes(self.t_cost)
            .lanes(self.lanes)
            .threads(self.threads)
            .tag_len(TagLen::bytes(self.outlen as u64))
            .build()?;

        let mut out = vec![0u8; self.outlen];
        match driver {
            Driver::PublicApi => Argon2::new(self.algorithm, self.version, params)
                .hash_into_with_ad(&self.pwd, &self.salt, &self.secret, &self.ad, &mut out)?,
            // Not reachable, and deliberately not silently handled here: the
            // entire point of `Driver::Pooled` is that its hasher outlives the
            // case, so it cannot be built from `&self`. `compare_with` owns one
            // for the whole batch and routes this variant before calling in. A
            // `_ =>` arm would let a future driver fall through to the one-shot
            // path and quietly test nothing.
            Driver::Pooled => unreachable!(
                "Driver::Pooled is answered by `pooled_result` in `compare_with`, \
                 which owns the batch-lived hasher"
            ),
            // SAFETY: `hash_with_backend` is only sound for a backend this CPU
            // can execute. `drivers()` builds the list from
            // `Backend::is_available`, so every value reaching here is runnable.
            Driver::Forced(backend) => unsafe {
                hash_with_backend(
                    backend,
                    self.algorithm,
                    self.version,
                    &params,
                    &self.pwd,
                    &self.salt,
                    &self.secret,
                    &self.ad,
                    &mut out,
                )
            }?,
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Drivers
// ---------------------------------------------------------------------------

/// Which Rust entry point produces the answer for a case.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Driver {
    /// [`Argon2::hash_into_with_ad`] — the public API, on whatever backend
    /// runtime detection picked. A fresh, freshly-zeroed arena per case. This
    /// is what a real caller gets.
    PublicApi,
    /// `Argon2::hasher()` — the same computation over an arena **reused across
    /// the whole batch**, so every case but the first starts on memory holding
    /// the previous case's derived material.
    ///
    /// Same backend as [`Driver::PublicApi`]; the only variable is where the
    /// arena came from. Any divergence is therefore attributable to reuse and
    /// nothing else, which is what makes this driver worth its runtime.
    Pooled,
    /// `__internal::hash_with_backend` — one specific backend, detection
    /// bypassed, so the slow paths get swept too.
    Forced(Backend),
}

impl fmt::Debug for Driver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Driver::PublicApi => write!(f, "public-api[{}]", argon2_rust::detected_backend()),
            Driver::Pooled => write!(f, "pooled[{}]", argon2_rust::detected_backend()),
            Driver::Forced(backend) => write!(f, "forced[{backend}]"),
        }
    }
}

/// The public API, the pooled API, and every backend this CPU can execute.
///
/// The backend list is driven off [`Backend::is_available`], so it widens by
/// itself on a host with AVX2 or AVX-512 and never returns one that would
/// fault.
fn drivers() -> Vec<Driver> {
    let mut drivers = vec![Driver::PublicApi, Driver::Pooled];
    drivers.extend(
        Backend::ALL
            .iter()
            .copied()
            .filter(|b| b.is_available())
            .map(Driver::Forced),
    );
    drivers
}

/// Answer `case` through `hasher`, following `Case::rust_result` step for step
/// so the only difference between the two is the arena.
fn pooled_result(hasher: &mut Hasher, case: &Case) -> Result<Vec<u8>, Error> {
    // Identical to `Case::rust_result`: `validate_inputs` first, so the C's
    // error *ordering* is reproduced, then construction, then the hash. Sharing
    // this order is what lets `compare` demand that every driver split the batch
    // the same way.
    validate_inputs(
        case.outlen,
        case.pwd.len(),
        case.salt.len(),
        case.secret.len(),
        case.ad.len(),
        case.m_cost,
        case.t_cost,
        case.lanes,
        case.threads,
    )?;
    let params = Params::builder()
        .memory(Memory::kib(u64::from(case.m_cost)))
        .passes(case.t_cost)
        .lanes(case.lanes)
        .threads(case.threads)
        .tag_len(TagLen::bytes(case.outlen as u64))
        .build()?;

    let mut out = vec![0u8; case.outlen];
    hasher.set_argon2(Argon2::new(case.algorithm, case.version, params));
    hasher.hash_into_with_ad(&case.pwd, &case.salt, &case.secret, &case.ad, &mut out)?;
    Ok(out)
}

/// What the C harness answered.
#[derive(Debug, PartialEq, Eq)]
enum CResult {
    Ok(Vec<u8>),
    Err(i32),
}

/// Feed a whole batch of cases through one harness process.
///
/// A dedicated writer thread pushes stdin while this thread drains stdout: the
/// batch is far larger than a pipe buffer, so writing everything first would
/// deadlock the moment the child's replies filled its own pipe.
fn run_c(cases: &[Case]) -> Vec<CResult> {
    let exe = harness();

    let mut child = Command::new(exe)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap_or_else(|e| panic!("cannot spawn {}: {e}", exe.display()));

    let mut stdin = child.stdin.take().expect("piped stdin");
    let requests: Vec<String> = cases.iter().map(Case::request).collect();
    let writer = std::thread::spawn(move || -> std::io::Result<()> {
        for request in &requests {
            stdin.write_all(request.as_bytes())?;
        }
        stdin.flush()
        // `stdin` drops here, closing the pipe so the harness sees EOF.
    });

    let stdout = child.stdout.take().expect("piped stdout");
    let mut results = Vec::with_capacity(cases.len());
    for (index, line) in BufReader::new(stdout).lines().enumerate() {
        let line = line.unwrap_or_else(|e| panic!("reading harness stdout: {e}"));
        let case = cases.get(index);
        results.push(parse_response(&line, case));
    }

    writer
        .join()
        .unwrap_or_else(|_| panic!("the harness stdin writer thread panicked"))
        .unwrap_or_else(|e| panic!("writing to the harness: {e}"));

    let status = child.wait().expect("waiting for the harness");
    assert!(status.success(), "the C harness exited with {status}");
    assert_eq!(
        results.len(),
        cases.len(),
        "the C harness answered {} of {} cases",
        results.len(),
        cases.len()
    );

    results
}

fn parse_response(line: &str, case: Option<&Case>) -> CResult {
    if let Some(tag) = line.strip_prefix("OK ") {
        let bytes: Option<Vec<u8>> = (0..tag.len() / 2)
            .map(|i| u8::from_str_radix(&tag[2 * i..2 * i + 2], 16).ok())
            .collect();
        match bytes {
            Some(bytes) if tag.len() % 2 == 0 => CResult::Ok(bytes),
            _ => panic!("the harness returned a malformed tag {tag:?} for {case:?}"),
        }
    } else if let Some(code) = line.strip_prefix("ERR ") {
        CResult::Err(
            code.trim()
                .parse()
                .unwrap_or_else(|e| panic!("bad error code {code:?} from the harness: {e}")),
        )
    } else {
        panic!("unexpected harness output {line:?} for {case:?}");
    }
}

// ---------------------------------------------------------------------------
// The comparison
// ---------------------------------------------------------------------------

/// Run a batch through both implementations and assert full agreement.
///
/// Returns `(hashed, rejected)`: how many cases produced a tag on both sides
/// and how many produced a matching error. Callers assert on those counts so a
/// batch that silently degenerated into all-errors cannot pass as coverage.
fn assert_batch_agrees(label: &str, cases: &[Case]) -> (usize, usize) {
    let c_results = run_c(cases);
    compare(label, cases, &c_results)
}

/// The assertion half of [`assert_batch_agrees`], split out so
/// [`the_comparison_catches_a_wrong_answer`] can feed it a deliberately
/// corrupted C answer and prove it fails.
///
/// Replays the batch once per [`Driver`]. Every driver must agree with the C
/// *and* produce the same `(hashed, rejected)` split — if two backends disagreed
/// about which cases hash, one of them is wrong even if both matched some tag.
fn compare(label: &str, cases: &[Case], c_results: &[CResult]) -> (usize, usize) {
    let mut agreed: Option<(Driver, (usize, usize))> = None;

    for driver in drivers() {
        let counts = compare_with(label, driver, cases, c_results);
        match agreed {
            None => agreed = Some((driver, counts)),
            Some((first, expected)) => assert_eq!(
                counts, expected,
                "{label}: {driver:?} split the batch {counts:?} but {first:?} split it {expected:?}"
            ),
        }
    }

    agreed.expect("drivers() always contains the public API").1
}

/// One pass of [`compare`] on a single [`Driver`].
fn compare_with(
    label: &str,
    driver: Driver,
    cases: &[Case],
    c_results: &[CResult],
) -> (usize, usize) {
    assert!(!cases.is_empty(), "{label}: empty batch");
    assert_eq!(cases.len(), c_results.len());

    let unique: BTreeSet<String> = cases.iter().map(Case::key).collect();
    let mut hashed = 0usize;
    let mut rejected = 0usize;

    // One hasher for the whole batch. Created here rather than per case,
    // because a per-case hasher would allocate a fresh arena every time and
    // `Driver::Pooled` would silently become a slower copy of
    // `Driver::PublicApi` — the failure mode this driver exists to rule out.
    // Unused by the other drivers, and it allocates nothing until first use.
    let mut pooled = Argon2::new(
        Algorithm::Argon2id,
        Version::V0x13,
        Params::builder()
            .memory(Memory::kib(8))
            .passes(1)
            .lanes(1)
            .tag_len(TagLen::bytes(32))
            .build()
            .expect("the minimum parameters are valid"),
    )
    .hasher();
    let mut pooled_blocks_high_water = 0usize;
    let mut pooled_allocations = 0usize;

    for (index, (case, c)) in cases.iter().zip(c_results).enumerate() {
        let rust = match driver {
            Driver::Pooled => {
                let result = pooled_result(&mut pooled, case);
                // The arena must only ever grow: a smaller `m_cost` has to
                // re-lend the existing allocation as a narrower window, never
                // replace it. Each growth is one allocation, and the assertion
                // after the loop bounds how many there may be.
                let reserved = pooled.reserved_blocks();
                assert!(
                    reserved >= pooled_blocks_high_water,
                    "{label}[{index}]: the pooled arena shrank from \
                     {pooled_blocks_high_water} to {reserved} blocks"
                );
                if reserved > pooled_blocks_high_water {
                    pooled_allocations += 1;
                    pooled_blocks_high_water = reserved;
                }
                result
            }
            _ => case.rust_result(driver),
        };

        match (c, &rust) {
            (CResult::Ok(c_tag), Ok(rust_tag)) => {
                hashed += 1;
                assert_eq!(
                    c_tag.len(),
                    case.outlen,
                    "{label}[{index}] {driver:?} {case:?}: C returned {} tag bytes, expected {}",
                    c_tag.len(),
                    case.outlen
                );
                if c_tag != rust_tag {
                    let first = c_tag
                        .iter()
                        .zip(rust_tag)
                        .position(|(a, b)| a != b)
                        .unwrap_or(0);
                    panic!(
                        "\n{label}[{index}] {driver:?}: TAG MISMATCH (seed {SEED:#018x})\n\
                         \x20 case: {case:?}\n\
                         \x20 pwd:    {}\n\
                         \x20 salt:   {}\n\
                         \x20 secret: {}\n\
                         \x20 ad:     {}\n\
                         \x20 C:    {}\n\
                         \x20 Rust: {}\n\
                         \x20 first differing byte: index {first} \
                         (C {:#04x} vs Rust {:#04x})\n\n\
                         This is a real bug in the port. Fix it; do not weaken this test.\n",
                        hex_or_dash(&case.pwd),
                        hex_or_dash(&case.salt),
                        hex_or_dash(&case.secret),
                        hex_or_dash(&case.ad),
                        hex(c_tag),
                        hex(rust_tag),
                        c_tag[first],
                        rust_tag[first],
                    );
                }
            }
            (CResult::Err(c_code), Err(rust_error)) => {
                rejected += 1;
                assert_eq!(
                    *c_code,
                    rust_error.as_c_code(),
                    "\n{label}[{index}] {driver:?}: ERROR CODE MISMATCH (seed {SEED:#018x})\n\
                     \x20 case: {case:?}\n\
                     \x20 C:    {c_code} ({:?})\n\
                     \x20 Rust: {} ({rust_error:?})\n",
                    Error::from_c_code(*c_code),
                    rust_error.as_c_code(),
                );
            }
            (CResult::Ok(c_tag), Err(rust_error)) => panic!(
                "\n{label}[{index}] {driver:?}: the C succeeded and Rust failed (seed {SEED:#018x})\n\
                 \x20 case: {case:?}\n\
                 \x20 C tag: {}\n\
                 \x20 Rust:  {rust_error:?} ({})\n",
                hex(c_tag),
                rust_error.as_c_code()
            ),
            (CResult::Err(c_code), Ok(rust_tag)) => panic!(
                "\n{label}[{index}] {driver:?}: Rust succeeded and the C failed (seed {SEED:#018x})\n\
                 \x20 case: {case:?}\n\
                 \x20 C:        {c_code} ({:?})\n\
                 \x20 Rust tag: {}\n",
                Error::from_c_code(*c_code),
                hex(rust_tag)
            ),
        }
    }

    if driver == Driver::Pooled {
        // Reuse is the whole point of this driver, so prove it happened rather
        // than assuming it. Without this, a regression that made every pooled
        // call allocate afresh would leave the driver passing while testing
        // nothing beyond `Driver::PublicApi`.
        //
        // Only meaningful once there are enough hashes to reuse across: the
        // two-case sanity batches necessarily spend one allocation on their one
        // hash, and demanding otherwise would be demanding the impossible.
        if hashed >= 8 {
            assert!(
                pooled_allocations * 4 <= hashed,
                "{label} {driver:?}: {pooled_allocations} arena allocations for \
                 {hashed} hashes — the arena is barely being reused, so this \
                 driver is not testing what it claims to"
            );
        }
        println!(
            "{label} {driver:?}: {hashed} hashes shared {pooled_allocations} arena \
             allocation(s), settling at {pooled_blocks_high_water} blocks",
        );
    }

    println!(
        "{label} {driver:?}: {} cases ({} distinct) -> {hashed} agreed on a tag, \
         {rejected} agreed on an error code",
        cases.len(),
        unique.len(),
    );
    (hashed, rejected)
}

// ---------------------------------------------------------------------------
// Deterministic PRNG
// ---------------------------------------------------------------------------

/// SplitMix64. Small, deterministic, and dependency-free — the sweep only
/// needs reproducible bytes, not statistical quality.
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

    fn below(&mut self, n: usize) -> usize {
        assert!(n > 0);
        (self.next_u64() % n as u64) as usize
    }

    fn pick<T: Copy>(&mut self, choices: &[T]) -> T {
        choices[self.below(choices.len())]
    }

    fn bytes(&mut self, len: usize) -> Vec<u8> {
        (0..len).map(|_| self.next_u64() as u8).collect()
    }
}

// ---------------------------------------------------------------------------
// The domains the spec calls out
// ---------------------------------------------------------------------------

/// `m_cost` values. The first ten are the spec's list; the rest are
/// deliberately *not* multiples of `4 * lanes` for any `lanes` in `1..=4`, so
/// the `memory_blocks = segment_length * (lanes * SYNC_POINTS)` truncation in
/// `argon2_ctx` step 2 is exercised in both directions.
const M_COSTS: &[u32] = &[
    8, 15, 16, 17, 32, 63, 64, 100, 256, 1024, // spec list
    33, 65, 129, 257, // 4*lanes never divides these for lanes in 1..=4
];

const T_COSTS: &[u32] = &[1, 2, 3, 4];
const LANES: &[u32] = &[1, 2, 3, 4];
const PWD_LENS: &[usize] = &[0, 1, 63, 64, 65, 127, 128, 129, 1000];
const SALT_LENS: &[usize] = &[8, 15, 16, 32];
const SECRET_LENS: &[usize] = &[0, 8];
const AD_LENS: &[usize] = &[0, 12];
const OUT_LENS: &[usize] = &[4, 12, 31, 32, 33, 63, 64, 65, 100, 1024];

fn algorithms() -> [Algorithm; 3] {
    Algorithm::ALL
}

fn versions() -> [Version; 2] {
    Version::ALL
}

// ---------------------------------------------------------------------------
// Test 1: the cost grid
// ---------------------------------------------------------------------------

/// Every `(type, version, m_cost, lanes, threads)` combination, with `t_cost`
/// rotating through `1..=4`.
///
/// `threads` takes both `1` and `lanes`, which is the spec's
/// "threads must not change the tag" property checked against the C rather than
/// against ourselves. Note that `m_cost` values below `8 * lanes` are kept:
/// those turn into `ARGON2_MEMORY_TOO_LITTLE` on both sides and the assertion
/// is then an error-code comparison.
#[test]
fn tag_parity_cost_grid() {
    let mut rng = Rng::new(SEED ^ 0x01);
    let mut cases = Vec::new();
    let mut t_index = 0usize;

    // Fixed buffers: this test varies costs, not I/O shapes.
    let pwd = rng.bytes(32);
    let salt = rng.bytes(16);
    let secret = rng.bytes(8);
    let ad = rng.bytes(12);

    for algorithm in algorithms() {
        for version in versions() {
            for &m_cost in M_COSTS {
                for &lanes in LANES {
                    let thread_counts: &[u32] = if lanes == 1 { &[1] } else { &[1, lanes] };
                    for &threads in thread_counts {
                        let t_cost = T_COSTS[t_index % T_COSTS.len()];
                        t_index += 1;
                        cases.push(Case {
                            algorithm,
                            version,
                            t_cost,
                            m_cost,
                            lanes,
                            threads,
                            outlen: 32,
                            pwd: pwd.clone(),
                            salt: salt.clone(),
                            secret: secret.clone(),
                            ad: ad.clone(),
                        });
                    }
                }
            }
        }
    }

    let (hashed, rejected) = assert_batch_agrees("cost-grid", &cases);
    // `m_cost` values below `8 * lanes` are intentionally kept, so both
    // outcomes must be well represented; a batch that collapsed to one or the
    // other would prove far less than the case count suggests.
    assert!(
        hashed >= 350,
        "only {hashed} of {} cases actually hashed",
        cases.len()
    );
    assert!(
        rejected >= 50,
        "only {rejected} of {} cases were rejected",
        cases.len()
    );
}

// ---------------------------------------------------------------------------
// Test 2: buffer shapes
// ---------------------------------------------------------------------------

/// Every password length crossed with every output length, then every
/// salt/secret/ad length combination.
///
/// The interesting lengths are the BLAKE2b block boundaries (63/64/65,
/// 127/128/129) and the `blake2b_long` branch at `outlen == 64`: at 64 it is a
/// single hash, at 65 it enters the 32-bytes-at-a-time loop, and at 1024 that
/// loop runs `32 + 29 * 32 + 64` times.
#[test]
fn tag_parity_buffer_shapes() {
    /// Deterministically walk the cost axes so that this test's *own* axis
    /// (buffer lengths) is the only thing it sweeps exhaustively, while the
    /// costs still vary underneath it.
    fn costs(rotate: usize) -> (Algorithm, Version, u32, u32, u32, u32) {
        let algorithm = Algorithm::ALL[rotate % Algorithm::ALL.len()];
        let version = Version::ALL[(rotate / Algorithm::ALL.len()) % Version::ALL.len()];
        let lanes = LANES[rotate % LANES.len()];
        let t_cost = T_COSTS[rotate % T_COSTS.len()];
        // Keep m_cost >= 8 * lanes so every case here actually hashes.
        let m_cost = (8 * lanes).max(M_COSTS[rotate % M_COSTS.len()]);
        let threads = if rotate.is_multiple_of(2) { 1 } else { lanes };
        (algorithm, version, t_cost, m_cost, lanes, threads)
    }

    let mut rng = Rng::new(SEED ^ 0x02);
    let mut cases = Vec::new();
    let mut rotate = 0usize;

    for &pwd_len in PWD_LENS {
        for &outlen in OUT_LENS {
            let (algorithm, version, t_cost, m_cost, lanes, threads) = costs(rotate);
            rotate += 1;
            cases.push(Case {
                algorithm,
                version,
                t_cost,
                m_cost,
                lanes,
                threads,
                outlen,
                pwd: rng.bytes(pwd_len),
                salt: rng.bytes(16),
                secret: rng.bytes(8),
                ad: rng.bytes(12),
            });
        }
    }

    for &salt_len in SALT_LENS {
        for &secret_len in SECRET_LENS {
            for &ad_len in AD_LENS {
                for &outlen in &[32usize, 1024] {
                    let (algorithm, version, t_cost, m_cost, lanes, threads) = costs(rotate);
                    rotate += 1;
                    cases.push(Case {
                        algorithm,
                        version,
                        t_cost,
                        m_cost,
                        lanes,
                        threads,
                        outlen,
                        pwd: rng.bytes(17),
                        salt: rng.bytes(salt_len),
                        secret: rng.bytes(secret_len),
                        ad: rng.bytes(ad_len),
                    });
                }
            }
        }
    }

    let (hashed, rejected) = assert_batch_agrees("buffer-shapes", &cases);
    // Every case here is deliberately valid.
    assert_eq!(rejected, 0, "buffer-shapes must not contain invalid cases");
    assert_eq!(hashed, cases.len());
}

// ---------------------------------------------------------------------------
// Test 3: randomised sweep
// ---------------------------------------------------------------------------

/// 320 configurations drawn uniformly from every domain at once.
///
/// The two structured tests each vary one axis; this one varies all of them
/// together, which is where interactions live (a short salt with 3 lanes and a
/// `m_cost` that truncates, say).
#[test]
fn tag_parity_randomised() {
    let mut rng = Rng::new(SEED ^ 0x03);
    let algorithms = algorithms();
    let versions = versions();

    let mut cases = Vec::with_capacity(320);
    for _ in 0..320 {
        let lanes = rng.pick(LANES);
        let threads = if rng.next_u64() & 1 == 0 { 1 } else { lanes };
        // A third of the draws deliberately keep m_cost as-is, so some land
        // below 8 * lanes and become error-parity checks.
        let raw = rng.pick(M_COSTS);
        let m_cost = if rng.below(3) == 0 {
            raw
        } else {
            raw.max(8 * lanes)
        };

        let pwd_len = rng.pick(PWD_LENS);
        let salt_len = rng.pick(SALT_LENS);
        let secret_len = rng.pick(SECRET_LENS);
        let ad_len = rng.pick(AD_LENS);
        let outlen = rng.pick(OUT_LENS);

        cases.push(Case {
            algorithm: rng.pick(&algorithms),
            version: rng.pick(&versions),
            t_cost: rng.pick(T_COSTS),
            m_cost,
            lanes,
            threads,
            outlen,
            pwd: rng.bytes(pwd_len),
            salt: rng.bytes(salt_len),
            secret: rng.bytes(secret_len),
            ad: rng.bytes(ad_len),
        });
    }

    let (hashed, rejected) = assert_batch_agrees("randomised", &cases);
    assert!(
        hashed >= 200,
        "only {hashed} of {} random cases hashed",
        cases.len()
    );
    assert!(
        rejected >= 10,
        "only {rejected} of {} random cases were rejected",
        cases.len()
    );
}

// ---------------------------------------------------------------------------
// Test 4: error-code parity
// ---------------------------------------------------------------------------

/// `validate_inputs()` parity, including its *ordering*.
///
/// The order of the checks in `core.c` decides which code surfaces when several
/// inputs are bad at once, so the sweep deliberately includes cases with two or
/// three simultaneous violations.
///
/// Two subtleties worth naming:
///
/// * `m_cost < 8 * lanes` is computed in `uint32_t` and **wraps**. With
///   `lanes = 0xFFFFFFFF`, `8 * lanes == 0xFFFFFFF8`, so `m_cost = 0xFFFFFFFF`
///   passes that check and the answer is `ARGON2_LANES_TOO_MANY`, whereas a
///   small `m_cost` gives `ARGON2_MEMORY_TOO_LITTLE`. Both are here.
/// * none of these cases reaches `allocate_memory()`, so the enormous `m_cost`
///   values below never allocate anything.
#[test]
fn error_code_parity() {
    let mut rng = Rng::new(SEED ^ 0x04);
    let pwd = rng.bytes(24);
    let good_salt = rng.bytes(16);
    let secret = rng.bytes(8);
    let ad = rng.bytes(12);

    let base = Case {
        algorithm: Algorithm::Argon2id,
        version: Version::V0x13,
        t_cost: 2,
        m_cost: 64,
        lanes: 2,
        threads: 2,
        outlen: 32,
        pwd: pwd.clone(),
        salt: good_salt.clone(),
        secret: secret.clone(),
        ad: ad.clone(),
    };

    /// `base` with one edit applied.
    fn variant(base: &Case, edit: impl FnOnce(&mut Case)) -> Case {
        let mut case = base.clone();
        edit(&mut case);
        case
    }

    let mut cases = Vec::new();

    // ---- outlen ---------------------------------------------------------
    for outlen in [0usize, 1, 2, 3] {
        cases.push(variant(&base, |c| c.outlen = outlen));
    }
    // The MIN_OUTLEN boundary itself must still succeed.
    cases.push(variant(&base, |c| c.outlen = 4));

    // ---- salt -----------------------------------------------------------
    for salt_len in [0usize, 1, 7] {
        let bytes = rng.bytes(salt_len);
        cases.push(variant(&base, |c| c.salt = bytes));
    }
    cases.push(variant(&base, |c| c.salt.truncate(8))); // 8 is still valid

    // ---- m_cost ---------------------------------------------------------
    for m_cost in [0u32, 1, 7] {
        cases.push(variant(&base, |c| c.m_cost = m_cost));
    }
    // At or above MIN_MEMORY but below 8 * lanes.
    cases.push(variant(&base, |c| {
        c.m_cost = 8;
        c.lanes = 4;
        c.threads = 4;
    }));
    cases.push(variant(&base, |c| {
        c.m_cost = 31;
        c.lanes = 4;
        c.threads = 4;
    }));
    // Exactly 8 * lanes: valid.
    cases.push(variant(&base, |c| {
        c.m_cost = 32;
        c.lanes = 4;
        c.threads = 4;
    }));

    // ---- t_cost ---------------------------------------------------------
    cases.push(variant(&base, |c| c.t_cost = 0));

    // ---- lanes ----------------------------------------------------------
    cases.push(variant(&base, |c| {
        // 8 * 0 == 0, so the m_cost check passes and LANES_TOO_FEW surfaces.
        c.lanes = 0;
        c.threads = 1;
    }));
    cases.push(variant(&base, |c| {
        // lanes = MAX_LANES + 1, with m_cost large enough to clear
        // `m_cost < 8 * lanes` (8 * 0x1000000 == 0x8000000).
        c.lanes = 0x0100_0000;
        c.threads = 1;
        c.m_cost = 0x0800_0000;
    }));
    cases.push(variant(&base, |c| {
        // The wrapping case: 8 * 0xFFFFFFFF == 0xFFFFFFF8 in uint32_t, so
        // m_cost = 0xFFFFFFFF clears it and LANES_TOO_MANY surfaces.
        c.lanes = 0xFFFF_FFFF;
        c.threads = 1;
        c.m_cost = 0xFFFF_FFFF;
    }));
    cases.push(variant(&base, |c| {
        // Same lanes, small m_cost: MEMORY_TOO_LITTLE wins instead.
        c.lanes = 0xFFFF_FFFF;
        c.threads = 1;
        c.m_cost = 64;
    }));

    // ---- threads --------------------------------------------------------
    cases.push(variant(&base, |c| c.threads = 0));
    cases.push(variant(&base, |c| c.threads = 0x0100_0000)); // MAX_THREADS + 1
    cases.push(variant(&base, |c| c.threads = 0x00FF_FFFF)); // MAX_THREADS: valid

    // ---- multiple violations at once (ordering matters) -----------------
    cases.push(variant(&base, |c| {
        // outlen is checked before salt: OUTPUT_TOO_SHORT wins.
        c.outlen = 2;
        c.salt = Vec::new();
        c.m_cost = 0;
        c.t_cost = 0;
        c.lanes = 0;
        c.threads = 0;
    }));
    cases.push(variant(&base, |c| {
        // salt is checked before m_cost: SALT_TOO_SHORT wins.
        c.salt = Vec::new();
        c.m_cost = 0;
        c.t_cost = 0;
        c.lanes = 0;
        c.threads = 0;
    }));
    cases.push(variant(&base, |c| {
        // m_cost is checked before t_cost: MEMORY_TOO_LITTLE wins.
        c.m_cost = 0;
        c.t_cost = 0;
        c.lanes = 0;
        c.threads = 0;
    }));
    cases.push(variant(&base, |c| {
        // t_cost is checked before lanes: TIME_TOO_SMALL wins.
        c.t_cost = 0;
        c.lanes = 0;
        c.threads = 0;
    }));
    cases.push(variant(&base, |c| {
        // lanes is checked before threads: LANES_TOO_FEW wins.
        c.lanes = 0;
        c.threads = 0;
    }));

    // ---- the same violations across every type and version --------------
    let fixed = cases.clone();
    for algorithm in algorithms() {
        for version in versions() {
            if algorithm == Algorithm::Argon2id && version == Version::V0x13 {
                continue; // already covered by `base`
            }
            for case in &fixed {
                let mut variant = case.clone();
                variant.algorithm = algorithm;
                variant.version = version;
                cases.push(variant);
            }
        }
    }

    let (hashed, rejected) = assert_batch_agrees("error-codes", &cases);
    assert!(
        rejected >= 120,
        "only {rejected} of {} cases were rejected",
        cases.len()
    );
    // The boundary cases (outlen 4, salt 8, m_cost == 8 * lanes, MAX_THREADS)
    // must still succeed, or the "valid side" of each boundary is untested.
    assert!(hashed >= 20, "only {hashed} boundary cases were accepted");

    // Nothing above should have been silently valid across the board: at least
    // one case per distinct expected error code must actually have failed.
    let codes: BTreeSet<i32> = cases
        .iter()
        .filter_map(|c| c.rust_result(Driver::PublicApi).err())
        .map(|e| e.as_c_code())
        .collect();
    println!("error codes exercised: {codes:?}");
    for expected in [
        Error::OutputTooShort,
        Error::SaltTooShort,
        Error::MemoryTooLittle,
        Error::TimeTooSmall,
        Error::LanesTooFew,
        Error::LanesTooMany,
        Error::ThreadsTooFew,
        Error::ThreadsTooMany,
    ] {
        assert!(
            codes.contains(&expected.as_c_code()),
            "the sweep never produced {expected:?} ({})",
            expected.as_c_code()
        );
    }
}

// ---------------------------------------------------------------------------
// Test 5: orderings chosen to break arena reuse
// ---------------------------------------------------------------------------

/// The sequences a reused arena is most likely to get wrong, pinned against the
/// C rather than left to the luck of the randomised sweep.
///
/// Every other batch in this file happens to exercise [`Driver::Pooled`],
/// because every batch is a sequence and the pooled driver replays it through
/// one arena. This one is *designed* for it. Three orderings, each targeting a
/// different way the reuse layer could be wrong:
///
/// 1. **Strictly descending `m_cost`.** After the first case the arena is
///    always larger than the hash needs, so every later case runs over a
///    *window* — `len()` blocks of a bigger `capacity()`. If the fill ever
///    addressed past `len()`, or the release wipe ever covered the wrong range,
///    the tag moves. This is the ordering with the least test coverage
///    elsewhere, because a randomised sweep almost never produces a long
///    descending run.
/// 2. **Sawtooth.** Alternating tiny and large, so the arena grows, is re-lent
///    narrow, grows again. Catches a growth path that forgets to carry the
///    previous contents' initialisation forward.
/// 3. **Repeats around a large hash.** The same case before and after a much
///    larger one: both copies must produce the same tag, and it must be the
///    C's. This is the interleaving property (`A, B, A`) checked against an
///    implementation that has no arena reuse at all.
///
/// Algorithm and version rotate through all six combinations across the batch,
/// so consecutive cases rarely share a configuration.
#[test]
fn tag_parity_reused_arena_worst_case_ordering() {
    let mut rng = Rng::new(SEED ^ 0x0B);
    let mut cases = Vec::new();

    // The six (algorithm, version) pairs, rotated so neighbours differ.
    let configs: Vec<(Algorithm, Version)> = algorithms()
        .into_iter()
        .flat_map(|a| versions().into_iter().map(move |v| (a, v)))
        .collect();

    let mut push = |cases: &mut Vec<Case>, m_cost: u32, lanes: u32, index: usize| {
        let (algorithm, version) = configs[index % configs.len()];
        cases.push(Case {
            algorithm,
            version,
            t_cost: 1 + (index % 3) as u32,
            m_cost,
            lanes,
            // `threads` must not move a tag, so let it disagree with `lanes`.
            threads: if index.is_multiple_of(2) { lanes } else { 1 },
            outlen: [4usize, 32, 33, 64][index % 4],
            pwd: rng.bytes(index % 17),
            salt: rng.bytes(8 + (index % 9)),
            secret: rng.bytes(index % 2 * 8),
            ad: rng.bytes(index % 3 * 4),
        });
    };

    // 1. Strictly descending m_cost, at each lane count.
    let descending: &[u32] = &[1024, 512, 257, 129, 100, 64, 33, 32, 17, 16, 15, 8];
    let mut index = 0usize;
    for &lanes in &[1u32, 2, 4] {
        for &m_cost in descending {
            if m_cost >= 8 * lanes {
                push(&mut cases, m_cost, lanes, index);
                index += 1;
            }
        }
    }

    // 2. Sawtooth: tiny, large, tiny, larger, ...
    for (step, &m_cost) in [8u32, 1024, 15, 512, 8, 257, 33, 1024, 16, 64]
        .iter()
        .enumerate()
    {
        push(&mut cases, m_cost, 1 + (step % 4) as u32, index);
        index += 1;
        // Guard against a lane count the m_cost cannot support.
        if let Some(last) = cases.last()
            && last.m_cost < 8 * last.lanes
        {
            cases.pop();
        }
    }

    // 3. `A, B, A`: the same small case before and after a much larger one, so
    //    the repeat runs on an arena that has since been grown and refilled.
    //    Built once and cloned — `push` draws fresh bytes from the RNG every
    //    call, so two `push`es with the same parameters would not be the same
    //    case at all, and the interleaving claim would be empty.
    push(&mut cases, 64, 2, index);
    index += 1;
    let anchor = cases.last().expect("just pushed").clone();
    for _ in 0..2 {
        push(&mut cases, 1024, 4, index);
        index += 1;
        cases.push(anchor.clone());
    }

    // The repeats must really be repeats.
    let anchor_key = anchor.key();
    let repeats = cases.iter().filter(|c| c.key() == anchor_key).count();
    assert_eq!(
        repeats, 3,
        "the anchor case must appear identically three times (A, B, A, B, A)"
    );

    // And the batch must actually descend, or ordering 1 proves nothing.
    let descents = cases
        .windows(2)
        .filter(|w| w[1].m_cost < w[0].m_cost)
        .count();
    assert!(
        descents >= 20,
        "only {descents} descending steps; ordering 1 is not being exercised"
    );

    let (hashed, rejected) = assert_batch_agrees("reuse-ordering", &cases);
    assert_eq!(rejected, 0, "every case here must be valid");
    assert_eq!(hashed, cases.len());
    println!("reuse-ordering: {hashed} cases, {descents} descending steps");
}

// ---------------------------------------------------------------------------
// Test 6: expensive parameters
// ---------------------------------------------------------------------------

/// Realistic, slow parameters. `#[ignore]`d so the default run stays quick.
///
/// ```text
/// cargo test --test differential --features internal-api --release -- --ignored
/// ```
#[test]
#[ignore = "allocates up to 1 GiB and takes seconds; run explicitly"]
fn tag_parity_large_params() {
    let mut rng = Rng::new(SEED ^ 0x05);
    let mut cases = Vec::new();

    for (m_cost, t_cost, lanes, threads, outlen) in [
        (1u32 << 14, 3u32, 1u32, 1u32, 32usize), // 16 MiB
        (1 << 16, 2, 4, 4, 32),                  // 64 MiB, 4 lanes
        (1 << 16, 2, 4, 1, 32),                  // same, single-threaded
        (1 << 16, 1, 8, 8, 1024),                // 8 lanes, long tag
        (1 << 18, 1, 2, 2, 64),                  // 256 MiB
        (65_537, 2, 3, 3, 33),                   // awkward m_cost, 3 lanes
        (1 << 20, 1, 4, 4, 32),                  // 1 GiB, the TEST_LARGE_RAM size
    ] {
        for algorithm in algorithms() {
            for version in versions() {
                cases.push(Case {
                    algorithm,
                    version,
                    t_cost,
                    m_cost,
                    lanes,
                    threads,
                    outlen,
                    pwd: rng.bytes(32),
                    salt: rng.bytes(16),
                    secret: rng.bytes(8),
                    ad: rng.bytes(12),
                });
            }
        }
    }

    let (hashed, rejected) = assert_batch_agrees("large-params", &cases);
    assert_eq!(rejected, 0, "large-params must not contain invalid cases");
    assert_eq!(hashed, cases.len());
}

// ---------------------------------------------------------------------------
// Meta-test: the harness itself must be able to disagree
// ---------------------------------------------------------------------------

/// Sanity: the harness really runs Argon2 and really reports errors.
///
/// If `run_c` silently echoed whatever Rust computed, every test above would
/// pass vacuously. This pins one known-answer tag from `src/test.c` and one
/// known error, both produced by the C, with the expected values hard-coded
/// here rather than taken from this crate.
#[test]
fn the_c_harness_is_really_running_argon2() {
    // src/test.c:155 — `hashtest(version, 2, 16, 1, "password", "somesalt", ...,
    // Argon2_i)`, and `hashtest` passes `1 << m`, so m_cost is 65536. The
    // encoded form on the next line spells that out: `m=65536,t=2,p=1`.
    let vector = Case {
        algorithm: Algorithm::Argon2i,
        version: Version::V0x13,
        t_cost: 2,
        m_cost: 1 << 16,
        lanes: 1,
        threads: 1,
        outlen: 32,
        pwd: b"password".to_vec(),
        salt: b"somesalt".to_vec(),
        secret: Vec::new(),
        ad: Vec::new(),
    };
    // A case the C must reject.
    let mut broken = vector.clone();
    broken.m_cost = 1;

    let results = run_c(&[vector.clone(), broken.clone()]);

    assert_eq!(
        results[0],
        CResult::Ok(
            (0..32)
                .map(|i| {
                    let hex = "c1628832147d9720c5bd1cfd61367078729f6dfb6f8fea9ff98158e0d7816ed0";
                    u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).expect("hex")
                })
                .collect()
        ),
        "the harness did not reproduce the official argon2i vector from src/test.c"
    );

    assert_eq!(
        results[1],
        CResult::Err(Error::MemoryTooLittle.as_c_code()),
        "the harness did not report ARGON2_MEMORY_TOO_LITTLE for m_cost = 1"
    );

    // And the comparison plumbing agrees on both.
    assert_batch_agrees("harness-sanity", &[vector, broken]);
}

/// Negative control for [`compare`].
///
/// Every "the tags matched" claim above is only worth as much as the
/// comparison's ability to notice when they do not. Corrupt the C's answers —
/// one flipped tag bit, one wrong error code, one success/failure swap — and
/// check each is caught with a message that names what went wrong.
#[test]
fn the_comparison_catches_a_wrong_answer() {
    let valid = Case {
        algorithm: Algorithm::Argon2id,
        version: Version::V0x13,
        t_cost: 2,
        m_cost: 64,
        lanes: 2,
        threads: 2,
        outlen: 32,
        pwd: b"password".to_vec(),
        salt: b"somesalt".to_vec(),
        secret: b"secret!!".to_vec(),
        ad: b"associated..".to_vec(),
    };
    let mut invalid = valid.clone();
    invalid.t_cost = 0;

    let truth = run_c(&[valid.clone(), invalid.clone()]);
    let CResult::Ok(good_tag) = &truth[0] else {
        panic!("the valid case must hash on the C side, got {:?}", truth[0]);
    };
    assert_eq!(truth[1], CResult::Err(Error::TimeTooSmall.as_c_code()));

    // Baseline: uncorrupted, this must pass.
    assert_eq!(
        compare(
            "control-baseline",
            &[valid.clone(), invalid.clone()],
            &truth
        ),
        (1, 1)
    );

    let expect_failure = |what: &str, cases: Vec<Case>, results: Vec<CResult>, needle: &str| {
        let payload = std::panic::catch_unwind(move || {
            compare("negative-control", &cases, &results);
        })
        .err()
        .unwrap_or_else(|| panic!("{what}: a corrupted C answer was accepted"));

        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("")
            .to_string();
        assert!(
            message.contains(needle),
            "{what}: the failure message should mention {needle:?}, got:\n{message}"
        );
    };

    // 1. One flipped bit in the tag.
    let mut flipped = good_tag.clone();
    flipped[17] ^= 0x40;
    expect_failure(
        "flipped tag bit",
        vec![valid.clone()],
        vec![CResult::Ok(flipped)],
        "TAG MISMATCH",
    );

    // 2. The right failure, the wrong code.
    expect_failure(
        "wrong error code",
        vec![invalid.clone()],
        vec![CResult::Err(Error::LanesTooFew.as_c_code())],
        "ERROR CODE MISMATCH",
    );

    // 3. C rejects what Rust accepts.
    expect_failure(
        "C failed, Rust succeeded",
        vec![valid.clone()],
        vec![CResult::Err(Error::MemoryTooLittle.as_c_code())],
        "Rust succeeded and the C failed",
    );

    // 4. C accepts what Rust rejects.
    expect_failure(
        "C succeeded, Rust failed",
        vec![invalid],
        vec![CResult::Ok(good_tag.clone())],
        "the C succeeded and Rust failed",
    );
}
