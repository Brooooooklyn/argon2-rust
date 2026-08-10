//! Does a pooled hasher grow the process? Asked of the kernel, in a process
//! that is doing nothing else.
//!
//! # Why this is a `harness = false` binary and not two more `#[test]`s
//!
//! Resident set size is a property of a **process**. `cargo test` runs the tests
//! of one binary concurrently *in one process*, so a `#[test]` that samples
//! `ps -o rss=` around its own loop is not measuring its own loop — it is
//! measuring twenty tests, nineteen of which it did not write. That is not a
//! theory: `tests/allocation_audit.rs` had exactly these two assertions, with
//! 8 MiB and 16 MiB of slack, and the churn one failed
//! `cargo test --release --target x86_64-apple-darwin --features internal-api`
//! **six runs out of six** with 9-11 MiB of sibling contamination, while the
//! library it was accusing was allocating nothing at all. Its own in-test
//! evidence said so: the allocator spy's `freed() == 0` passed on every one of
//! those failing runs.
//!
//! Widening the slack would have been a lie, and deleting the check would have
//! given up the only measurement in the suite that comes from the kernel rather
//! than from a counter this repository wrote. So the check moved here instead.
//! `harness = false` means this file *is* `fn main()`: one process, one thread,
//! nothing else in it, no libtest scheduler. The contamination is gone by
//! construction rather than by tolerance, which is why the budgets below can be
//! tight instead of generous.
//!
//! What `tests/allocation_audit.rs` keeps is the *exact* form of the same
//! question — a thread-local allocation balance that has to come back to zero
//! byte for byte. The two are complementary and neither replaces the other:
//!
//! ```text
//!   allocation balance   the crate asked the allocator for nothing net
//!   resident set size    the kernel agrees the process did not grow
//! ```
//!
//! # Reading the numbers
//!
//! `libmalloc` on macOS never returns an arena's pages to the kernel — freeing a
//! 1 GiB arena leaves RSS exactly where it was — so this can only ever prove
//! that memory did *not* grow. It cannot prove memory was returned, and nothing
//! here claims it does.
//!
//! ```text
//! cargo test --release --features internal-api --test rss_isolation
//! ```

use std::process::ExitCode;

use argon2_rust::params::{Memory, TagLen};
use argon2_rust::{Algorithm, Argon2, Params, Version};

/// Resident set size of this process in KiB, straight from `ps(1)`.
///
/// `Hasher::reserved_blocks()` is the crate's own bookkeeping; this is the
/// kernel's. Only ever used for "did it grow", never for absolute numbers.
fn rss_kib() -> u64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .expect("ps");
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .expect("parsing ps output")
}

/// Collects failures rather than panicking on the first, so one run reports
/// everything that is wrong instead of just the earliest thing.
struct Report {
    failures: Vec<String>,
}

impl Report {
    fn check(&mut self, name: &str, grew_kib: i64, budget_kib: i64, would_leak_kib: i64) {
        let verdict = if grew_kib <= budget_kib {
            "ok"
        } else {
            "FAILED"
        };
        println!(
            "{name}: RSS {grew_kib:+} KiB (budget {budget_kib} KiB, \
             a leak would be about {would_leak_kib} KiB) ... {verdict}"
        );
        if grew_kib > budget_kib {
            self.failures.push(format!(
                "{name}: RSS grew by {grew_kib} KiB, budget is {budget_kib} KiB"
            ));
        }
    }
}

/// The headline claim, asked of the kernel: many pooled hashes, no growth.
///
/// 1 MiB arena. Leaking one arena per hash would be +614 400 KiB, so the 4 MiB
/// budget is 150x tighter than the thing it is looking for — and it can be that
/// tight only because nothing else is running in this process.
fn hundreds_of_pooled_hashes_do_not_grow_the_process(report: &mut Report) {
    let params = Params::builder()
        .memory(Memory::kib(1 << 10))
        .passes(1)
        .lanes(1)
        .tag_len(TagLen::bytes(32))
        .build()
        .expect("params");
    let mut hasher = Argon2::new(Algorithm::Argon2id, Version::V0x13, params).hasher();
    let mut tag = [0u8; 32];

    // Warm up: the first hash allocates the arena, and the allocator settles.
    for _ in 0..16 {
        hasher
            .hash_into(b"password", b"somesalt", &mut tag)
            .expect("warm");
    }

    let before = rss_kib();
    const ROUNDS: u64 = 600;
    for i in 0..ROUNDS {
        hasher
            .hash_into(b"password", b"somesalt", &mut tag)
            .unwrap_or_else(|e| panic!("hash {i}: {e}"));
    }
    let after = rss_kib();

    report.check(
        "hundreds_of_pooled_hashes_do_not_grow_the_process",
        after as i64 - before as i64,
        4 * 1024,
        (ROUNDS * 1024) as i64,
    );
    assert_eq!(
        hasher.reserved_blocks(),
        params.memory_blocks() as usize,
        "the hasher should be holding exactly one arena"
    );
}

/// Churning between two `m_cost`s must settle on the larger arena and stay
/// there, rather than reallocating on every switch or accumulating one per
/// switch. 200 hashes alternating 256 KiB / 2 MiB.
fn alternating_costs_do_not_grow_the_process(report: &mut Report) {
    let small = Params::builder()
        .memory(Memory::kib(1 << 8))
        .passes(1)
        .lanes(1)
        .tag_len(TagLen::bytes(32))
        .build()
        .expect("small");
    let large = Params::builder()
        .memory(Memory::kib(1 << 11))
        .passes(1)
        .lanes(1)
        .tag_len(TagLen::bytes(32))
        .build()
        .expect("large");
    let mut hasher = Argon2::new(Algorithm::Argon2id, Version::V0x13, small).hasher();
    let mut tag = [0u8; 32];

    // The `argon2_rust::Hasher` in this signature is the point of the type being
    // re-exported from `src/lib.rs`: before that, a hasher could only ever be a
    // `let` binding, never a parameter, a field or a `Vec` element.
    let round = |hasher: &mut argon2_rust::Hasher, tag: &mut [u8; 32]| {
        for params in [small, large] {
            hasher.set_argon2(Argon2::new(Algorithm::Argon2id, Version::V0x13, params));
            hasher
                .hash_into(b"password", b"somesalt", tag)
                .expect("hash");
        }
    };

    for _ in 0..4 {
        round(&mut hasher, &mut tag);
    }

    let before = rss_kib();
    const ROUNDS: u64 = 100;
    for _ in 0..ROUNDS {
        round(&mut hasher, &mut tag);
    }
    let after = rss_kib();

    report.check(
        "alternating_costs_do_not_grow_the_process",
        after as i64 - before as i64,
        4 * 1024,
        // Reallocating the large arena on every switch and keeping it.
        (ROUNDS * 2 * 1024) as i64,
    );
    assert_eq!(
        hasher.reserved_blocks(),
        large.memory_blocks() as usize,
        "the pool should have settled on the larger arena"
    );
}

/// The control, and it is not optional: without it, two "RSS did not grow"
/// results are equally consistent with `ps` reporting a constant.
///
/// Hold 64 MiB and touch every page of it. RSS *must* move, by roughly that
/// much. Only the lower bound is asserted — the upper is the allocator's
/// business, not this test's.
fn the_rss_probe_can_see_growth(report: &mut Report) {
    const MIB: usize = 64;
    let before = rss_kib();

    let mut held: Vec<u8> = Vec::new();
    held.try_reserve(MIB << 20).expect("64 MiB");
    held.resize(MIB << 20, 0);
    // Touch every page so the pages are resident, not merely mapped.
    for i in (0..held.len()).step_by(4096) {
        held[i] = 1;
    }
    let after = rss_kib();
    let grew = after as i64 - before as i64;
    // Keep it alive across the measurement.
    assert_eq!(held[0], 1);
    drop(held);

    let floor = ((MIB as i64) << 10) / 2;
    let verdict = if grew >= floor { "ok" } else { "FAILED" };
    println!(
        "the_rss_probe_can_see_growth: RSS {grew:+} KiB \
         (must be at least {floor} KiB) ... {verdict}"
    );
    if grew < floor {
        report.failures.push(format!(
            "the RSS probe did not notice {MIB} MiB being held: it moved {grew} KiB. \
             The two budget checks above are therefore meaningless."
        ));
    }
}

/// Whether `ps(1)` can be spawned at all.
///
/// Deliberately checks only that the *spawn* succeeded. A `ps` that runs and
/// prints something unparseable still panics in [`rss_kib`], because that is a
/// broken measurement and this file must not paper over one.
///
/// What this catches is a sandbox that denies subprocess execution — Codex's
/// `workspace-write` profile does, and so do some CI containers. There the
/// measurement is impossible rather than failing, which is not a statement
/// about the crate. Before this check, such an environment saw
/// `panicked at 'ps: PermissionDenied'` and reported the whole suite as broken.
fn ps_can_run() -> bool {
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .is_ok()
}

fn main() -> ExitCode {
    // Under Miri there is no `ps`, no real process and no resident set.
    // `tests/allocation_audit.rs` carries the part of this question that Miri
    // can actually answer.
    if cfg!(miri) {
        println!("rss_isolation: skipped under Miri (no resident set to measure)");
        return ExitCode::SUCCESS;
    }

    // Same call, for an environment that forbids running `ps` at all.
    if !ps_can_run() {
        println!(
            "rss_isolation: skipped (this environment cannot spawn `ps`, so \
             resident set size is unmeasurable here)"
        );
        return ExitCode::SUCCESS;
    }

    let mut report = Report {
        failures: Vec::new(),
    };

    // The control first: if the probe is blind, the rest of the run is noise and
    // it is better to say so before printing two reassuring numbers.
    the_rss_probe_can_see_growth(&mut report);
    hundreds_of_pooled_hashes_do_not_grow_the_process(&mut report);
    alternating_costs_do_not_grow_the_process(&mut report);

    if report.failures.is_empty() {
        println!("\nrss_isolation result: ok. 3 passed; 0 failed");
        ExitCode::SUCCESS
    } else {
        println!("\nfailures:");
        for failure in &report.failures {
            println!("    {failure}");
        }
        println!(
            "\nrss_isolation result: FAILED. {} passed; {} failed",
            3 - report.failures.len(),
            report.failures.len()
        );
        ExitCode::FAILURE
    }
}
