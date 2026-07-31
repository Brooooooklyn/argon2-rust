//! Fuzzes the PHC string decoder through `Argon2::verify_encoded`.
//!
//! The decoder parses attacker-controlled input (any `verify_encoded` caller
//! is feeding it stored-or-worse strings), and before this target existed it
//! was covered only by a handful of hand-picked malformed strings. The
//! property: no input may panic, overflow, or index out of bounds; errors are
//! fine.
#![no_main]

use argon2_rust::{Algorithm, Argon2};
use libfuzzer_sys::fuzz_target;

/// A string the decoder accepts with a large `m=` costs a full hash per fuzz
/// iteration, which starves the fuzzer. Pre-filter those cheaply: this is NOT
/// the real parser, it just caps work on inputs that look valid-and-big; a
/// mismatch only costs one slow iteration.
fn looks_expensive(s: &str) -> bool {
    let Some(pos) = s.find("m=") else {
        return false;
    };
    let digits: String = s[pos + 2..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse::<u64>().map_or(false, |m| m > 8192)
}

fuzz_target!(|data: &[u8]| {
    let Ok(s) = std::str::from_utf8(data) else { return };
    if s.len() > 512 || looks_expensive(s) {
        return;
    }
    for algorithm in [
        Algorithm::Argon2d,
        Algorithm::Argon2i,
        Algorithm::Argon2id,
    ] {
        let _ = Argon2::verify_encoded(s, b"fuzz-password", algorithm);
    }
});
