//! Fuzzes parameter validation and the whole hash pipeline with tiny,
//! fuzzer-derived parameters: `Params` construction, initial hash, first
//! blocks, segment fill (every backend the CPU has, via the normal dispatch),
//! and finalize. Memory is capped at 64 KiB so iterations stay fast.
//!
//! The property: any `Params` the validator accepts must hash without a
//! panic, and any `Params` it rejects must fail cleanly.
#![no_main]

use argon2_rust::params::{Memory, TagLen};
use argon2_rust::{Algorithm, Argon2, Params, Version};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if data.len() < 16 {
        return;
    }
    let lanes = 1 + (data[0] as u32 % 4); // 1..=4
    let threads = 1 + (data[1] as u32 % 8); // 1..=8; threads != lanes is legal
    let m_cost = (8 * lanes) + (data[2] as u32 % 57); // valid floor .. ~64 KiB
    let t_cost = 1 + (data[3] as u32 % 3); // 1..=3
    let outlen = 4 + (data[4] as usize % 61); // 4..=64
    let algorithm = match data[5] % 3 {
        0 => Algorithm::Argon2d,
        1 => Algorithm::Argon2i,
        _ => Algorithm::Argon2id,
    };
    let version = if data[6] % 2 == 0 {
        Version::V0x10
    } else {
        Version::V0x13
    };

    // Every value is fuzzer-derived and none of them is a builder default, so
    // all five are spelled out. `build()` stays fallible on purpose: this
    // target's whole contract is "never panics", which rules out
    // `build_or_panic`.
    let Ok(params) = Params::builder()
        .memory(Memory::kib(u64::from(m_cost)))
        .passes(t_cost)
        .lanes(lanes)
        .threads(threads)
        .tag_len(TagLen::bytes(outlen as u64))
        .build()
    else {
        return;
    };
    let argon2 = Argon2::new(algorithm, version, params);

    // Split the rest into pwd / salt / ad of fuzzed shapes; salt may end up
    // shorter than 8 bytes, which must return `SaltTooShort`, never panic.
    let rest = &data[8..];
    let pwd_len = rest.len() / 3;
    let salt_len = (rest.len() - pwd_len) / 2;
    let (pwd, tail) = rest.split_at(pwd_len);
    let (salt, ad) = tail.split_at(salt_len);

    let mut out = vec![0u8; outlen];
    let _ = argon2.hash_into_with_ad(pwd, salt, ad, pwd, &mut out);
});
