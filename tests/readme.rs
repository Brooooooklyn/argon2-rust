//! The README's Rust examples, compiled and run.
//!
//! Nothing else in the tree compiles `README.md`. That is not hypothetical
//! rot: the quick-start called `verify_encoded` as a *method* for as long as
//! it existed, when it is an associated function, and no build ever noticed.
//!
//! Adding `#[doc = include_str!("../README.md")]` under `cfg(doctest)` would
//! also work, but every block would then need `# `-prefixed setup lines to be
//! self-contained — and rustdoc's hidden-line marker is not hidden on GitHub
//! or crates.io, which is where this file is actually read. So the blocks stay
//! clean prose and are transcribed here instead.
//!
//! A transcription drifts. [`transcription_still_matches_the_readme`] is what
//! stops it: it re-extracts every ```rust block from `README.md` and asserts
//! each line is present verbatim below, so editing the README without editing
//! this file fails the suite.

use argon2_rust::{Algorithm, Argon2, Params, Version};

/// Block 1 (quick start), block 2 (bounded verify) and block 3 (pooled
/// hasher) in order, joined by the glue the README elides — the blocks are
/// written as one continuing session, so `encoded` and `argon2` carry over.
#[test]
fn readme_examples_compile_and_run() -> Result<(), argon2_rust::Error> {
    // --- block 1 ---
    let params = Params::new(65536, 3, 4, 32)?; // m=64 MiB, t=3, p=4, out=32 B
    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);

    let mut tag = [0u8; 32];
    argon2.hash_into(b"password", b"random salt 16B!", &mut tag)?;

    // PHC string format
    let encoded = argon2.hash_encoded(b"password", b"random salt 16B!")?;
    assert!(Argon2::verify_encoded(&encoded, b"password", Algorithm::Argon2id).is_ok());

    // Or let the crate draw a fresh 16-byte salt from the OS:
    // (an OS entropy source needs `std`; without it this shadow simply does not
    // happen and `encoded` stays the explicit-salt string from just above)
    #[cfg(feature = "std")]
    let encoded = argon2.hash_password_with_random_salt(b"password")?;

    // --- block 2 ---
    let ceiling = Params::new(1 << 16, 8, 4, 32)?; // no stored hash should exceed this
    Argon2::verify_encoded_bounded(&encoded, b"password", Algorithm::Argon2id, &ceiling)?;

    // --- block 3 ---
    let credentials = [(&b"password"[..], &b"random salt 16B!"[..])];
    let mut hasher = argon2.hasher();
    for (pwd, salt) in &credentials {
        let mut tag = [0u8; 32];
        hasher.hash_into(pwd, salt, &mut tag)?;
    }

    Ok(())
}

/// Guards the transcription above against the README moving out from under it.
///
/// Line-by-line rather than block-by-block on purpose: the README blocks are
/// fragments that share state, so they are not textually contiguous here, but
/// every *line* of code in them must still appear.
#[test]
fn transcription_still_matches_the_readme() {
    let readme = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))
        .expect("README.md must be readable from the manifest dir");
    let this_file =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/readme.rs"))
            .expect("this test file must be readable");

    // Compare code, not layout: drop the trailing comment and collapse runs of
    // whitespace, so `cargo fmt` reflowing this file (it does — it closes up
    // the README's aligned `//` columns) is not a failure.
    fn normalize(line: &str) -> String {
        let code = match line.find("//") {
            Some(i) => &line[..i],
            None => line,
        };
        code.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    let transcribed: Vec<String> = this_file.lines().map(normalize).collect();

    let mut in_rust_block = false;
    let mut blocks = 0usize;
    let mut checked = 0usize;

    for line in readme.lines() {
        if line.trim_start().starts_with("```") {
            if in_rust_block {
                in_rust_block = false;
            } else if line.trim() == "```rust" {
                in_rust_block = true;
                blocks += 1;
            }
            continue;
        }
        if !in_rust_block {
            continue;
        }
        let want = normalize(line);
        if want.is_empty() {
            continue; // blank line, or a line that is only a comment
        }
        assert!(
            transcribed.contains(&want),
            "README.md has a line of Rust that this file does not transcribe, so it is \
             compiled by nothing:\n    {}\nAdd it to `readme_examples_compile_and_run`.",
            line.trim()
        );
        checked += 1;
    }

    assert_eq!(
        blocks, 3,
        "README gained or lost a ```rust block; transcribe it here too"
    );
    assert!(
        checked > 10,
        "extraction found suspiciously little code: {checked} lines"
    );
}
