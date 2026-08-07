//! The official Argon2 test vectors, transcribed from
//! `phc-winner-argon2/src/test.c`.
//!
//! Every `hashtest(...)` call in `test.c` is one entry of [`VECTORS`], generated
//! by parsing `test.c` rather than retyped, and every entry gets its own
//! `#[test]` so the suite runs them in parallel. Each `#[test]` name encodes the
//! parameters, and each table entry carries the `test.c` line it came from.
//!
//! For each vector this reproduces exactly what `hashtest()` asserts:
//!
//! 1. `argon2_hash()` succeeds and the raw tag matches the hex reference;
//! 2. the encoded PHC string matches the reference — but **only** when
//!    `version == ARGON2_VERSION_NUMBER`, because that is the condition
//!    `test.c:55` puts on the comparison. The `v=0x10` references predate the
//!    `$v=` field and have no `$v=`, while `encode_string()` in the C always
//!    emits one, so for those the C compares nothing and neither does this;
//! 3. `argon2_verify()` accepts the string this crate just produced;
//! 4. `argon2_verify()` accepts the reference string from `test.c`.
//!
//! Plus one assertion the C gets for free from `argon2_verify` but is worth
//! pinning separately here: [`Argon2::verify`] against the raw tag.
//!
//! # What is skipped
//!
//! The two `#ifdef TEST_LARGE_RAM` vectors (`m = 1 << 20`, i.e. 1 GiB) are
//! `#[ignore]`d, exactly like the C build skips them unless `TEST_LARGE_RAM` is
//! defined. Run them with:
//!
//! ```text
//! cargo test --release --test vectors -- --ignored
//! ```
//!
//! Nothing else is skipped: 24 of the 26 vectors run by default, alongside all
//! eight malformed-encoding cases and the three common error-state cases.
//!
//! Three things here are not in `test.c`. Spec item (12) — the thread count is
//! a pure performance knob, only `lanes` changes the tag — is asserted for
//! `p = 2` and `p = 4` across all three algorithms and both versions in
//! [`threads_do_not_change_the_tag`]. [`password_named_api_reproduces_a_vector`]
//! runs one vector through the `*_password` spellings of the three entry points.
//! And [`every_official_vector_survives_a_reused_arena`] replays every default
//! vector through a single `Argon2::hasher()` in descending `m_cost` order, so
//! each published tag is also reproduced on an arena that a *different*
//! published vector just filled — the one thing the per-vector `#[test]`s
//! cannot check, because each of them gets a fresh arena by construction.

use argon2_rust::{Algorithm, Argon2, Error, Params, Version};

/// `test.c`'s `#define OUT_LEN 32`.
const OUT_LEN: usize = 32;

/// One `hashtest(version, t, m, p, pwd, salt, hexref, mcfref, type)` call.
struct Vector {
    version: Version,
    algorithm: Algorithm,
    t_cost: u32,
    /// `test.c` passes `1 << m` as the memory cost.
    m_cost_log2: u32,
    lanes: u32,
    pwd: &'static [u8],
    salt: &'static [u8],
    tag_hex: &'static str,
    encoded: &'static str,
    /// Guarded by `#ifdef TEST_LARGE_RAM` in `test.c`.
    large_ram: bool,
}

const VECTORS: &[Vector] = &[
    Vector {
        // src/test.c:77
        version: Version::V0x10,
        algorithm: Algorithm::Argon2i,
        t_cost: 2,
        m_cost_log2: 16,
        lanes: 1,
        pwd: b"password",
        salt: b"somesalt",
        tag_hex: "f6c4db4a54e2a370627aff3db6176b94a2a209a62c8e36152711802f7b30c694",
        encoded: "$argon2i$m=65536,t=2,p=1$c29tZXNhbHQ$9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ",
        large_ram: false,
    },
    Vector {
        // src/test.c:82
        version: Version::V0x10,
        algorithm: Algorithm::Argon2i,
        t_cost: 2,
        m_cost_log2: 20,
        lanes: 1,
        pwd: b"password",
        salt: b"somesalt",
        tag_hex: "9690ec55d28d3ed32562f2e73ea62b02b018757643a2ae6e79528459de8106e9",
        encoded: "$argon2i$m=1048576,t=2,p=1$c29tZXNhbHQ$lpDsVdKNPtMlYvLnPqYrArAYdXZDoq5ueVKEWd6BBuk",
        large_ram: true,
    },
    Vector {
        // src/test.c:87
        version: Version::V0x10,
        algorithm: Algorithm::Argon2i,
        t_cost: 2,
        m_cost_log2: 18,
        lanes: 1,
        pwd: b"password",
        salt: b"somesalt",
        tag_hex: "3e689aaa3d28a77cf2bc72a51ac53166761751182f1ee292e3f677a7da4c2467",
        encoded: "$argon2i$m=262144,t=2,p=1$c29tZXNhbHQ$Pmiaqj0op3zyvHKlGsUxZnYXURgvHuKS4/Z3p9pMJGc",
        large_ram: false,
    },
    Vector {
        // src/test.c:91
        version: Version::V0x10,
        algorithm: Algorithm::Argon2i,
        t_cost: 2,
        m_cost_log2: 8,
        lanes: 1,
        pwd: b"password",
        salt: b"somesalt",
        tag_hex: "fd4dd83d762c49bdeaf57c47bdcd0c2f1babf863fdeb490df63ede9975fccf06",
        encoded: "$argon2i$m=256,t=2,p=1$c29tZXNhbHQ$/U3YPXYsSb3q9XxHvc0MLxur+GP960kN9j7emXX8zwY",
        large_ram: false,
    },
    Vector {
        // src/test.c:95
        version: Version::V0x10,
        algorithm: Algorithm::Argon2i,
        t_cost: 2,
        m_cost_log2: 8,
        lanes: 2,
        pwd: b"password",
        salt: b"somesalt",
        tag_hex: "b6c11560a6a9d61eac706b79a2f97d68b4463aa3ad87e00c07e2b01e90c564fb",
        encoded: "$argon2i$m=256,t=2,p=2$c29tZXNhbHQ$tsEVYKap1h6scGt5ovl9aLRGOqOth+AMB+KwHpDFZPs",
        large_ram: false,
    },
    Vector {
        // src/test.c:99
        version: Version::V0x10,
        algorithm: Algorithm::Argon2i,
        t_cost: 1,
        m_cost_log2: 16,
        lanes: 1,
        pwd: b"password",
        salt: b"somesalt",
        tag_hex: "81630552b8f3b1f48cdb1992c4c678643d490b2b5eb4ff6c4b3438b5621724b2",
        encoded: "$argon2i$m=65536,t=1,p=1$c29tZXNhbHQ$gWMFUrjzsfSM2xmSxMZ4ZD1JCytetP9sSzQ4tWIXJLI",
        large_ram: false,
    },
    Vector {
        // src/test.c:103
        version: Version::V0x10,
        algorithm: Algorithm::Argon2i,
        t_cost: 4,
        m_cost_log2: 16,
        lanes: 1,
        pwd: b"password",
        salt: b"somesalt",
        tag_hex: "f212f01615e6eb5d74734dc3ef40ade2d51d052468d8c69440a3a1f2c1c2847b",
        encoded: "$argon2i$m=65536,t=4,p=1$c29tZXNhbHQ$8hLwFhXm6110c03D70Ct4tUdBSRo2MaUQKOh8sHChHs",
        large_ram: false,
    },
    Vector {
        // src/test.c:107
        version: Version::V0x10,
        algorithm: Algorithm::Argon2i,
        t_cost: 2,
        m_cost_log2: 16,
        lanes: 1,
        pwd: b"differentpassword",
        salt: b"somesalt",
        tag_hex: "e9c902074b6754531a3a0be519e5baf404b30ce69b3f01ac3bf21229960109a3",
        encoded: "$argon2i$m=65536,t=2,p=1$c29tZXNhbHQ$6ckCB0tnVFMaOgvlGeW69ASzDOabPwGsO/ISKZYBCaM",
        large_ram: false,
    },
    Vector {
        // src/test.c:111
        version: Version::V0x10,
        algorithm: Algorithm::Argon2i,
        t_cost: 2,
        m_cost_log2: 16,
        lanes: 1,
        pwd: b"password",
        salt: b"diffsalt",
        tag_hex: "79a103b90fe8aef8570cb31fc8b22259778916f8336b7bdac3892569d4f1c497",
        encoded: "$argon2i$m=65536,t=2,p=1$ZGlmZnNhbHQ$eaEDuQ/orvhXDLMfyLIiWXeJFvgza3vaw4kladTxxJc",
        large_ram: false,
    },
    Vector {
        // src/test.c:156
        version: Version::V0x13,
        algorithm: Algorithm::Argon2i,
        t_cost: 2,
        m_cost_log2: 16,
        lanes: 1,
        pwd: b"password",
        salt: b"somesalt",
        tag_hex: "c1628832147d9720c5bd1cfd61367078729f6dfb6f8fea9ff98158e0d7816ed0",
        encoded: "$argon2i$v=19$m=65536,t=2,p=1$c29tZXNhbHQ$wWKIMhR9lyDFvRz9YTZweHKfbftvj+qf+YFY4NeBbtA",
        large_ram: false,
    },
    Vector {
        // src/test.c:161
        version: Version::V0x13,
        algorithm: Algorithm::Argon2i,
        t_cost: 2,
        m_cost_log2: 20,
        lanes: 1,
        pwd: b"password",
        salt: b"somesalt",
        tag_hex: "d1587aca0922c3b5d6a83edab31bee3c4ebaef342ed6127a55d19b2351ad1f41",
        encoded: "$argon2i$v=19$m=1048576,t=2,p=1$c29tZXNhbHQ$0Vh6ygkiw7XWqD7asxvuPE667zQu1hJ6VdGbI1GtH0E",
        large_ram: true,
    },
    Vector {
        // src/test.c:166
        version: Version::V0x13,
        algorithm: Algorithm::Argon2i,
        t_cost: 2,
        m_cost_log2: 18,
        lanes: 1,
        pwd: b"password",
        salt: b"somesalt",
        tag_hex: "296dbae80b807cdceaad44ae741b506f14db0959267b183b118f9b24229bc7cb",
        encoded: "$argon2i$v=19$m=262144,t=2,p=1$c29tZXNhbHQ$KW266AuAfNzqrUSudBtQbxTbCVkmexg7EY+bJCKbx8s",
        large_ram: false,
    },
    Vector {
        // src/test.c:170
        version: Version::V0x13,
        algorithm: Algorithm::Argon2i,
        t_cost: 2,
        m_cost_log2: 8,
        lanes: 1,
        pwd: b"password",
        salt: b"somesalt",
        tag_hex: "89e9029f4637b295beb027056a7336c414fadd43f6b208645281cb214a56452f",
        encoded: "$argon2i$v=19$m=256,t=2,p=1$c29tZXNhbHQ$iekCn0Y3spW+sCcFanM2xBT63UP2sghkUoHLIUpWRS8",
        large_ram: false,
    },
    Vector {
        // src/test.c:174
        version: Version::V0x13,
        algorithm: Algorithm::Argon2i,
        t_cost: 2,
        m_cost_log2: 8,
        lanes: 2,
        pwd: b"password",
        salt: b"somesalt",
        tag_hex: "4ff5ce2769a1d7f4c8a491df09d41a9fbe90e5eb02155a13e4c01e20cd4eab61",
        encoded: "$argon2i$v=19$m=256,t=2,p=2$c29tZXNhbHQ$T/XOJ2mh1/TIpJHfCdQan76Q5esCFVoT5MAeIM1Oq2E",
        large_ram: false,
    },
    Vector {
        // src/test.c:178
        version: Version::V0x13,
        algorithm: Algorithm::Argon2i,
        t_cost: 1,
        m_cost_log2: 16,
        lanes: 1,
        pwd: b"password",
        salt: b"somesalt",
        tag_hex: "d168075c4d985e13ebeae560cf8b94c3b5d8a16c51916b6f4ac2da3ac11bbecf",
        encoded: "$argon2i$v=19$m=65536,t=1,p=1$c29tZXNhbHQ$0WgHXE2YXhPr6uVgz4uUw7XYoWxRkWtvSsLaOsEbvs8",
        large_ram: false,
    },
    Vector {
        // src/test.c:182
        version: Version::V0x13,
        algorithm: Algorithm::Argon2i,
        t_cost: 4,
        m_cost_log2: 16,
        lanes: 1,
        pwd: b"password",
        salt: b"somesalt",
        tag_hex: "aaa953d58af3706ce3df1aefd4a64a84e31d7f54175231f1285259f88174ce5b",
        encoded: "$argon2i$v=19$m=65536,t=4,p=1$c29tZXNhbHQ$qqlT1YrzcGzj3xrv1KZKhOMdf1QXUjHxKFJZ+IF0zls",
        large_ram: false,
    },
    Vector {
        // src/test.c:186
        version: Version::V0x13,
        algorithm: Algorithm::Argon2i,
        t_cost: 2,
        m_cost_log2: 16,
        lanes: 1,
        pwd: b"differentpassword",
        salt: b"somesalt",
        tag_hex: "14ae8da01afea8700c2358dcef7c5358d9021282bd88663a4562f59fb74d22ee",
        encoded: "$argon2i$v=19$m=65536,t=2,p=1$c29tZXNhbHQ$FK6NoBr+qHAMI1jc73xTWNkCEoK9iGY6RWL1n7dNIu4",
        large_ram: false,
    },
    Vector {
        // src/test.c:190
        version: Version::V0x13,
        algorithm: Algorithm::Argon2i,
        t_cost: 2,
        m_cost_log2: 16,
        lanes: 1,
        pwd: b"password",
        salt: b"diffsalt",
        tag_hex: "b0357cccfbef91f3860b0dba447b2348cbefecadaf990abfe9cc40726c521271",
        encoded: "$argon2i$v=19$m=65536,t=2,p=1$ZGlmZnNhbHQ$sDV8zPvvkfOGCw26RHsjSMvv7K2vmQq/6cxAcmxSEnE",
        large_ram: false,
    },
    Vector {
        // src/test.c:233
        version: Version::V0x13,
        algorithm: Algorithm::Argon2id,
        t_cost: 2,
        m_cost_log2: 16,
        lanes: 1,
        pwd: b"password",
        salt: b"somesalt",
        tag_hex: "09316115d5cf24ed5a15a31a3ba326e5cf32edc24702987c02b6566f61913cf7",
        encoded: "$argon2id$v=19$m=65536,t=2,p=1$c29tZXNhbHQ$CTFhFdXPJO1aFaMaO6Mm5c8y7cJHAph8ArZWb2GRPPc",
        large_ram: false,
    },
    Vector {
        // src/test.c:237
        version: Version::V0x13,
        algorithm: Algorithm::Argon2id,
        t_cost: 2,
        m_cost_log2: 18,
        lanes: 1,
        pwd: b"password",
        salt: b"somesalt",
        tag_hex: "78fe1ec91fb3aa5657d72e710854e4c3d9b9198c742f9616c2f085bed95b2e8c",
        encoded: "$argon2id$v=19$m=262144,t=2,p=1$c29tZXNhbHQ$eP4eyR+zqlZX1y5xCFTkw9m5GYx0L5YWwvCFvtlbLow",
        large_ram: false,
    },
    Vector {
        // src/test.c:241
        version: Version::V0x13,
        algorithm: Algorithm::Argon2id,
        t_cost: 2,
        m_cost_log2: 8,
        lanes: 1,
        pwd: b"password",
        salt: b"somesalt",
        tag_hex: "9dfeb910e80bad0311fee20f9c0e2b12c17987b4cac90c2ef54d5b3021c68bfe",
        encoded: "$argon2id$v=19$m=256,t=2,p=1$c29tZXNhbHQ$nf65EOgLrQMR/uIPnA4rEsF5h7TKyQwu9U1bMCHGi/4",
        large_ram: false,
    },
    Vector {
        // src/test.c:245
        version: Version::V0x13,
        algorithm: Algorithm::Argon2id,
        t_cost: 2,
        m_cost_log2: 8,
        lanes: 2,
        pwd: b"password",
        salt: b"somesalt",
        tag_hex: "6d093c501fd5999645e0ea3bf620d7b8be7fd2db59c20d9fff9539da2bf57037",
        encoded: "$argon2id$v=19$m=256,t=2,p=2$c29tZXNhbHQ$bQk8UB/VmZZF4Oo79iDXuL5/0ttZwg2f/5U52iv1cDc",
        large_ram: false,
    },
    Vector {
        // src/test.c:249
        version: Version::V0x13,
        algorithm: Algorithm::Argon2id,
        t_cost: 1,
        m_cost_log2: 16,
        lanes: 1,
        pwd: b"password",
        salt: b"somesalt",
        tag_hex: "f6a5adc1ba723dddef9b5ac1d464e180fcd9dffc9d1cbf76cca2fed795d9ca98",
        encoded: "$argon2id$v=19$m=65536,t=1,p=1$c29tZXNhbHQ$9qWtwbpyPd3vm1rB1GThgPzZ3/ydHL92zKL+15XZypg",
        large_ram: false,
    },
    Vector {
        // src/test.c:253
        version: Version::V0x13,
        algorithm: Algorithm::Argon2id,
        t_cost: 4,
        m_cost_log2: 16,
        lanes: 1,
        pwd: b"password",
        salt: b"somesalt",
        tag_hex: "9025d48e68ef7395cca9079da4c4ec3affb3c8911fe4f86d1a2520856f63172c",
        encoded: "$argon2id$v=19$m=65536,t=4,p=1$c29tZXNhbHQ$kCXUjmjvc5XMqQedpMTsOv+zyJEf5PhtGiUghW9jFyw",
        large_ram: false,
    },
    Vector {
        // src/test.c:257
        version: Version::V0x13,
        algorithm: Algorithm::Argon2id,
        t_cost: 2,
        m_cost_log2: 16,
        lanes: 1,
        pwd: b"differentpassword",
        salt: b"somesalt",
        tag_hex: "0b84d652cf6b0c4beaef0dfe278ba6a80df6696281d7e0d2891b817d8c458fde",
        encoded: "$argon2id$v=19$m=65536,t=2,p=1$c29tZXNhbHQ$C4TWUs9rDEvq7w3+J4umqA32aWKB1+DSiRuBfYxFj94",
        large_ram: false,
    },
    Vector {
        // src/test.c:261
        version: Version::V0x13,
        algorithm: Algorithm::Argon2id,
        t_cost: 2,
        m_cost_log2: 16,
        lanes: 1,
        pwd: b"password",
        salt: b"diffsalt",
        tag_hex: "bdf32b05ccc42eb15d58fd19b1f856b113da1e9a5874fdcc544308565aa8141c",
        encoded: "$argon2id$v=19$m=65536,t=2,p=1$ZGlmZnNhbHQ$vfMrBczELrFdWP0ZsfhWsRPaHppYdP3MVEMIVlqoFBw",
        large_ram: false,
    },
];

fn to_hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        s.push_str(&format!("{byte:02x}"));
    }
    s
}

/// Everything `hashtest()` in `test.c` asserts, for one vector.
fn check(v: &Vector) {
    let params = Params::new(1 << v.m_cost_log2, v.t_cost, v.lanes, OUT_LEN)
        .expect("test.c parameters are valid");
    let argon2 = Argon2::new(v.algorithm, v.version, params);

    // ret = argon2_hash(...); assert(ret == ARGON2_OK);
    // assert(memcmp(hex_out, hexref, OUT_LEN * 2) == 0);
    let mut out = [0u8; OUT_LEN];
    argon2
        .hash_into(v.pwd, v.salt, &mut out)
        .expect("argon2_hash returned ARGON2_OK");
    assert_eq!(to_hex(&out), v.tag_hex, "raw tag");

    // if (ARGON2_VERSION_NUMBER == version)
    //     assert(memcmp(encoded, mcfref, strlen(mcfref)) == 0);
    let encoded = argon2.hash_encoded(v.pwd, v.salt).expect("encode");
    if v.version == Version::V0x13 {
        assert_eq!(encoded, v.encoded, "encoded string");
    } else {
        // The v=0x10 references have no `$v=` field; the C's `encode_string`
        // always writes one, so the two differ by exactly that field.
        assert_eq!(
            encoded,
            v.encoded.replacen(
                &format!("${}$", v.algorithm.as_str()),
                &format!("${}$v=16$", v.algorithm.as_str()),
                1
            ),
            "encoded string (v=0x10, `$v=16` added)"
        );
    }

    // ret = argon2_verify(encoded, pwd, strlen(pwd), type);
    assert_eq!(
        Argon2::verify_encoded(&encoded, v.pwd, v.algorithm),
        Ok(()),
        "verify our own encoding"
    );

    // ret = argon2_verify(mcfref, pwd, strlen(pwd), type);
    assert_eq!(
        Argon2::verify_encoded(v.encoded, v.pwd, v.algorithm),
        Ok(()),
        "verify the reference encoding"
    );

    // Not in test.c, but the raw-tag counterpart of the two lines above.
    assert_eq!(argon2.verify(v.pwd, v.salt, &out), Ok(()), "verify raw tag");
}

macro_rules! vector_tests {
    ($($name:ident => $index:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let v = &VECTORS[$index];
                assert!(!v.large_ram, "large-RAM vectors belong in `large_ram_vectors`");
                check(v);
            }
        )*
    };
}

macro_rules! ignored_vector_tests {
    ($($name:ident => $index:expr,)*) => {
        $(
            #[test]
            #[ignore = "TEST_LARGE_RAM: needs 1 GiB; the C build skips it too"]
            fn $name() {
                let v = &VECTORS[$index];
                assert!(v.large_ram);
                check(v);
            }
        )*
    };
}

vector_tests! {
    argon2i_v0x10_t2_m16_p1_password_somesalt => 0,
    argon2i_v0x10_t2_m18_p1_password_somesalt => 2,
    argon2i_v0x10_t2_m8_p1_password_somesalt => 3,
    argon2i_v0x10_t2_m8_p2_password_somesalt => 4,
    argon2i_v0x10_t1_m16_p1_password_somesalt => 5,
    argon2i_v0x10_t4_m16_p1_password_somesalt => 6,
    argon2i_v0x10_t2_m16_p1_differentpassword_somesalt => 7,
    argon2i_v0x10_t2_m16_p1_password_diffsalt => 8,
    argon2i_v0x13_t2_m16_p1_password_somesalt => 9,
    argon2i_v0x13_t2_m18_p1_password_somesalt => 11,
    argon2i_v0x13_t2_m8_p1_password_somesalt => 12,
    argon2i_v0x13_t2_m8_p2_password_somesalt => 13,
    argon2i_v0x13_t1_m16_p1_password_somesalt => 14,
    argon2i_v0x13_t4_m16_p1_password_somesalt => 15,
    argon2i_v0x13_t2_m16_p1_differentpassword_somesalt => 16,
    argon2i_v0x13_t2_m16_p1_password_diffsalt => 17,
    argon2id_v0x13_t2_m16_p1_password_somesalt => 18,
    argon2id_v0x13_t2_m18_p1_password_somesalt => 19,
    argon2id_v0x13_t2_m8_p1_password_somesalt => 20,
    argon2id_v0x13_t2_m8_p2_password_somesalt => 21,
    argon2id_v0x13_t1_m16_p1_password_somesalt => 22,
    argon2id_v0x13_t4_m16_p1_password_somesalt => 23,
    argon2id_v0x13_t2_m16_p1_differentpassword_somesalt => 24,
    argon2id_v0x13_t2_m16_p1_password_diffsalt => 25,
}

ignored_vector_tests! {
    argon2i_v0x10_t2_m20_p1_password_somesalt => 1,
    argon2i_v0x13_t2_m20_p1_password_somesalt => 10,
}

// ---------------------------------------------------------------------------
// Coverage: every `hashtest()` call in test.c has a test above
// ---------------------------------------------------------------------------

#[test]
fn the_table_covers_every_hashtest_call_in_test_c() {
    // test.c has 26 `hashtest(...)` calls: nine Argon2i v=0x10, nine Argon2i
    // v=0x13 and eight Argon2id v=0x13. Two of the eighteen Argon2i ones are
    // `#ifdef TEST_LARGE_RAM`.
    assert_eq!(VECTORS.len(), 26);

    let count = |algorithm: Algorithm, version: Version| {
        VECTORS
            .iter()
            .filter(|v| v.algorithm == algorithm && v.version == version)
            .count()
    };
    assert_eq!(count(Algorithm::Argon2i, Version::V0x10), 9);
    assert_eq!(count(Algorithm::Argon2i, Version::V0x13), 9);
    assert_eq!(count(Algorithm::Argon2id, Version::V0x13), 8);
    assert_eq!(count(Algorithm::Argon2d, Version::V0x10), 0);
    assert_eq!(count(Algorithm::Argon2d, Version::V0x13), 0);

    assert_eq!(VECTORS.iter().filter(|v| v.large_ram).count(), 2);
    // Every tag is 32 hex-encoded bytes, and every encoding names its algorithm.
    for v in VECTORS {
        assert_eq!(v.tag_hex.len(), OUT_LEN * 2, "{}", v.encoded);
        assert!(v.tag_hex.bytes().all(|b| b.is_ascii_hexdigit()));
        assert!(
            v.encoded
                .starts_with(&format!("${}$", v.algorithm.as_str())),
            "{}",
            v.encoded
        );
    }
}

// ---------------------------------------------------------------------------
// Error state tests (test.c:118-148 and test.c:198-228)
// ---------------------------------------------------------------------------

/// `argon2_verify(encoded, "password", 8, Argon2_i)`.
fn verify_i(encoded: &str) -> Result<(), Error> {
    Argon2::verify_encoded(encoded, b"password", Algorithm::Argon2i)
}

#[test]
fn v0x10_invalid_encoding_missing_dollar_before_salt() {
    // test.c:119. The `$` between `p=1` and the salt is missing.
    assert_eq!(
        verify_i(
            "$argon2i$m=65536,t=2,p=1c29tZXNhbHQ\
             $9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ"
        ),
        Err(Error::DecodingFail)
    );
}

#[test]
fn v0x10_invalid_encoding_missing_dollar_before_tag() {
    // test.c:126. The `$` between the salt and the tag is missing.
    assert_eq!(
        verify_i(
            "$argon2i$m=65536,t=2,p=1$c29tZXNhbHQ\
             9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ"
        ),
        Err(Error::DecodingFail)
    );
}

#[test]
fn v0x10_invalid_encoding_salt_too_short() {
    // test.c:133. Empty salt. `decode_string` runs the whole `validate_inputs`
    // before the trailing-character check, so this is SALT_TOO_SHORT and not
    // DECODING_FAIL.
    assert_eq!(
        verify_i(
            "$argon2i$m=65536,t=2,p=1$\
             $9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ"
        ),
        Err(Error::SaltTooShort)
    );
}

#[test]
fn v0x10_mismatching_hash() {
    // test.c:140. The encoded password is "passwore".
    assert_eq!(
        verify_i(
            "$argon2i$m=65536,t=2,p=1$c29tZXNhbHQ\
             $b2G3seW+uPzerwQQC+/E1K50CLLO7YXy0JRcaTuswRo"
        ),
        Err(Error::VerifyMismatch)
    );
    // ... and it is the *right* hash for "passwore".
    assert_eq!(
        Argon2::verify_encoded(
            "$argon2i$m=65536,t=2,p=1$c29tZXNhbHQ\
             $b2G3seW+uPzerwQQC+/E1K50CLLO7YXy0JRcaTuswRo",
            b"passwore",
            Algorithm::Argon2i
        ),
        Ok(())
    );
}

#[test]
fn v0x13_invalid_encoding_missing_dollar_before_salt() {
    // test.c:199.
    assert_eq!(
        verify_i(
            "$argon2i$v=19$m=65536,t=2,p=1c29tZXNhbHQ\
             $wWKIMhR9lyDFvRz9YTZweHKfbftvj+qf+YFY4NeBbtA"
        ),
        Err(Error::DecodingFail)
    );
}

#[test]
fn v0x13_invalid_encoding_missing_dollar_before_tag() {
    // test.c:206.
    assert_eq!(
        verify_i(
            "$argon2i$v=19$m=65536,t=2,p=1$c29tZXNhbHQ\
             wWKIMhR9lyDFvRz9YTZweHKfbftvj+qf+YFY4NeBbtA"
        ),
        Err(Error::DecodingFail)
    );
}

#[test]
fn v0x13_invalid_encoding_salt_too_short() {
    // test.c:213.
    assert_eq!(
        verify_i(
            "$argon2i$v=19$m=65536,t=2,p=1$\
             $9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ"
        ),
        Err(Error::SaltTooShort)
    );
}

#[test]
fn v0x13_mismatching_hash() {
    // test.c:220. The encoded password is "passwore".
    assert_eq!(
        verify_i(
            "$argon2i$v=19$m=65536,t=2,p=1$c29tZXNhbHQ\
             $8iIuixkI73Js3G1uMbezQXD0b8LG4SXGsOwoQkdAQIM"
        ),
        Err(Error::VerifyMismatch)
    );
    assert_eq!(
        Argon2::verify_encoded(
            "$argon2i$v=19$m=65536,t=2,p=1$c29tZXNhbHQ\
             $8iIuixkI73Js3G1uMbezQXD0b8LG4SXGsOwoQkdAQIM",
            b"passwore",
            Algorithm::Argon2i
        ),
        Ok(())
    );
}

#[test]
fn error_message_matches_argon2_error_message() {
    // test.c:146 `assert(strcmp(msg, "Decoding failed") == 0);`
    assert_eq!(Error::DecodingFail.message(), "Decoding failed");
    assert_eq!(Error::DecodingFail.as_c_code(), -32);
}

// ---------------------------------------------------------------------------
// Common error state tests (test.c:266-286)
// ---------------------------------------------------------------------------

#[test]
fn common_error_fail_on_invalid_memory() {
    // test.c:271 `argon2_hash(2, 1, 1, "password", .., "diffsalt", ..)`
    //            `assert(ret == ARGON2_MEMORY_TOO_LITTLE);`
    //
    // `m_cost = 1` is below ARGON2_MIN_MEMORY (8), so this crate rejects it when
    // the `Params` are built rather than when the hash runs. Same code, earlier.
    assert_eq!(Params::new(1, 2, 1, OUT_LEN), Err(Error::MemoryTooLittle));
}

#[test]
fn common_error_fail_on_invalid_null_pointer() {
    // test.c:277 `argon2_hash(2, 1 << 12, 1, NULL, strlen("password"), ..)`
    //            `assert(ret == ARGON2_PWD_PTR_MISMATCH);`
    //
    // NOT REPRODUCIBLE, and deliberately so: the C's failure is "pwd == NULL but
    // pwdlen != 0", and a Rust `&[u8]` carries its own length, so the two can
    // never disagree. `Error::PwdPtrMismatch` exists only to keep
    // `Error::from_c_code` total.
    assert_eq!(Error::PwdPtrMismatch.as_c_code(), -18);

    // The reachable neighbours of that case both work: an empty password is
    // fine (ARGON2_MIN_PWD_LENGTH is 0), and it is not the same as any other.
    let params = Params::new(1 << 12, 2, 1, OUT_LEN).expect("params");
    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let empty = argon2
        .hash(b"", b"diffsalt")
        .expect("empty password hashes");
    let nonempty = argon2.hash(b"password", b"diffsalt").expect("hash");
    assert_ne!(empty, nonempty);
}

#[test]
fn common_error_fail_on_salt_too_short() {
    // test.c:283 `argon2_hash(2, 1 << 12, 1, "password", .., "s", 1, ..)`
    //            `assert(ret == ARGON2_SALT_TOO_SHORT);`
    let params = Params::new(1 << 12, 2, 1, OUT_LEN).expect("params");
    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let mut out = [0u8; OUT_LEN];
    assert_eq!(
        argon2.hash_into(b"password", b"s", &mut out),
        Err(Error::SaltTooShort)
    );
}

// ---------------------------------------------------------------------------
// Every official vector, through one reused arena
// ---------------------------------------------------------------------------

/// The published tags must still be the published tags when the arena is
/// **reused** rather than freshly allocated.
///
/// Each `#[test]` generated above gives its vector a brand-new `alloc_zeroed`
/// arena, which is the one case where arena reuse cannot be wrong. This runs
/// every default vector through a single `Argon2::hasher()`, so all but the
/// first start on memory holding a *different published vector's* derived
/// material — a different password, salt, algorithm, version and cost.
///
/// # Ordering, and why it is descending
///
/// The vectors are visited in **descending `m_cost`**, which does two things at
/// once. It makes the arena allocate exactly once, at the largest size, so
/// every later vector genuinely runs on a reused arena rather than on a fresh
/// one it happened to need to grow into. And it puts every later vector on a
/// *window* — `len()` blocks of a larger `capacity()` — which is the shape with
/// the most room to be wrong. Both are asserted below, not assumed.
///
/// # Cost
///
/// One hash per vector, not the five `check()` performs: the encoded forms, the
/// two `verify_encoded`s and the raw-tag verify are already pinned per vector
/// above, and repeating them here would triple the runtime of the slowest test
/// binary in the suite to re-test the encoder rather than the reuse layer. The
/// two `#[ignore]`d 1 GiB vectors are skipped for the same reason they are
/// skipped above — `test.c` guards them behind `TEST_LARGE_RAM`.
#[test]
fn every_official_vector_survives_a_reused_arena() {
    let mut order: Vec<&Vector> = VECTORS.iter().filter(|v| !v.large_ram).collect();
    // Descending m_cost; ties broken deterministically so the sequence does not
    // depend on the sort's stability guarantees changing.
    order.sort_by_key(|v| {
        (
            std::cmp::Reverse(v.m_cost_log2),
            v.algorithm.as_u32(),
            v.version.as_u32(),
            v.t_cost,
            v.lanes,
        )
    });
    assert!(
        order.len() >= 20,
        "expected the default vector set, got {}",
        order.len()
    );

    let largest = 1u32 << order[0].m_cost_log2;
    let mut hasher = Argon2::new(
        order[0].algorithm,
        order[0].version,
        Params::new(largest, order[0].t_cost, order[0].lanes, OUT_LEN).expect("params"),
    )
    .hasher();

    let mut reserved_after_first = 0usize;
    for (step, v) in order.iter().enumerate() {
        let params = Params::new(1 << v.m_cost_log2, v.t_cost, v.lanes, OUT_LEN)
            .expect("test.c parameters are valid");
        hasher.set_argon2(Argon2::new(v.algorithm, v.version, params));

        let mut out = [0u8; OUT_LEN];
        hasher
            .hash_into(v.pwd, v.salt, &mut out)
            .expect("argon2_hash returned ARGON2_OK");
        assert_eq!(
            to_hex(&out),
            v.tag_hex,
            "step {step}: {:?} v={:#x} m=2^{} t={} p={} produced the wrong tag on a \
             reused arena. The published vector is ground truth; fix the reuse \
             layer, do not loosen this test.",
            v.algorithm,
            v.version.as_u32(),
            v.m_cost_log2,
            v.t_cost,
            v.lanes,
        );

        if step == 0 {
            reserved_after_first = hasher.reserved_blocks();
            assert_eq!(
                reserved_after_first,
                params.memory_blocks() as usize,
                "the first vector should size the arena"
            );
        } else {
            // No growth after the first: every one of these ran on the arena
            // the first vector allocated, and on a window strictly inside it
            // whenever the m_cost is smaller.
            assert_eq!(
                hasher.reserved_blocks(),
                reserved_after_first,
                "step {step}: the arena was resized, so this vector did not run \
                 on a reused arena and the test proves nothing about it"
            );
        }
    }

    // The descent must be real, or "runs on a window of a larger arena" is an
    // empty claim.
    let smaller = order
        .iter()
        .filter(|v| (1u32 << v.m_cost_log2) < largest)
        .count();
    assert!(
        smaller >= 15,
        "only {smaller} vectors are smaller than the largest, so almost nothing \
         ran on a narrowed window"
    );
    println!(
        "official vectors: {} hashes on 1 arena of {reserved_after_first} blocks, \
         {smaller} of them on a narrowed window",
        order.len()
    );
}

// ---------------------------------------------------------------------------
// The `*_password` spellings of the same three entry points
// ---------------------------------------------------------------------------

#[test]
fn password_named_api_reproduces_a_vector() {
    // `Argon2::hash_password_into` / `hash_password` / `verify_password` are the
    // same functions as `hash_into` / `hash_encoded` / `verify_encoded`, so one
    // vector is enough to prove they are wired to the same computation. Pick the
    // cheapest Argon2id one (`m = 1 << 8`, `test.c:241`) so this stays fast.
    let v = VECTORS
        .iter()
        .find(|v| {
            v.algorithm == Algorithm::Argon2id
                && v.version == Version::V0x13
                && v.m_cost_log2 == 8
                && v.lanes == 1
        })
        .expect("test.c has an argon2id v=19 m=256 p=1 vector");

    let params = Params::new(1 << v.m_cost_log2, v.t_cost, v.lanes, OUT_LEN).expect("params");
    let argon2 = Argon2::new(v.algorithm, v.version, params);

    let mut out = [0u8; OUT_LEN];
    argon2
        .hash_password_into(v.pwd, v.salt, &mut out)
        .expect("hash_password_into");
    assert_eq!(to_hex(&out), v.tag_hex);

    let encoded = argon2.hash_password(v.pwd, v.salt).expect("hash_password");
    assert_eq!(encoded, v.encoded);

    assert_eq!(
        Argon2::verify_password(&encoded, v.pwd, v.algorithm),
        Ok(())
    );
    assert_eq!(
        Argon2::verify_password(v.encoded, v.pwd, v.algorithm),
        Ok(())
    );
    assert_eq!(
        Argon2::verify_password(v.encoded, b"wrong password", v.algorithm),
        Err(Error::VerifyMismatch)
    );
}

// ---------------------------------------------------------------------------
// Spec item (12): threads is a performance knob, lanes is not
// ---------------------------------------------------------------------------

#[test]
fn threads_do_not_change_the_tag() {
    // For p = 2 and p = 4, the tag must be identical whether the fill runs
    // single-threaded or with `threads == p`. `m = 1 << 12` with p = 4 gives
    // segment_length = 256, so every slice really is filled by four workers and
    // the cross-lane `index_alpha` references are exercised. t = 2 puts one pass
    // in the `pass > 0` branch as well.
    for lanes in [2u32, 4] {
        for algorithm in [Algorithm::Argon2d, Algorithm::Argon2i, Algorithm::Argon2id] {
            for version in [Version::V0x10, Version::V0x13] {
                let st = Params::new_with_threads(1 << 12, 2, lanes, 1, OUT_LEN).expect("st");
                let mt = Params::new_with_threads(1 << 12, 2, lanes, lanes, OUT_LEN).expect("mt");
                assert_eq!(st.effective_threads(), 1);
                assert_eq!(mt.effective_threads(), lanes);

                let single = Argon2::new(algorithm, version, st)
                    .hash(b"password", b"somesalt")
                    .expect("single-threaded");
                let multi = Argon2::new(algorithm, version, mt)
                    .hash(b"password", b"somesalt")
                    .expect("multi-threaded");

                assert_eq!(
                    to_hex(&single),
                    to_hex(&multi),
                    "{algorithm:?} {version:?} lanes={lanes}"
                );
            }
        }
    }
}

#[test]
fn threads_above_lanes_are_clamped_and_still_agree() {
    // `argon2_ctx` does `if (instance.threads > instance.lanes) instance.threads
    // = instance.lanes;`, so asking for more threads than lanes is legal and
    // changes nothing.
    let params = Params::new_with_threads(1 << 12, 2, 2, 64, OUT_LEN).expect("params");
    assert_eq!(params.effective_threads(), 2);
    let clamped = Argon2::new(Algorithm::Argon2id, Version::V0x13, params)
        .hash(b"password", b"somesalt")
        .expect("hash");

    let plain = Params::new(1 << 12, 2, 2, OUT_LEN).expect("params");
    let normal = Argon2::new(Algorithm::Argon2id, Version::V0x13, plain)
        .hash(b"password", b"somesalt")
        .expect("hash");

    assert_eq!(clamped, normal);
}

// ---------------------------------------------------------------------------
// verify_encoded_with_ad — the C's argon2_verify_ctx surface
// ---------------------------------------------------------------------------

const AD_PWD: &[u8] = b"password";
const AD_SALT: &[u8] = b"somesaltsomesalt";
/// The genkat.c secret and associated data, the reference's own values.
const AD_SECRET: &[u8] = &[0x03; 8];
const AD_AD: &[u8] = &[0x04; 12];

fn ad_fixture() -> (Argon2, String) {
    let params = Params::new(32, 3, 4, 32).expect("params");
    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let mut tag = [0u8; 32];
    argon2
        .hash_into_with_ad(AD_PWD, AD_SALT, AD_SECRET, AD_AD, &mut tag)
        .expect("hash with ad");
    let encoded = argon2_rust::__internal::encode_string_alloc(
        Algorithm::Argon2id,
        Version::V0x13,
        &params,
        AD_SALT,
        &tag,
    )
    .expect("encode");
    (argon2, encoded)
}

#[test]
fn verify_encoded_with_ad_accepts_the_right_inputs() {
    let (_, encoded) = ad_fixture();
    Argon2::verify_encoded_with_ad(&encoded, AD_PWD, AD_SECRET, AD_AD, Algorithm::Argon2id)
        .expect("must verify");
}

#[test]
fn verify_encoded_with_ad_rejects_any_changed_input() {
    let (_, encoded) = ad_fixture();
    for (what, pwd, secret, ad) in [
        ("wrong password", b"passwore".as_slice(), AD_SECRET, AD_AD),
        ("wrong secret", AD_PWD, [0x99; 8].as_slice(), AD_AD),
        ("wrong ad", AD_PWD, AD_SECRET, [0x99; 12].as_slice()),
        ("missing secret", AD_PWD, [].as_slice(), AD_AD),
        ("missing ad", AD_PWD, AD_SECRET, [].as_slice()),
    ] {
        assert_eq!(
            Argon2::verify_encoded_with_ad(&encoded, pwd, secret, ad, Algorithm::Argon2id),
            Err(Error::VerifyMismatch),
            "{what}"
        );
    }
}

#[test]
fn verify_encoded_with_ad_rejects_the_wrong_algorithm() {
    let (_, encoded) = ad_fixture();
    assert_eq!(
        Argon2::verify_encoded_with_ad(&encoded, AD_PWD, AD_SECRET, AD_AD, Algorithm::Argon2i),
        Err(Error::DecodingFail)
    );
}

#[test]
fn pooled_verify_encoded_with_ad_matches_the_one_shot_api() {
    let (argon2, encoded) = ad_fixture();
    let mut hasher = argon2.hasher();
    hasher
        .verify_encoded_with_ad(&encoded, AD_PWD, AD_SECRET, AD_AD, Algorithm::Argon2id)
        .expect("pooled must verify");
    assert_eq!(
        hasher.verify_encoded_with_ad(&encoded, AD_PWD, &[0x99; 8], AD_AD, Algorithm::Argon2id),
        Err(Error::VerifyMismatch)
    );
}

#[test]
fn encoded_len_is_the_c_buffer_size_including_nul() {
    let params = Params::new(32, 3, 4, 32).expect("params");
    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let encoded = argon2.hash_encoded(AD_PWD, AD_SALT).expect("encode");
    assert_eq!(
        encoded.len() + 1,
        argon2_rust::encoded_len(Algorithm::Argon2id, 3, 32, 4, AD_SALT.len() as u32, 32),
        "the C's argon2_encodedlen counts the NUL; a Rust String is one byte shorter"
    );
}

// ---------------------------------------------------------------------------
// Random-salt convenience API
// ---------------------------------------------------------------------------
//
// These live here rather than in `src/random.rs`'s unit tests because the wasm
// CI legs run named integration suites (`--test vectors`, ...) and never
// `--lib`. Without a call from one of these, `random_get` is dead code that
// wasm-ld drops, and the WASI arm would ship untested.

/// A cheap-but-real configuration. `m_cost` is deliberately tiny: this is
/// testing the salt source and the round trip, not the KDF.
#[cfg(feature = "std")]
fn rand_salt_argon2() -> Argon2 {
    let params = Params::new(32, 1, 1, 32).expect("params");
    Argon2::new(Algorithm::Argon2id, Version::V0x13, params)
}

#[cfg(feature = "std")]
#[test]
fn random_salt_round_trips_through_verify() {
    let argon2 = rand_salt_argon2();
    let encoded = argon2
        .hash_password_with_random_salt(b"password")
        .expect("the OS entropy source works");
    Argon2::verify_encoded(&encoded, b"password", Algorithm::Argon2id)
        .expect("a string we just produced must verify");
    assert_eq!(
        Argon2::verify_encoded(&encoded, b"wrong", Algorithm::Argon2id),
        Err(Error::VerifyMismatch)
    );
}

/// The whole point of the API: the salt is *fresh* per call. Same password,
/// same params, different string — which can only come from the salt.
#[cfg(feature = "std")]
#[test]
fn random_salt_differs_between_calls() {
    let argon2 = rand_salt_argon2();
    let a = argon2
        .hash_password_with_random_salt(b"password")
        .expect("entropy");
    let b = argon2
        .hash_password_with_random_salt(b"password")
        .expect("entropy");
    assert_ne!(a, b, "two hashes of one password shared a salt");
}

/// The decoded salt really is `RANDOM_SALT_LEN` bytes, not whatever length the
/// base64 happened to round to.
#[cfg(feature = "std")]
#[test]
fn random_salt_is_the_documented_length() {
    let argon2 = rand_salt_argon2();
    let encoded = argon2
        .hash_password_with_random_salt(b"password")
        .expect("entropy");
    let salt_b64 = encoded.split('$').nth(4).expect("PHC has a salt field");
    // Unpadded base64: n bytes -> ceil(4n/3) chars.
    let expect = argon2_rust::encoded_len(
        Algorithm::Argon2id,
        1,
        32,
        1,
        argon2_rust::RANDOM_SALT_LEN as u32,
        32,
    );
    assert_eq!(
        encoded.len() + 1,
        expect,
        "salt is not RANDOM_SALT_LEN bytes"
    );
    assert_eq!(salt_b64.len(), 22, "16 bytes is 22 unpadded base64 chars");
}

#[cfg(feature = "std")]
#[test]
fn pooled_random_salt_round_trips_and_stays_fresh() {
    let mut hasher = rand_salt_argon2().hasher();
    let a = hasher
        .hash_password_with_random_salt(b"password")
        .expect("entropy");
    let b = hasher
        .hash_password_with_random_salt(b"password")
        .expect("entropy");
    assert_ne!(a, b, "the pooled path reused a salt");
    hasher
        .verify_encoded(&a, b"password", Algorithm::Argon2id)
        .expect("pooled must verify what it produced");
}

// ---------------------------------------------------------------------------
// verify_encoded_bounded
// ---------------------------------------------------------------------------

/// `m=4294967295` is a 4 TiB arena. The assertion that matters is not the error
/// code — it is that this test *finishes*: the ceiling has to be applied while
/// the cost is still a number, before anything is allocated.
#[test]
fn bounded_verify_rejects_a_hostile_cost_without_allocating() {
    let hostile = "$argon2id$v=19$m=4294967295,t=1,p=1$c29tZXNhbHQ\
                   $CTFhFdXPJO1aFaMaO6Mm5c8y7cJHAph8ArZWb2GRPPc";
    let ceiling = Params::new(1 << 16, 8, 4, 32).expect("ceiling");
    assert_eq!(
        Argon2::verify_encoded_bounded(hostile, b"password", Algorithm::Argon2id, &ceiling),
        Err(Error::MemoryTooMuch)
    );
}

/// Each cost gets its own C error code, and they are checked m, t, p in order.
#[test]
fn bounded_verify_reports_the_c_code_for_each_cost() {
    let ceiling = Params::new(64, 2, 1, 32).expect("ceiling");
    let cases = [
        ("m=4096,t=1,p=1", Error::MemoryTooMuch),
        ("m=64,t=99,p=1", Error::TimeTooLarge),
        ("m=64,t=1,p=4", Error::LanesTooMany),
    ];
    for (costs, want) in cases {
        let encoded = format!(
            "$argon2id$v=19${costs}$c29tZXNhbHQ\
             $CTFhFdXPJO1aFaMaO6Mm5c8y7cJHAph8ArZWb2GRPPc"
        );
        assert_eq!(
            Argon2::verify_encoded_bounded(&encoded, b"password", Algorithm::Argon2id, &ceiling),
            Err(want),
            "costs {costs}"
        );
    }
}

/// A string inside the ceiling must behave exactly like `verify_encoded` —
/// the bound is a gate, not a different algorithm.
#[test]
fn bounded_verify_agrees_with_plain_verify_inside_the_ceiling() {
    let params = Params::new(32, 1, 1, 32).expect("params");
    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let encoded = argon2
        .hash_encoded(b"password", b"somesalt")
        .expect("encode");
    let ceiling = Params::new(1 << 16, 8, 4, 32).expect("ceiling");

    Argon2::verify_encoded_bounded(&encoded, b"password", Algorithm::Argon2id, &ceiling)
        .expect("inside the ceiling this is just verify_encoded");
    assert_eq!(
        Argon2::verify_encoded_bounded(&encoded, b"wrong", Algorithm::Argon2id, &ceiling),
        Err(Error::VerifyMismatch)
    );

    let mut hasher = argon2.hasher();
    hasher
        .verify_encoded_bounded(&encoded, b"password", Algorithm::Argon2id, &ceiling)
        .expect("pooled bounded verify");
    assert_eq!(
        hasher.verify_encoded_bounded(&encoded, b"password", Algorithm::Argon2id, &{
            Params::new(8, 1, 1, 32).expect("tiny ceiling")
        }),
        Err(Error::MemoryTooMuch),
        "the pooled path must apply the ceiling too"
    );
}

/// The keyed entry points get the ceiling too — a deployment using a secret is
/// if anything *more* likely to be the one parsing untrusted strings.
#[test]
fn bounded_verify_with_ad_applies_the_ceiling_and_still_verifies() {
    let (argon2, encoded) = ad_fixture();
    let generous = Params::new(1 << 16, 8, 4, 32).expect("ceiling");
    let stingy = Params::new(8, 1, 1, 32).expect("tiny ceiling");

    Argon2::verify_encoded_bounded_with_ad(
        &encoded,
        AD_PWD,
        AD_SECRET,
        AD_AD,
        Algorithm::Argon2id,
        &generous,
    )
    .expect("inside the ceiling this is verify_encoded_with_ad");

    assert_eq!(
        Argon2::verify_encoded_bounded_with_ad(
            &encoded,
            AD_PWD,
            AD_SECRET,
            AD_AD,
            Algorithm::Argon2id,
            &stingy,
        ),
        Err(Error::MemoryTooMuch)
    );

    // A wrong secret must still be a mismatch, not a ceiling error.
    assert_eq!(
        Argon2::verify_encoded_bounded_with_ad(
            &encoded,
            AD_PWD,
            &[0x99; 8],
            AD_AD,
            Algorithm::Argon2id,
            &generous,
        ),
        Err(Error::VerifyMismatch)
    );

    let mut hasher = argon2.hasher();
    hasher
        .verify_encoded_bounded_with_ad(
            &encoded,
            AD_PWD,
            AD_SECRET,
            AD_AD,
            Algorithm::Argon2id,
            &generous,
        )
        .expect("pooled bounded verify with ad");
    assert_eq!(
        hasher.verify_encoded_bounded_with_ad(
            &encoded,
            AD_PWD,
            AD_SECRET,
            AD_AD,
            Algorithm::Argon2id,
            &stingy,
        ),
        Err(Error::MemoryTooMuch),
        "the pooled keyed path must apply the ceiling too"
    );
}

/// The hostile-cost string must be rejected on the keyed path without
/// allocating either. As with the unkeyed case, finishing *is* the assertion.
#[test]
fn bounded_verify_with_ad_rejects_a_hostile_cost_without_allocating() {
    let hostile = "$argon2id$v=19$m=4294967295,t=1,p=1$c29tZXNhbHQ\
                   $CTFhFdXPJO1aFaMaO6Mm5c8y7cJHAph8ArZWb2GRPPc";
    let ceiling = Params::new(1 << 16, 8, 4, 32).expect("ceiling");
    assert_eq!(
        Argon2::verify_encoded_bounded_with_ad(
            hostile,
            AD_PWD,
            AD_SECRET,
            AD_AD,
            Algorithm::Argon2id,
            &ceiling,
        ),
        Err(Error::MemoryTooMuch)
    );
}

// ---------------------------------------------------------------------------
// Bounded verify: the ceiling must bound *allocation*, not just cost numbers
// ---------------------------------------------------------------------------
//
// Regression tests for a real defect: `check_ceiling` once ran only after
// `decode_string`, and only looked at m/t/p. A well-formed string with tiny
// costs and a 16 MiB Base64 tag therefore decoded (peak 36 MiB live, measured
// with an allocator spy), passed the ceiling, allocated a 12 MiB comparison
// buffer and ran a full Argon2 — under a ceiling whose `output_len` was 32.

/// The tag field is the lever: costs are all inside the ceiling, so nothing but
/// a size check can stop this. Finishing quickly *is* the assertion.
#[test]
fn bounded_verify_rejects_an_oversized_tag_before_decoding() {
    let huge_tag = "A".repeat(4 * 1024 * 1024);
    let encoded = format!("$argon2id$v=19$m=8,t=1,p=1$c29tZXNhbHQ${huge_tag}");
    let ceiling = Params::new(8, 1, 1, 32).expect("ceiling");
    assert_eq!(
        Argon2::verify_encoded_bounded(&encoded, b"pw", Algorithm::Argon2id, &ceiling),
        Err(Error::DecodingLengthFail),
        "an oversized tag must be refused on length, before the decoder allocates"
    );
    // Same for the keyed and pooled entry points.
    assert_eq!(
        Argon2::verify_encoded_bounded_with_ad(
            &encoded,
            b"pw",
            &[],
            &[],
            Algorithm::Argon2id,
            &ceiling
        ),
        Err(Error::DecodingLengthFail)
    );
    let mut hasher = Argon2::new(
        Algorithm::Argon2id,
        Version::V0x13,
        Params::new(8, 1, 1, 32).expect("params"),
    )
    .hasher();
    assert_eq!(
        hasher.verify_encoded_bounded(&encoded, b"pw", Algorithm::Argon2id, &ceiling),
        Err(Error::DecodingLengthFail)
    );
}

/// An oversized *salt* is the same lever through the other field.
#[test]
fn bounded_verify_rejects_an_oversized_salt_before_decoding() {
    let huge_salt = "A".repeat(4 * 1024 * 1024);
    let encoded = format!(
        "$argon2id$v=19$m=8,t=1,p=1${huge_salt}\
         $CTFhFdXPJO1aFaMaO6Mm5c8y7cJHAph8ArZWb2GRPPc"
    );
    let ceiling = Params::new(8, 1, 1, 32).expect("ceiling");
    assert_eq!(
        Argon2::verify_encoded_bounded(&encoded, b"pw", Algorithm::Argon2id, &ceiling),
        Err(Error::DecodingLengthFail)
    );
}

/// A tag that clears the length gate but still exceeds `ceiling.output_len()`
/// must be refused too — the ceiling is four numbers and all four count.
#[test]
fn bounded_verify_enforces_the_output_length_ceiling() {
    // 64-byte tag => 86 unpadded base64 chars. Short enough to pass the size
    // gate computed from a 1024-byte salt allowance, too long for the ceiling.
    let params = Params::new(32, 1, 1, 64).expect("params");
    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let encoded = argon2.hash_encoded(b"pw", b"somesalt").expect("encode");
    let ceiling = Params::new(1 << 16, 8, 4, 32).expect("ceiling");
    assert_eq!(
        Argon2::verify_encoded_bounded(&encoded, b"pw", Algorithm::Argon2id, &ceiling),
        Err(Error::OutputTooLong)
    );
    // Raise only the output ceiling and the very same string verifies.
    let ceiling = Params::new(1 << 16, 8, 4, 64).expect("ceiling");
    Argon2::verify_encoded_bounded(&encoded, b"pw", Algorithm::Argon2id, &ceiling)
        .expect("inside every part of the ceiling");
}

/// A salt at the documented maximum must still be accepted — the gate is a
/// bound, not a trap for legitimate producers.
#[test]
fn bounded_verify_accepts_a_salt_at_the_documented_maximum() {
    let salt = vec![0x5Au8; argon2_rust::BOUNDED_MAX_SALT_LEN as usize];
    let params = Params::new(32, 1, 1, 32).expect("params");
    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let encoded = argon2.hash_encoded(b"pw", &salt).expect("encode");
    let ceiling = Params::new(1 << 16, 8, 4, 32).expect("ceiling");
    Argon2::verify_encoded_bounded(&encoded, b"pw", Algorithm::Argon2id, &ceiling)
        .expect("a salt of exactly BOUNDED_MAX_SALT_LEN must verify");
}

/// The size gate is a budget computed by `encoded_len`, which always accounts
/// for a `$v=` field. The decoder also accepts the pre-1.3 form that has no
/// such field — shorter, so the budget still covers it. Pinned because a gate
/// that ever computed a *smaller* budget than a legitimate string would turn
/// this legacy shape into a spurious `DecodingLengthFail`.
#[test]
fn bounded_verify_accepts_the_legacy_form_with_no_version_field() {
    let params = Params::new(64, 2, 1, 32).expect("params");
    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x10, params);
    let encoded = argon2
        .hash_encoded(b"password", b"somesalt1234")
        .expect("encode");
    // The encoder always writes `$v=` (C parity); strip it to get the form the
    // decoder must still accept, which defaults to V0x10.
    let legacy = encoded.replace("$v=16", "");
    assert!(!legacy.contains("$v="), "stripping failed: {legacy}");
    Argon2::verify_encoded(&legacy, b"password", Algorithm::Argon2id)
        .expect("legacy form must verify unbounded");
    Argon2::verify_encoded_bounded(&legacy, b"password", Algorithm::Argon2id, &params)
        .expect("legacy form must verify bounded");
}

/// A worker budget below the string's `p` must not change which strings verify.
///
/// This is the property that makes the clamp in `decode_bounded` safe to apply
/// unconditionally: `threads` is a scheduling knob and only `lanes` feeds the
/// tag, so the same string must verify identically whether it runs on one
/// worker or on `p` of them. If that were ever false, the clamp would silently
/// reject legitimate credentials.
#[test]
fn bounded_verify_is_indifferent_to_the_worker_budget() {
    const LANES: u32 = 16;
    let params = Params::new(8 * LANES, 2, LANES, 32).expect("params");
    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let encoded = argon2
        .hash_encoded(b"password", b"somesalt")
        .expect("encode");

    for budget in [1, 2, 3, LANES - 1, LANES] {
        let ceiling = Params::new_with_threads(8 * LANES, 2, LANES, budget, 32).expect("ceiling");
        Argon2::verify_encoded_bounded(&encoded, b"password", Algorithm::Argon2id, &ceiling)
            .unwrap_or_else(|e| panic!("worker budget {budget} changed the verdict: {e:?}"));
        // and a wrong password stays wrong at every budget
        assert_eq!(
            Argon2::verify_encoded_bounded(&encoded, b"wrong", Algorithm::Argon2id, &ceiling),
            Err(Error::VerifyMismatch),
            "worker budget {budget}"
        );
    }
}

/// The pooled spelling clamps too — all four bounded entry points share the
/// same gate, and a `Hasher` is the one a server actually holds.
#[test]
fn pooled_bounded_verify_honours_the_worker_budget() {
    const LANES: u32 = 16;
    let params = Params::new(8 * LANES, 2, LANES, 32).expect("params");
    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let encoded = argon2
        .hash_encoded(b"password", b"somesalt")
        .expect("encode");

    let ceiling = Params::new_with_threads(8 * LANES, 2, LANES, 1, 32).expect("ceiling");
    let mut hasher = argon2.hasher();
    hasher
        .verify_encoded_bounded(&encoded, b"password", Algorithm::Argon2id, &ceiling)
        .expect("one worker must still verify");
    hasher
        .verify_encoded_bounded_with_ad(
            &encoded,
            b"password",
            &[],
            &[],
            Algorithm::Argon2id,
            &ceiling,
        )
        .expect("the _with_ad spelling shares the gate");
}
