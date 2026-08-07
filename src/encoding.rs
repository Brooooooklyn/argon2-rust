//! Base64 (standard alphabet, **no** `=` padding) and the PHC hash string.
//!
//! Format: `$argon2<T>[$v=<num>]$m=<num>,t=<num>,p=<num>$<b64 salt>$<b64 hash>`.
//!
//! Ported line by line from `phc-winner-argon2/src/encoding.c`
//! (`b64_byte_to_char`, `b64_char_to_byte`, `to_base64`, `from_base64`,
//! `decode_decimal`, `encode_string`, `decode_string`, `b64len`, `numlen`) plus
//! `argon2_encodedlen` from `src/argon2.c`.
//!
//! Details the C reference gets subtly right, and this port does too:
//!
//! * The `$v=` field is **optional**. When absent the version defaults to
//!   `0x10`, not `0x13`.
//! * After decoding, `threads = lanes`.
//! * `"argon2i"` is a prefix of `"argon2id"`. The C matches the type string and
//!   then relies on the *next* character failing to parse (`$` is expected), so
//!   `"$argon2id$..."` is rejected when the caller asked for `Argon2i`.
//!   Reproduced here rather than special-cased.
//! * Decoding stops at the first non-base64 character, and then **rejects** if
//!   `acc_len > 4` or any buffered low bits are non-zero.
//! * `b64_char_to_byte` returns `0xFF` for an invalid character; a computed
//!   value of 0 is only valid for `'A'`.
//! * `decode_decimal` rejects an empty run of digits and rejects non-minimal
//!   encodings (a leading `'0'` followed by more digits).
//! * [`decode_string`] finishes by running the full `validate_inputs()`, and
//!   only then requires the string to be fully consumed. The order matters: a
//!   string with both a zero-length salt *and* trailing junk reports
//!   [`Error::SaltTooShort`], exactly like the C.
//!
//! The character classification is branch-free, as in the C: the `EQ`/`GT`/
//! `GE`/`LT`/`LE` macros become the [`eq`]/[`gt`]/[`ge`]/[`lt`]/[`le`] helpers
//! below, which return `0x00` for false and `0xFF` for true without branching.
//! It matters because these run over salt and tag material.
//!
//! # Known divergences from the C reference
//!
//! * **Unrepresentable versions.** `validate_inputs()` never looks at
//!   `ctx->version` (`core.c:388`), so the C accepts `$v=99`. It then applies
//!   the `0x13` *fill rule* to anything that is not `0x10` (`ref.c:181`) — but
//!   the raw value is also hashed into H0 (`core.c:561`
//!   `store32(&value, context->version)`), so the tag is version-specific and
//!   `$v=99` is a self-consistent, verifiable C record rather than an alias for
//!   `$v=19`. Measured against `libargon2.a`: `version=99` yields tag
//!   `2d4d864d…` where `version=19` yields `c1628832…`. [`Version`] is a closed
//!   enum, so [`decode_string`] reports [`Error::DecodingFail`] instead. This
//!   is the one divergence reachable from a conforming ASCII PHC string, and
//!   only from a producer emitting a version other than 16 or 19. The check is
//!   deliberately the *last* thing it does, so every other error code still
//!   matches the C.
//! * **Embedded NULs.** The C stops at the first `'\0'`; a Rust `&str` has no
//!   terminator, so the whole slice must be consumed. This is strictly
//!   stricter: it only rejects strings the C would have accepted by truncating.
//! * **Bytes `>= 0x80` (a bug in the C).** `b64_char_to_byte(*src)` passes a
//!   `char`. Where `char` is signed — Apple, x86 Linux, MSVC — every byte
//!   `>= 0x80` sign-extends to a negative `int`, and `EQ`, which its own
//!   comment says is only valid "over values in the 0..255 range", then reports
//!   a spurious match against both `'+'` and `'/'`. The byte decodes as 63
//!   instead of being rejected. Measured on `aarch64-apple-darwin` against
//!   `libargon2.a`: `"$argon2i$v=19$m=65536,t=2,p=1$é9tZXNhbHQ$<tag>"` and
//!   `"…$//9tZXNhbHQ$<tag>"` both decode, to the same salt `ff ff 6d 65 …`.
//!   Where `char` is unsigned (aarch64 Linux) the same byte is rejected. This
//!   port always rejects, which matches the unsigned-`char` platforms and the
//!   evident intent, and is unreachable for any string the encoder produced.
//!   See `b64_char_to_byte_rejects_everything_else` below.
//! * **Password length.** `argon2_verify` fills in `ctx.pwd`/`ctx.pwdlen`
//!   before calling `decode_string`, so the C's final `validate_inputs()` also
//!   sees the password. [`decode_string`] validates with `pwd_len = 0`; the two
//!   differ only for a password longer than [`crate::params::MAX_PWD_LENGTH`]
//!   (4 GiB), which the verify path rejects a moment later anyway.

use alloc::string::String;
use alloc::vec::Vec;

use crate::error::Error;
use crate::params::{Algorithm, Params, Version, validate_inputs};

// ---------------------------------------------------------------------------
// Constant-time classification (encoding.c lines 74-78)
// ---------------------------------------------------------------------------
//
//   #define EQ(x, y) ((((0U - ((unsigned)(x) ^ (unsigned)(y))) >> 8) & 0xFF) ^ 0xFF)
//   #define GT(x, y) ((((unsigned)(y) - (unsigned)(x)) >> 8) & 0xFF)
//   #define GE(x, y) (GT(y, x) ^ 0xFF)
//   #define LT(x, y) GT(y, x)
//   #define LE(x, y) GE(y, x)
//
// Defined over 0..=255, returning 0x00 for false and 0xFF for true. The C
// evaluates them in `unsigned` with wrapping arithmetic, hence `wrapping_sub`.

/// `EQ(x, y)`: `0xFF` when `x == y`, else `0x00`.
#[inline]
const fn eq(x: u32, y: u32) -> u32 {
    ((0u32.wrapping_sub(x ^ y) >> 8) & 0xFF) ^ 0xFF
}

/// `GT(x, y)`: `0xFF` when `x > y`, else `0x00`.
#[inline]
const fn gt(x: u32, y: u32) -> u32 {
    (y.wrapping_sub(x) >> 8) & 0xFF
}

/// `GE(x, y)`: `0xFF` when `x >= y`, else `0x00`.
#[inline]
const fn ge(x: u32, y: u32) -> u32 {
    gt(y, x) ^ 0xFF
}

/// `LT(x, y)`: `0xFF` when `x < y`, else `0x00`.
#[inline]
const fn lt(x: u32, y: u32) -> u32 {
    gt(y, x)
}

/// `LE(x, y)`: `0xFF` when `x <= y`, else `0x00`.
#[inline]
const fn le(x: u32, y: u32) -> u32 {
    ge(y, x)
}

/// `b64_byte_to_char(x)`: map `0..64` to the standard base64 alphabet.
///
/// Branch-free, like the C. `x` is always masked to 6 bits by the caller.
#[inline]
const fn b64_byte_to_char(x: u32) -> u8 {
    // `'a' - 26` and `'0' - 52` are evaluated in the C as `int`; the second one
    // is negative (-4), which `wrapping_add` reproduces.
    let a_off = b'A' as u32;
    let lower_off = (b'a' as u32).wrapping_sub(26);
    let digit_off = (b'0' as u32).wrapping_sub(52);

    let c = (lt(x, 26) & x.wrapping_add(a_off))
        | (ge(x, 26) & lt(x, 52) & x.wrapping_add(lower_off))
        | (ge(x, 52) & lt(x, 62) & x.wrapping_add(digit_off))
        | (eq(x, 62) & b'+' as u32)
        | (eq(x, 63) & b'/' as u32);
    c as u8
}

/// `b64_char_to_byte(c)`: map a base64 character to its 6-bit value.
///
/// Returns `0xFF` for anything that is not a base64 character. Note the final
/// fixup: a computed value of 0 is only accepted for `'A'`, so every invalid
/// character (which also computes 0) is turned into `0xFF`.
///
/// `c` is always a `u8` widened to `u32`, i.e. `0..=255`, which is the range
/// the `EQ`/`GE`/`LE` macros are defined over. The C instead passes a `char`,
/// which sign-extends on most targets and makes bytes `>= 0x80` decode as 63
/// rather than being rejected — see the module docs.
#[inline]
const fn b64_char_to_byte(c: u32) -> u32 {
    let a_off = b'A' as u32;
    let lower_off = (b'a' as u32).wrapping_sub(26);
    let digit_off = (b'0' as u32).wrapping_sub(52);

    let x = (ge(c, b'A' as u32) & le(c, b'Z' as u32) & c.wrapping_sub(a_off))
        | (ge(c, b'a' as u32) & le(c, b'z' as u32) & c.wrapping_sub(lower_off))
        | (ge(c, b'0' as u32) & le(c, b'9' as u32) & c.wrapping_sub(digit_off))
        | (eq(c, b'+' as u32) & 62)
        | (eq(c, b'/' as u32) & 63);

    x | (eq(x, 0) & (eq(c, b'A' as u32) ^ 0xFF))
}

// ---------------------------------------------------------------------------
// Lengths
// ---------------------------------------------------------------------------

/// `b64len(len)`: length of the unpadded base64 encoding of `len` bytes.
///
/// ```text
/// olen = (len / 3) << 2;
/// switch (len % 3) { case 2: olen++; /* fall through */ case 1: olen += 2; }
/// ```
///
/// # 32-bit targets
///
/// On a target where `usize` is 32 bits, a `len` near `u32::MAX` makes
/// `(len / 3) << 2` discard its top bits, so the answer wraps. That is not a
/// defect to fix: `encoding.c:441` computes `((size_t)len / 3) << 2` and wraps
/// identically where `size_t` is 32 bits, and this function exists to match the
/// C. It cannot panic (`<<` discards bits rather than trapping, and the sums in
/// [`encoded_len`] stay inside `usize` even after a wrap), and it is not on the
/// allocation path — [`encode_string_alloc`] sizes its buffer with
/// [`encoded_len_usize`], which takes real slice lengths and cannot wrap.
#[must_use]
pub const fn b64_len(len: u32) -> usize {
    b64_len_usize(len as usize)
}

/// [`b64_len`] over a `usize`, for slices.
///
/// Cannot overflow **when fed a slice length**, which is every caller: a slice
/// is at most `isize::MAX` bytes, and `(isize::MAX / 3) * 4 < usize::MAX` on
/// both 32- and 64-bit targets. The `u32` entry point above has no such bound —
/// see its note.
#[inline]
const fn b64_len_usize(len: usize) -> usize {
    let mut olen = (len / 3) << 2;
    // The C's `case 2` falls through into `case 1`, so it adds 1 + 2.
    match len % 3 {
        2 => olen += 3,
        1 => olen += 2,
        _ => {}
    }
    olen
}

/// `numlen(num)`: number of decimal digits in `num` (1 for 0).
#[must_use]
pub const fn num_len(num: u32) -> usize {
    let mut len = 1usize;
    let mut n = num;
    while n >= 10 {
        len += 1;
        n /= 10;
    }
    len
}

/// `argon2_encodedlen(...)` from `src/argon2.c`.
///
/// ```text
/// strlen("$$v=$m=,t=,p=$$") + strlen(type) + numlen(t_cost) + numlen(m_cost)
///   + numlen(parallelism) + b64len(saltlen) + b64len(hashlen)
///   + numlen(ARGON2_VERSION_NUMBER) + 1
/// ```
///
/// The trailing `+ 1` is the C string's NUL terminator. It is kept so the value
/// matches the C byte for byte; a Rust [`String`] is one byte shorter. It is
/// also exactly the buffer size [`encode_string`] wants, which reserves the
/// same byte (see there).
///
/// Note the C uses `numlen(ARGON2_VERSION_NUMBER)` and not the version actually
/// being encoded. Both `0x10` (16) and `0x13` (19) are two digits, so it makes
/// no difference; it is kept verbatim. (In the C it *can* differ, because
/// `ctx->version` is a raw `uint32_t`: with `version = 0` the string is one
/// byte shorter than advertised, and a buffer of `argon2_encodedlen() - 1`
/// suffices. Measured on 1051 of 30000 fuzzed cases, all of them `version = 0`.
/// [`Version`] is a closed enum of two two-digit values, so the size is always
/// exact here — see `encode_needs_encoded_len_bytes_exactly`.)
///
/// # Argument order
///
/// `t_cost` comes before `m_cost` here, which is the reverse of
/// [`Params::new`]:
///
/// ```text
/// Params::new(m_cost, t_cost, lanes, output_len)
/// encoded_len(algorithm, t_cost, m_cost, lanes, salt_len, hash_len)
///                        ^^^^^^^^^^^^^^ reversed against Params::new
/// ```
///
/// The order is the C's, kept so a call can be transcribed position for
/// position: `argon2_encodedlen(t_cost, m_cost, parallelism, saltlen, hashlen,
/// type)`, declared at `argon2.h:429` and defined at `argon2.c:447`. One
/// argument did move. `type` went from last to first and became `algorithm`,
/// so it sits where every other entry point in this crate takes its
/// [`Algorithm`]. The other five kept their order among themselves;
/// `parallelism` is spelled `lanes` here, the name [`Params`] uses for it.
///
/// A caller who supplies the two costs the other way round gets no error back,
/// and the reason is arithmetic rather than luck. `t_cost` and `m_cost` reach
/// the result through one term each, `num_len(t_cost)` and `num_len(m_cost)`,
/// and the two terms are added: the sum of a pair of decimal digit counts does
/// not depend on which count came from which cost. Nothing else in the body
/// reads either value. The `m=` and `t=` fields of the string itself are
/// written by `encode_string` out of the [`Params`] it is handed, never out of
/// anything passed here, so a swap at this call site cannot reach the output
/// either. That makes the equality a property of this one formula and not a
/// rule about the crate: `encoded_len_is_symmetric_in_m_and_t` pins it at every
/// digit-count boundary, and will fail there first if the length ever stops
/// being a plain sum. Naming the arguments in the documented order keeps the
/// call site readable and keeps it correct under any later formula.
///
/// ```
/// use argon2_rust::{Algorithm, Params, encoded_len};
///
/// // Params::new takes m_cost first.
/// let params = Params::new(65536, 3, 1, 32).unwrap();
/// assert_eq!((params.m_cost(), params.t_cost()), (65536, 3));
///
/// // encoded_len takes t_cost first: the same two costs, the other way round.
/// let n = encoded_len(
///     Algorithm::Argon2id,
///     params.t_cost(),
///     params.m_cost(),
///     params.lanes(),
///     16, // salt_len
///     32, // hash_len
/// );
/// assert_eq!(n, 98);
/// ```
///
/// # Against a string the crate really produced
///
/// The value is a C buffer size, so it counts the NUL terminator described
/// above and a Rust [`String`] is one byte shorter. That is the whole of the
/// relationship, and it is exact rather than an upper bound - see
/// `encode_needs_encoded_len_bytes_exactly`:
///
/// ```
/// use argon2_rust::{Algorithm, Argon2, Params, Version, encoded_len};
///
/// let params = Params::new(64, 1, 1, 32)?;
/// let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
/// let encoded = argon2.hash_encoded(b"password", b"somesalt")?;
///
/// // `salt_len` is the salt's own 8 bytes, not the 11 its base64 occupies.
/// let n = encoded_len(Algorithm::Argon2id, 1, 64, 1, 8, 32);
/// assert_eq!(n, 84);
/// assert_eq!(encoded.len(), n - 1);
/// assert_eq!(
///     encoded,
///     "$argon2id$v=19$m=64,t=1,p=1$c29tZXNhbHQ$cpx6VEQbwTVZvcpxNIxOVUWZ5xnAipUmAe1cg2GMG70",
/// );
/// # Ok::<(), argon2_rust::Error>(())
/// ```
#[must_use]
pub fn encoded_len(
    algorithm: Algorithm,
    t_cost: u32,
    m_cost: u32,
    lanes: u32,
    salt_len: u32,
    hash_len: u32,
) -> usize {
    "$$v=$m=,t=,p=$$".len()
        + algorithm.as_str().len()
        + num_len(t_cost)
        + num_len(m_cost)
        + num_len(lanes)
        + b64_len(salt_len)
        + b64_len(hash_len)
        + num_len(Version::DEFAULT.as_u32())
        + 1
}

/// [`encoded_len`] computed from slice lengths, without the `u32` casts.
///
/// [`encode_string_alloc`] sizes its buffer with this so a salt longer than
/// `u32::MAX` cannot silently wrap into a too-small allocation.
fn encoded_len_usize(
    algorithm: Algorithm,
    t_cost: u32,
    m_cost: u32,
    lanes: u32,
    salt_len: usize,
    hash_len: usize,
) -> usize {
    "$$v=$m=,t=,p=$$".len()
        + algorithm.as_str().len()
        + num_len(t_cost)
        + num_len(m_cost)
        + num_len(lanes)
        + b64_len_usize(salt_len)
        + b64_len_usize(hash_len)
        + num_len(Version::DEFAULT.as_u32())
        + 1
}

// ---------------------------------------------------------------------------
// Base64
// ---------------------------------------------------------------------------

/// `to_base64(dst, dst_len, src, src_len)`.
///
/// Writes the unpadded base64 of `src` into `dst` and returns how many bytes
/// were written. No NUL terminator is written (the C writes one; Rust does not
/// need it), but the capacity check is kept identical: the C requires
/// `dst_len > olen`, so this requires `dst.len() > b64_len(src.len())`.
///
/// # Errors
///
/// [`Error::EncodingFail`] if `dst` is too small.
pub fn to_base64(dst: &mut [u8], src: &[u8]) -> Result<usize, Error> {
    let olen = b64_len_usize(src.len());
    if dst.len() <= olen {
        return Err(Error::EncodingFail);
    }

    let mut acc: u32 = 0;
    let mut acc_len: u32 = 0;
    let mut written = 0usize;

    for &byte in src {
        // The C writes `(acc << 8) + *buf++`; the low 8 bits of `acc << 8` are
        // zero, so `|` is the same value and cannot overflow in debug builds.
        acc = (acc << 8) | byte as u32;
        acc_len += 8;
        while acc_len >= 6 {
            acc_len -= 6;
            dst[written] = b64_byte_to_char((acc >> acc_len) & 0x3F);
            written += 1;
        }
    }
    if acc_len > 0 {
        dst[written] = b64_byte_to_char((acc << (6 - acc_len)) & 0x3F);
        written += 1;
    }

    debug_assert!(written == olen);
    Ok(written)
}

/// `from_base64(dst, dst_len, src)`.
///
/// Decodes until the first non-base64 byte. Returns
/// `(bytes_written, bytes_consumed)`, where `bytes_consumed` indexes the first
/// non-base64 byte in `src` — the equivalent of the pointer the C returns. The
/// end of the slice acts as the C's terminating NUL, which is itself not a
/// base64 character.
///
/// # Errors
///
/// [`Error::DecodingFail`] if `dst` is too small, if `acc_len > 4` at the end,
/// or if any buffered low bits are non-zero.
pub fn from_base64(dst: &mut [u8], src: &[u8]) -> Result<(usize, usize), Error> {
    let mut len = 0usize;
    let mut acc: u32 = 0;
    let mut acc_len: u32 = 0;
    let mut consumed = 0usize;

    loop {
        // Past the end of the slice, feed the NUL the C would have read.
        let c = match src.get(consumed) {
            Some(&byte) => byte as u32,
            None => 0,
        };
        let d = b64_char_to_byte(c);
        if d == 0xFF {
            break;
        }
        consumed += 1;
        // As in `to_base64`, `|` matches the C's `+` bit for bit.
        acc = (acc << 6) | d;
        acc_len += 6;
        if acc_len >= 8 {
            acc_len -= 8;
            // The C is `if ((len++) >= *dst_len) return NULL;`, i.e. the test
            // uses the pre-increment value.
            if len >= dst.len() {
                return Err(Error::DecodingFail);
            }
            dst[len] = ((acc >> acc_len) & 0xFF) as u8;
            len += 1;
        }
    }

    // An input length of 1 modulo 4 leaves 6 unprocessed bits, which is
    // invalid; otherwise 0, 2 or 4 bits are buffered and they must be zero.
    if acc_len > 4 || (acc & ((1u32 << acc_len) - 1)) != 0 {
        return Err(Error::DecodingFail);
    }

    Ok((len, consumed))
}

// ---------------------------------------------------------------------------
// decode_decimal
// ---------------------------------------------------------------------------

/// `decode_decimal(str, v)`.
///
/// Returns `(value, digits_consumed)`, or `None` when there is no digit at all,
/// when the encoding is not minimal (a leading `'0'` with more digits after
/// it), or when the value overflows.
///
/// The C accumulates in `unsigned long`, which is 64-bit on every target this
/// crate supports, so `u64` matches it. On a hypothetical 32-bit `unsigned
/// long` the outcome would still be the same, because every caller is
/// `DECIMAL_U32`, which rejects anything above `u32::MAX` anyway.
fn decode_decimal(src: &[u8]) -> Option<(u64, usize)> {
    let mut acc: u64 = 0;
    let mut i = 0usize;

    while let Some(&c) = src.get(i) {
        if !c.is_ascii_digit() {
            break;
        }
        let digit = (c - b'0') as u64;
        if acc > u64::MAX / 10 {
            return None;
        }
        acc *= 10;
        if digit > u64::MAX - acc {
            return None;
        }
        acc += digit;
        i += 1;
    }

    // `if (str == orig || (*orig == '0' && str != (orig + 1))) return NULL;`
    if i == 0 {
        return None;
    }
    if src[0] == b'0' && i != 1 {
        return None;
    }

    Some((acc, i))
}

// ---------------------------------------------------------------------------
// encode_string
// ---------------------------------------------------------------------------

/// The C's `SS`/`SX`/`SB` macros: a cursor that always keeps one byte spare for
/// the NUL terminator the C writes, so the capacity requirement is identical.
struct Writer<'a> {
    dst: &'a mut [u8],
    pos: usize,
}

impl Writer<'_> {
    /// `SS(str)`: `if (pp_len >= dst_len) return ARGON2_ENCODING_FAIL;`.
    fn put(&mut self, bytes: &[u8]) -> Result<(), Error> {
        let remaining = self.dst.len() - self.pos;
        if bytes.len() >= remaining {
            return Err(Error::EncodingFail);
        }
        let end = self.pos + bytes.len();
        self.dst[self.pos..end].copy_from_slice(bytes);
        self.pos = end;
        Ok(())
    }

    /// `SX(x)`: the decimal form of `x`, no allocation.
    fn put_u32(&mut self, value: u32) -> Result<(), Error> {
        // `u32::MAX` is 4294967295: ten digits.
        let mut buf = [0u8; 10];
        let mut i = buf.len();
        let mut n = value;
        loop {
            i -= 1;
            buf[i] = b'0' + (n % 10) as u8;
            n /= 10;
            if n == 0 {
                break;
            }
        }
        self.put(&buf[i..])
    }

    /// `SB(buf, len)`: base64, which does its own capacity check.
    fn put_base64(&mut self, src: &[u8]) -> Result<(), Error> {
        let written = to_base64(&mut self.dst[self.pos..], src)?;
        self.pos += written;
        Ok(())
    }
}

/// `validate_inputs(ctx)` as `encode_string` and `decode_string` run it.
///
/// `out_len` is the *tag* length, which is what `ctx->outlen` holds in both
/// call sites, and `pwd_len` is 0 (see the module-level divergence note).
fn validate_for_string(params: &Params, salt_len: usize, hash_len: usize) -> Result<(), Error> {
    validate_inputs(
        hash_len,
        0,
        salt_len,
        0,
        0,
        params.m_cost(),
        params.t_cost(),
        params.lanes(),
        params.threads(),
    )
}

/// `encode_string(dst, dst_len, ctx, type)`.
///
/// Writes the PHC string into `dst` (no NUL terminator) and returns its length.
/// Always emits `$v=`, as the C does.
///
/// `dst` must hold the string **plus one byte**, because the C reserves room
/// for its NUL terminator and this port keeps the capacity rule identical:
/// a buffer of exactly [`encoded_len`] bytes is what succeeds, in Rust and in
/// C alike.
///
/// `hash.len()` plays the role of `ctx->outlen` — it is the length that gets
/// encoded, so it, and not [`Params::output_len`], is what the leading
/// `validate_inputs()` checks. For a tag produced from these `params` the two
/// are the same value.
///
/// # Errors
///
/// [`Error::EncodingFail`] if `dst` is too small, or whatever
/// [`crate::params::validate_inputs`] returns — `encode_string` runs it first,
/// exactly like the C.
pub fn encode_string(
    dst: &mut [u8],
    algorithm: Algorithm,
    version: Version,
    params: &Params,
    salt: &[u8],
    hash: &[u8],
) -> Result<usize, Error> {
    validate_for_string(params, salt.len(), hash.len())?;

    let mut w = Writer { dst, pos: 0 };

    w.put(b"$")?;
    w.put(algorithm.as_str().as_bytes())?;

    w.put(b"$v=")?;
    w.put_u32(version.as_u32())?;

    w.put(b"$m=")?;
    w.put_u32(params.m_cost())?;
    w.put(b",t=")?;
    w.put_u32(params.t_cost())?;
    w.put(b",p=")?;
    w.put_u32(params.lanes())?;

    w.put(b"$")?;
    w.put_base64(salt)?;

    w.put(b"$")?;
    w.put_base64(hash)?;

    Ok(w.pos)
}

/// [`encode_string`] into a freshly allocated [`String`].
///
/// # Errors
///
/// As [`encode_string`], plus [`Error::MemoryAllocationError`] if the buffer
/// cannot be allocated (the C returns the same code when its `malloc` fails).
pub fn encode_string_alloc(
    algorithm: Algorithm,
    version: Version,
    params: &Params,
    salt: &[u8],
    hash: &[u8],
) -> Result<String, Error> {
    // Validate before allocating, so an over-long salt reports SaltTooLong
    // rather than failing to allocate a buffer sized from it.
    validate_for_string(params, salt.len(), hash.len())?;

    let capacity = encoded_len_usize(
        algorithm,
        params.t_cost(),
        params.m_cost(),
        params.lanes(),
        salt.len(),
        hash.len(),
    );

    let mut buf = alloc_zeroed_vec(capacity)?;
    let written = encode_string(&mut buf, algorithm, version, params, salt, hash)?;
    buf.truncate(written);

    // Every byte written comes from the base64 alphabet, the decimal digits or
    // the ASCII punctuation above, so this is always valid UTF-8.
    String::from_utf8(buf).map_err(|_| Error::EncodingFail)
}

/// A zeroed `Vec<u8>` of `len` bytes, without the abort-on-OOM of
/// `Vec::with_capacity`.
fn alloc_zeroed_vec(len: usize) -> Result<Vec<u8>, Error> {
    let mut v = Vec::new();
    v.try_reserve(len)
        .map_err(|_| Error::MemoryAllocationError)?;
    // Cannot reallocate: the capacity was just reserved.
    v.resize(len, 0);
    Ok(v)
}

// ---------------------------------------------------------------------------
// decode_string
// ---------------------------------------------------------------------------

/// The fields a PHC string yields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decoded {
    /// The algorithm named by the string. Always equal to the one requested,
    /// since a mismatch is a decoding failure.
    pub algorithm: Algorithm,
    /// The version. `0x10` when the `$v=` field is absent.
    pub version: Version,
    /// `m_cost`, `t_cost`, `lanes`, `threads == lanes`, and
    /// `output_len == hash.len()`.
    pub params: Params,
    /// The decoded salt.
    pub salt: Vec<u8>,
    /// The decoded tag.
    pub hash: Vec<u8>,
}

/// The remainder of `src` from `pos`, never panicking.
#[inline]
fn rest(src: &[u8], pos: usize) -> &[u8] {
    src.get(pos..).unwrap_or(&[])
}

/// The `CC(prefix)` macro: consume `prefix` or fail.
fn expect(src: &[u8], pos: &mut usize, prefix: &[u8]) -> Result<(), Error> {
    if rest(src, *pos).starts_with(prefix) {
        *pos += prefix.len();
        Ok(())
    } else {
        Err(Error::DecodingFail)
    }
}

/// The `CC_opt(prefix, code)` macro: consume `prefix` if it is there.
fn expect_opt(src: &[u8], pos: &mut usize, prefix: &[u8]) -> bool {
    if rest(src, *pos).starts_with(prefix) {
        *pos += prefix.len();
        true
    } else {
        false
    }
}

/// The `DECIMAL_U32(x)` macro.
fn decimal_u32(src: &[u8], pos: &mut usize) -> Result<u32, Error> {
    let (value, consumed) = decode_decimal(rest(src, *pos)).ok_or(Error::DecodingFail)?;
    if value > u32::MAX as u64 {
        return Err(Error::DecodingFail);
    }
    *pos += consumed;
    Ok(value as u32)
}

/// The `BIN(buf, max_len, len)` macro.
///
/// The C sizes the destination at `strlen(encoded)` (see `argon2_verify`), so
/// the "output buffer too small" branch of `from_base64` is unreachable there.
/// The bound used here — three bytes out per four characters in — is likewise
/// never exceeded, so the two agree.
fn decode_bin(src: &[u8], pos: &mut usize) -> Result<Vec<u8>, Error> {
    let tail = rest(src, *pos);
    // `n` base64 characters decode to `floor(3n / 4)` bytes; `n/4*3 + 3` is an
    // upper bound for every `n`, and cannot overflow for any real slice.
    let max_len = tail.len() / 4 * 3 + 3;

    let mut buf = alloc_zeroed_vec(max_len)?;
    let (written, consumed) = from_base64(&mut buf, tail)?;
    // `bin_len > UINT32_MAX` is a decoding failure in the C.
    if written > u32::MAX as usize {
        return Err(Error::DecodingFail);
    }
    buf.truncate(written);
    *pos += consumed;
    Ok(buf)
}

/// `decode_string(ctx, str, type)`.
///
/// # Errors
///
/// [`Error::DecodingFail`] for a malformed string, or whatever
/// [`crate::params::validate_inputs`] returns — the C runs the full validation
/// before accepting the string, so a well-formed string with a zero-length salt
/// yields [`Error::SaltTooShort`] and not [`Error::DecodingFail`].
///
/// See the module documentation for the three known divergences from the C
/// (unrepresentable versions, embedded NULs, and the password length).
pub fn decode_string(encoded: &str, algorithm: Algorithm) -> Result<Decoded, Error> {
    let src = encoded.as_bytes();
    let mut pos = 0usize;

    // argon2.c:268-271
    //   encoded_len = strlen(encoded);
    //   if (encoded_len > UINT32_MAX) return ARGON2_DECODING_FAIL;
    //
    // The C puts this in `argon2_verify`, one level up, and computes
    // `max_field_len` from it. Here it lives in the decoder because all four
    // verify entry points funnel through this function, so one check covers
    // them and cannot drift; through the public API the behaviour is identical.
    // Reachable only from a `&str` over 4 GiB.
    if src.len() > u32::MAX as usize {
        return Err(Error::DecodingFail);
    }

    // CC("$"); CC(type_string);
    //
    // No `ARGON2_INCORRECT_TYPE` branch: `argon2_type2string` only returns NULL
    // for a type outside the enum, which `Algorithm` cannot represent.
    expect(src, &mut pos, b"$")?;
    expect(src, &mut pos, algorithm.as_str().as_bytes())?;

    // ctx->version = ARGON2_VERSION_10; CC_opt("$v=", DECIMAL_U32(version));
    let mut version_value = Version::V0x10.as_u32();
    if expect_opt(src, &mut pos, b"$v=") {
        version_value = decimal_u32(src, &mut pos)?;
    }

    expect(src, &mut pos, b"$m=")?;
    let m_cost = decimal_u32(src, &mut pos)?;
    expect(src, &mut pos, b",t=")?;
    let t_cost = decimal_u32(src, &mut pos)?;
    expect(src, &mut pos, b",p=")?;
    let lanes = decimal_u32(src, &mut pos)?;
    // `ctx->threads = ctx->lanes;`
    let threads = lanes;

    expect(src, &mut pos, b"$")?;
    let salt = decode_bin(src, &mut pos)?;
    expect(src, &mut pos, b"$")?;
    let hash = decode_bin(src, &mut pos)?;

    // "On return, must have valid context": the full validate_inputs(), in the
    // C's order, before the trailing-character check.
    validate_inputs(
        hash.len(),
        0,
        salt.len(),
        0,
        0,
        m_cost,
        t_cost,
        lanes,
        threads,
    )?;

    // "Can't have any additional characters".
    if pos != src.len() {
        return Err(Error::DecodingFail);
    }

    // Last, so that every error code above still matches the C exactly.
    let version = Version::from_u32(version_value).ok_or(Error::DecodingFail)?;

    // Cannot fail: `validate_inputs` above already accepted these values, and
    // `Params` re-checks them with a salt length that is valid by construction.
    let params = Params::new_with_threads(m_cost, t_cost, lanes, threads, hash.len())?;

    Ok(Decoded {
        algorithm,
        version,
        params,
        salt,
        hash,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // From `phc-winner-argon2/src/test.c`.
    const V13_ARGON2I: &str = "$argon2i$v=19$m=65536,t=2,p=1$c29tZXNhbHQ\
                               $wWKIMhR9lyDFvRz9YTZweHKfbftvj+qf+YFY4NeBbtA";
    const V10_ARGON2I: &str = "$argon2i$m=65536,t=2,p=1$c29tZXNhbHQ\
                               $9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ";
    const V13_ARGON2ID: &str = "$argon2id$v=19$m=65536,t=2,p=1$c29tZXNhbHQ\
                                $CTFhFdXPJO1aFaMaO6Mm5c8y7cJHAph8ArZWb2GRPPc";

    /// `f6c4db4a...`, the raw tag of the v=0x10 Argon2i vector in test.c.
    const V10_TAG: [u8; 32] = [
        0xf6, 0xc4, 0xdb, 0x4a, 0x54, 0xe2, 0xa3, 0x70, 0x62, 0x7a, 0xff, 0x3d, 0xb6, 0x17, 0x6b,
        0x94, 0xa2, 0xa2, 0x09, 0xa6, 0x2c, 0x8e, 0x36, 0x15, 0x27, 0x11, 0x80, 0x2f, 0x7b, 0x30,
        0xc6, 0x94,
    ];

    fn b64(src: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; b64_len_usize(src.len()) + 1];
        let n = to_base64(&mut out, src).expect("buffer sized by b64_len");
        out.truncate(n);
        out
    }

    fn unb64(src: &[u8]) -> Result<Vec<u8>, Error> {
        let mut out = vec![0u8; src.len()];
        let (n, consumed) = from_base64(&mut out, src)?;
        assert_eq!(consumed, src.len(), "test inputs are pure base64");
        out.truncate(n);
        Ok(out)
    }

    // -- lengths ------------------------------------------------------------

    #[test]
    fn b64_len_matches_c() {
        assert_eq!(b64_len(0), 0);
        assert_eq!(b64_len(1), 2);
        assert_eq!(b64_len(2), 3);
        assert_eq!(b64_len(3), 4);
        assert_eq!(b64_len(4), 6);
        // "somesalt" -> "c29tZXNhbHQ", "…" -> the 43-char tag.
        assert_eq!(b64_len(8), 11);
        assert_eq!(b64_len(32), 43);
        // Cross-check against the encoder for every small length.
        for len in 0u32..64 {
            let src = vec![0xABu8; len as usize];
            assert_eq!(b64(&src).len(), b64_len(len), "len {len}");
        }
    }

    #[test]
    fn num_len_matches_c() {
        assert_eq!(num_len(0), 1);
        assert_eq!(num_len(9), 1);
        assert_eq!(num_len(10), 2);
        assert_eq!(num_len(19), 2);
        assert_eq!(num_len(65536), 5);
        assert_eq!(num_len(u32::MAX), 10);
    }

    #[test]
    fn encoded_len_matches_the_c_vector() {
        // $argon2id$v=19$m=65536,t=2,p=1$<11>$<43> is 86 chars + NUL.
        let n = encoded_len(Algorithm::Argon2id, 2, 65536, 1, 8, 32);
        assert_eq!(n, V13_ARGON2ID.len() + 1);
        assert_eq!(n, 87);
        assert_eq!(
            encoded_len(Algorithm::Argon2i, 2, 65536, 1, 8, 32),
            V13_ARGON2I.len() + 1
        );
    }

    /// Pins the claim in `encoded_len`'s `# Argument order` section: the C's
    /// `argon2_encodedlen` (`argon2.c:447`) takes `t_cost` before `m_cost`,
    /// this port keeps that order, and it is the reverse of `Params::new`. A
    /// caller who swaps the two gets the same number, because both costs enter
    /// the result only as `num_len(t_cost) + num_len(m_cost)` and that sum is
    /// blind to which digit count came from which cost. If the formula ever
    /// stops being a plain sum of the two, this fails and the doc gets fixed
    /// with it.
    #[test]
    fn encoded_len_is_symmetric_in_m_and_t() {
        // The pair the doc example uses, the C's own test vector, and the
        // extreme: `u32::MAX` is ten digits against one, the widest the two
        // terms can differ.
        assert_eq!(encoded_len(Algorithm::Argon2id, 3, 65536, 1, 16, 32), 98);
        assert_eq!(encoded_len(Algorithm::Argon2id, 65536, 3, 1, 16, 32), 98);
        assert_eq!(encoded_len(Algorithm::Argon2id, 2, 65536, 1, 8, 32), 87);
        assert_eq!(encoded_len(Algorithm::Argon2id, 65536, 2, 1, 8, 32), 87);
        assert_eq!(encoded_len(Algorithm::Argon2id, 1, u32::MAX, 1, 16, 32), 103);
        assert_eq!(encoded_len(Algorithm::Argon2id, u32::MAX, 1, 1, 16, 32), 103);

        // Every place `num_len` changes answer, both sides of each step, over
        // all three algorithm strings, so the property is pinned rather than
        // sampled at a few lucky points.
        const BOUNDARIES: [u32; 22] = [
            0,
            1,
            9,
            10,
            99,
            100,
            999,
            1_000,
            9_999,
            10_000,
            99_999,
            100_000,
            999_999,
            1_000_000,
            9_999_999,
            10_000_000,
            99_999_999,
            100_000_000,
            999_999_999,
            1_000_000_000,
            65536,
            u32::MAX,
        ];
        for algorithm in [Algorithm::Argon2d, Algorithm::Argon2i, Algorithm::Argon2id] {
            for t_cost in BOUNDARIES {
                for m_cost in BOUNDARIES {
                    assert_eq!(
                        encoded_len(algorithm, t_cost, m_cost, 1, 16, 32),
                        encoded_len(algorithm, m_cost, t_cost, 1, 16, 32),
                        "{algorithm:?} t_cost={t_cost} m_cost={m_cost}"
                    );
                }
            }
        }
    }

    // -- base64 -------------------------------------------------------------

    #[test]
    fn b64_tables_are_the_standard_alphabet() {
        const ALPHABET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (value, &ch) in ALPHABET.iter().enumerate() {
            assert_eq!(b64_byte_to_char(value as u32), ch, "value {value}");
            assert_eq!(b64_char_to_byte(ch as u32), value as u32, "char {ch}");
        }
    }

    #[test]
    fn b64_char_to_byte_rejects_everything_else() {
        // The 'A' quirk: a computed 0 is only valid for 'A'.
        assert_eq!(b64_char_to_byte(b'A' as u32), 0);
        for c in 0u32..256 {
            let is_b64 = (c as u8).is_ascii_alphanumeric() || c == b'+' as u32 || c == b'/' as u32;
            if !is_b64 {
                assert_eq!(b64_char_to_byte(c), 0xFF, "char {c} must be invalid");
            }
        }
        assert_eq!(b64_char_to_byte(b'=' as u32), 0xFF); // no padding
        assert_eq!(b64_char_to_byte(0), 0xFF); // NUL terminator
        assert_eq!(b64_char_to_byte(b'$' as u32), 0xFF); // field separator

        // Bytes >= 0x80 are rejected here. The C's are not, wherever `char` is
        // signed — it returns 63 for all 128 of them. See the module docs; this
        // loop is the pin for the port's (stricter, portable) choice.
        for c in 0x80u32..256 {
            assert_eq!(b64_char_to_byte(c), 0xFF, "byte {c:#04x} must be invalid");
        }
    }

    /// `b64_char_to_byte` over `0..=127`, dumped from the C reference.
    ///
    /// Produced by a harness that `#include`s `encoding.c` (the function is
    /// `static`) and prints `b64_char_to_byte(i)` for every `i`, linked against
    /// `phc-winner-argon2/libargon2.a`. Only the ASCII half is transcribed: the
    /// upper half is the signed-`char` bug documented at the top of this file,
    /// where the C returns 63 for all 128 values.
    #[rustfmt::skip]
    const C_CHAR_TO_BYTE_ASCII: [u32; 128] = [
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  62, 255, 255, 255,  63,
         52,  53,  54,  55,  56,  57,  58,  59,  60,  61, 255, 255, 255, 255, 255, 255,
        255,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
         15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25, 255, 255, 255, 255, 255,
        255,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,
         41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51, 255, 255, 255, 255, 255,
    ];

    #[test]
    fn b64_char_to_byte_matches_the_c_dump() {
        for (c, &want) in C_CHAR_TO_BYTE_ASCII.iter().enumerate() {
            assert_eq!(b64_char_to_byte(c as u32), want, "char {c}");
        }
    }

    /// `b64_byte_to_char` over `0..64`, dumped from the same C harness.
    #[rustfmt::skip]
    const C_BYTE_TO_CHAR: [u8; 64] = [
        65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 97, 98, 99, 100, 101, 102,
        103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
        119, 120, 121, 122, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 43, 47,
    ];

    #[test]
    fn b64_byte_to_char_matches_the_c_dump() {
        for (x, &want) in C_BYTE_TO_CHAR.iter().enumerate() {
            assert_eq!(b64_byte_to_char(x as u32), want, "value {x}");
        }
    }

    #[test]
    fn non_ascii_bytes_are_rejected_in_a_field() {
        // Measured: the C decodes both of these, identically, to the salt
        // ff ff 6d 65 73 61 6c 74, because 0xC3 and 0xA9 each read as '/'.
        // This port rejects the first and accepts the second.
        let utf8 = "$argon2i$v=19$m=65536,t=2,p=1$é9tZXNhbHQ\
                    $9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ";
        assert_eq!(
            decode_string(utf8, Algorithm::Argon2i),
            Err(Error::DecodingFail)
        );
        let slashes = "$argon2i$v=19$m=65536,t=2,p=1$//9tZXNhbHQ\
                       $9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ";
        let d = decode_string(slashes, Algorithm::Argon2i).unwrap();
        assert_eq!(d.salt, [0xff, 0xff, 0x6d, 0x65, 0x73, 0x61, 0x6c, 0x74]);
    }

    #[test]
    fn base64_known_vectors() {
        assert_eq!(b64(b"somesalt"), b"c29tZXNhbHQ");
        assert_eq!(b64(b"diffsalt"), b"ZGlmZnNhbHQ");
        assert_eq!(
            b64(&V10_TAG),
            b"9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ"
        );
        assert_eq!(unb64(b"c29tZXNhbHQ").unwrap(), b"somesalt");
        assert_eq!(
            unb64(b"9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ").unwrap(),
            &V10_TAG
        );
    }

    #[test]
    fn base64_round_trips_every_short_length() {
        for len in 0usize..96 {
            let src: Vec<u8> = (0..len)
                .map(|i| (i as u8).wrapping_mul(37) ^ 0x5A)
                .collect();
            let encoded = b64(&src);
            assert_eq!(unb64(&encoded).unwrap(), src, "len {len}");
        }
    }

    #[test]
    fn to_base64_rejects_a_tight_buffer() {
        // The C requires dst_len > olen, i.e. room for the NUL as well.
        let mut exact = [0u8; 11];
        assert_eq!(to_base64(&mut exact, b"somesalt"), Err(Error::EncodingFail));
        let mut roomy = [0u8; 12];
        assert_eq!(to_base64(&mut roomy, b"somesalt"), Ok(11));
        // Even an empty input needs one byte for the terminator.
        assert_eq!(to_base64(&mut [], b""), Err(Error::EncodingFail));
        assert_eq!(to_base64(&mut [0u8; 1], b""), Ok(0));
    }

    #[test]
    fn from_base64_stops_at_the_first_non_b64_char() {
        let mut out = [0u8; 16];
        let (written, consumed) = from_base64(&mut out, b"c29tZXNhbHQ$rest").unwrap();
        assert_eq!(written, 8);
        assert_eq!(consumed, 11);
        assert_eq!(&out[..8], b"somesalt");

        // An empty run is fine and consumes nothing (this is how the C accepts
        // "$argon2i$m=…,p=1$$<tag>" and only later reports SaltTooShort).
        let (written, consumed) = from_base64(&mut out, b"$tag").unwrap();
        assert_eq!((written, consumed), (0, 0));
    }

    #[test]
    fn from_base64_rejects_leftover_bits() {
        let mut out = [0u8; 16];
        // 5 chars -> 30 bits -> acc_len == 6 > 4.
        assert_eq!(
            from_base64(&mut out, b"AAAAA"),
            Err(Error::DecodingFail),
            "acc_len > 4"
        );
        // 1 char -> 6 bits -> acc_len == 6 > 4.
        assert_eq!(from_base64(&mut out, b"A"), Err(Error::DecodingFail));
        // 2 chars, 4 buffered bits, non-zero: 'B' == 1.
        assert_eq!(from_base64(&mut out, b"AB"), Err(Error::DecodingFail));
        // Same shape but the buffered bits are zero.
        assert_eq!(from_base64(&mut out, b"AA"), Ok((1, 2)));
        // 3 chars, 2 buffered bits, non-zero: 'B' == 000001.
        assert_eq!(from_base64(&mut out, b"AAB"), Err(Error::DecodingFail));
        assert_eq!(from_base64(&mut out, b"AAA"), Ok((2, 3)));
    }

    #[test]
    fn from_base64_rejects_a_short_buffer() {
        let mut out = [0u8; 4];
        assert_eq!(
            from_base64(&mut out, b"c29tZXNhbHQ"),
            Err(Error::DecodingFail)
        );
        // The C's `if ((len++) >= *dst_len)` tests the pre-increment value, so
        // an exactly-sized buffer is fine and one byte less is not.
        let mut exact = [0u8; 8];
        assert_eq!(from_base64(&mut exact, b"c29tZXNhbHQ"), Ok((8, 11)));
        assert_eq!(&exact, b"somesalt");
        let mut tight = [0u8; 7];
        assert_eq!(
            from_base64(&mut tight, b"c29tZXNhbHQ"),
            Err(Error::DecodingFail)
        );
    }

    // -- decode_decimal -----------------------------------------------------

    #[test]
    fn decode_decimal_matches_c() {
        assert_eq!(decode_decimal(b"0"), Some((0, 1)));
        assert_eq!(decode_decimal(b"19"), Some((19, 2)));
        assert_eq!(decode_decimal(b"65536,t=2"), Some((65536, 5)));
        assert_eq!(decode_decimal(b"4294967295"), Some((4294967295, 10)));
        // No digits at all.
        assert_eq!(decode_decimal(b""), None);
        assert_eq!(decode_decimal(b"$"), None);
        assert_eq!(decode_decimal(b"x1"), None);
        // Non-minimal.
        assert_eq!(decode_decimal(b"01"), None);
        assert_eq!(decode_decimal(b"00"), None);
        assert_eq!(decode_decimal(b"0019"), None);
        // Overflow of `unsigned long`.
        assert_eq!(
            decode_decimal(b"18446744073709551615"),
            Some((u64::MAX, 20))
        );
        assert_eq!(decode_decimal(b"18446744073709551616"), None);
        assert_eq!(decode_decimal(b"99999999999999999999999"), None);
    }

    #[test]
    fn decode_decimal_matches_the_c_dump() {
        // Every pair below was produced by running the C's `decode_decimal`
        // (it is `static`, so via a harness that #includes encoding.c) linked
        // against phc-winner-argon2/libargon2.a. `None` is the C's NULL.
        /// `(input, decode_decimal(input))`, the C's NULL being `None`.
        type Case = (&'static [u8], Option<(u64, usize)>);

        let cases: &[Case] = &[
            (b"", None),
            (b"0", Some((0, 1))),
            (b"00", None),
            (b"000", None),
            (b"01", None),
            (b"0019", None),
            (b"1", Some((1, 1))),
            (b"9", Some((9, 1))),
            (b"19", Some((19, 2))),
            (b"10", Some((10, 2))),
            (b"007", None),
            (b"4294967295", Some((4294967295, 10))),
            (b"4294967296", Some((4294967296, 10))),
            (b"9223372036854775807", Some((9223372036854775807, 19))),
            (b"18446744073709551615", Some((18446744073709551615, 20))),
            (b"18446744073709551616", None),
            (b"18446744073709551620", None),
            (b"19999999999999999999", None),
            (b"20000000000000000000", None),
            (b"99999999999999999999", None),
            (b"1000000000000000000000000", None),
            (b"65536,t=2", Some((65536, 5))),
            (b"1,t=2", Some((1, 1))),
            (b"x", None),
            (b"1x", Some((1, 1))),
            (b" 1", None),
            (b"+1", None),
            (b"-1", None),
            (b"1 ", Some((1, 1))),
            // "0" is minimal on its own, so the 'x' just ends the run.
            (b"0x10", Some((0, 1))),
            (b"12$", Some((12, 2))),
            (b"0,", Some((0, 1))),
            (b"10$", Some((10, 2))),
            (b"$", None),
            (b"1234567890", Some((1234567890, 10))),
        ];
        for (input, want) in cases {
            assert_eq!(
                decode_decimal(input),
                *want,
                "input {:?}",
                core::str::from_utf8(input).unwrap_or("<non-utf8>")
            );
        }
    }

    #[test]
    fn decimal_fields_are_u32_bounded() {
        // DECIMAL_U32 rejects anything above UINT32_MAX.
        let too_big = "$argon2i$v=19$m=4294967296,t=2,p=1$c29tZXNhbHQ\
                       $9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ";
        assert_eq!(
            decode_string(too_big, Algorithm::Argon2i),
            Err(Error::DecodingFail)
        );
        let leading_zero = "$argon2i$v=19$m=065536,t=2,p=1$c29tZXNhbHQ\
                            $9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ";
        assert_eq!(
            decode_string(leading_zero, Algorithm::Argon2i),
            Err(Error::DecodingFail)
        );
    }

    // -- encode -------------------------------------------------------------

    #[test]
    fn encode_matches_the_official_vectors() {
        let params = Params::new(65536, 2, 1, 32).unwrap();
        let tag = unb64(b"wWKIMhR9lyDFvRz9YTZweHKfbftvj+qf+YFY4NeBbtA").unwrap();
        let encoded = encode_string_alloc(
            Algorithm::Argon2i,
            Version::V0x13,
            &params,
            b"somesalt",
            &tag,
        )
        .unwrap();
        assert_eq!(encoded, V13_ARGON2I);

        let tag = unb64(b"CTFhFdXPJO1aFaMaO6Mm5c8y7cJHAph8ArZWb2GRPPc").unwrap();
        let encoded = encode_string_alloc(
            Algorithm::Argon2id,
            Version::V0x13,
            &params,
            b"somesalt",
            &tag,
        )
        .unwrap();
        assert_eq!(encoded, V13_ARGON2ID);

        // The C always emits "$v=", even for 0x10.
        let encoded = encode_string_alloc(
            Algorithm::Argon2i,
            Version::V0x10,
            &params,
            b"somesalt",
            &V10_TAG,
        )
        .unwrap();
        assert_eq!(
            encoded,
            "$argon2i$v=16$m=65536,t=2,p=1$c29tZXNhbHQ\
             $9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ"
        );
    }

    #[test]
    fn encode_needs_encoded_len_bytes_exactly() {
        let params = Params::new(65536, 2, 1, 32).unwrap();
        let want = encoded_len(Algorithm::Argon2i, 2, 65536, 1, 8, 32);
        assert_eq!(want, V13_ARGON2I.len() + 1);

        let mut buf = vec![0u8; want];
        let n = encode_string(
            &mut buf,
            Algorithm::Argon2i,
            Version::V0x13,
            &params,
            b"somesalt",
            &V10_TAG,
        )
        .unwrap();
        assert_eq!(n, want - 1);

        // One byte less is a failure, exactly as in the C.
        let mut buf = vec![0u8; want - 1];
        assert_eq!(
            encode_string(
                &mut buf,
                Algorithm::Argon2i,
                Version::V0x13,
                &params,
                b"somesalt",
                &V10_TAG,
            ),
            Err(Error::EncodingFail)
        );
    }

    #[test]
    fn encode_validates_first() {
        let params = Params::new(65536, 2, 1, 32).unwrap();
        // Short salt: validate_inputs runs before anything is written.
        assert_eq!(
            encode_string_alloc(
                Algorithm::Argon2i,
                Version::V0x13,
                &params,
                b"short",
                &V10_TAG
            ),
            Err(Error::SaltTooShort)
        );
        // Short tag.
        assert_eq!(
            encode_string_alloc(
                Algorithm::Argon2i,
                Version::V0x13,
                &params,
                b"somesalt",
                &[0u8; 3]
            ),
            Err(Error::OutputTooShort)
        );
    }

    // -- decode -------------------------------------------------------------

    #[test]
    fn decode_the_v13_vector() {
        let d = decode_string(V13_ARGON2I, Algorithm::Argon2i).unwrap();
        assert_eq!(d.algorithm, Algorithm::Argon2i);
        assert_eq!(d.version, Version::V0x13);
        assert_eq!(d.params.m_cost(), 65536);
        assert_eq!(d.params.t_cost(), 2);
        assert_eq!(d.params.lanes(), 1);
        assert_eq!(d.params.output_len(), 32);
        assert_eq!(d.salt, b"somesalt");
        assert_eq!(
            d.hash,
            unb64(b"wWKIMhR9lyDFvRz9YTZweHKfbftvj+qf+YFY4NeBbtA").unwrap()
        );
    }

    #[test]
    fn decode_defaults_the_version_to_0x10() {
        let d = decode_string(V10_ARGON2I, Algorithm::Argon2i).unwrap();
        assert_eq!(d.version, Version::V0x10);
        assert_eq!(d.salt, b"somesalt");
        assert_eq!(d.hash, &V10_TAG);
    }

    #[test]
    fn decode_sets_threads_to_lanes() {
        let s = "$argon2id$v=19$m=65536,t=2,p=4$c29tZXNhbHQ\
                 $CTFhFdXPJO1aFaMaO6Mm5c8y7cJHAph8ArZWb2GRPPc";
        let d = decode_string(s, Algorithm::Argon2id).unwrap();
        assert_eq!(d.params.lanes(), 4);
        assert_eq!(d.params.threads(), 4);
    }

    #[test]
    fn encode_decode_round_trip() {
        let salt = b"0123456789abcdef";
        let tag: Vec<u8> = (0u8..48).collect();
        for algorithm in Algorithm::ALL {
            for version in Version::ALL {
                for lanes in [1u32, 2, 255] {
                    let params = Params::new(1 << 16, 3, lanes, tag.len()).unwrap();
                    let encoded =
                        encode_string_alloc(algorithm, version, &params, salt, &tag).unwrap();
                    let d = decode_string(&encoded, algorithm).unwrap();
                    assert_eq!(d.algorithm, algorithm);
                    assert_eq!(d.version, version);
                    assert_eq!(d.params, params);
                    assert_eq!(d.salt, salt);
                    assert_eq!(d.hash, tag);
                    // And re-encoding is byte-identical.
                    assert_eq!(
                        encode_string_alloc(d.algorithm, d.version, &d.params, &d.salt, &d.hash)
                            .unwrap(),
                        encoded
                    );
                }
            }
        }
    }

    // The four malformed strings from `src/test.c`, both version flavours.

    #[test]
    fn decode_rejects_a_missing_dollar_before_the_salt() {
        // "…,p=1c29tZXNhbHQ$…": the '$' after p=1 is gone.
        let v10 = "$argon2i$m=65536,t=2,p=1c29tZXNhbHQ\
                   $9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ";
        assert_eq!(
            decode_string(v10, Algorithm::Argon2i),
            Err(Error::DecodingFail)
        );
        let v13 = "$argon2i$v=19$m=65536,t=2,p=1c29tZXNhbHQ\
                   $wWKIMhR9lyDFvRz9YTZweHKfbftvj+qf+YFY4NeBbtA";
        assert_eq!(
            decode_string(v13, Algorithm::Argon2i),
            Err(Error::DecodingFail)
        );
    }

    #[test]
    fn decode_rejects_a_missing_dollar_before_the_tag() {
        // The salt and tag run together into one 54-character base64 field,
        // which decodes cleanly (54 chars leave 4 zero bits); the failure comes
        // from the CC("$") that follows.
        let v10 = "$argon2i$m=65536,t=2,p=1$c29tZXNhbHQ\
                   9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ";
        assert_eq!(
            decode_string(v10, Algorithm::Argon2i),
            Err(Error::DecodingFail)
        );
        let v13 = "$argon2i$v=19$m=65536,t=2,p=1$c29tZXNhbHQ\
                   wWKIMhR9lyDFvRz9YTZweHKfbftvj+qf+YFY4NeBbtA";
        assert_eq!(
            decode_string(v13, Algorithm::Argon2i),
            Err(Error::DecodingFail)
        );
    }

    #[test]
    fn decode_reports_salt_too_short_not_decoding_fail() {
        // This is the distinction tests/vectors.rs relies on: the string parses,
        // and it is validate_inputs() that rejects it.
        let v10 = "$argon2i$m=65536,t=2,p=1$\
                   $9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ";
        assert_eq!(
            decode_string(v10, Algorithm::Argon2i),
            Err(Error::SaltTooShort)
        );
        let v13 = "$argon2i$v=19$m=65536,t=2,p=1$\
                   $9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ";
        assert_eq!(
            decode_string(v13, Algorithm::Argon2i),
            Err(Error::SaltTooShort)
        );
        // A 7-byte salt is also too short, and still not a DecodingFail.
        let short = "$argon2i$v=19$m=65536,t=2,p=1$c2hvcnRz\
                     $9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ";
        assert_eq!(
            decode_string(short, Algorithm::Argon2i),
            Err(Error::SaltTooShort)
        );
    }

    #[test]
    fn decode_argon2i_is_a_prefix_of_argon2id() {
        // CC("argon2i") matches the first seven characters of "argon2id"; the
        // leftover 'd' then fails the CC("$m=") (or the CC_opt("$v=")).
        assert_eq!(
            decode_string(V13_ARGON2ID, Algorithm::Argon2i),
            Err(Error::DecodingFail)
        );
        let v10_id = "$argon2id$m=65536,t=2,p=1$c29tZXNhbHQ\
                      $9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ";
        assert_eq!(
            decode_string(v10_id, Algorithm::Argon2i),
            Err(Error::DecodingFail)
        );
        // And the other way round: "argon2i" is not "argon2id".
        assert_eq!(
            decode_string(V13_ARGON2I, Algorithm::Argon2id),
            Err(Error::DecodingFail)
        );
        assert_eq!(
            decode_string(V13_ARGON2I, Algorithm::Argon2d),
            Err(Error::DecodingFail)
        );
        // The correct type still works, of course.
        assert!(decode_string(V13_ARGON2ID, Algorithm::Argon2id).is_ok());
    }

    #[test]
    fn decode_rejects_structural_damage() {
        let cases: &[&str] = &[
            "",
            "$",
            "argon2i$v=19$m=65536,t=2,p=1$c29tZXNhbHQ$9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ",
            "$argon2i",
            "$argon2i$v=19",
            "$argon2i$v=19$m=65536,t=2,p=1",
            "$argon2i$v=19$m=65536,t=2,p=1$c29tZXNhbHQ",
            // 'x' is not a decimal digit.
            "$argon2i$v=19$m=x,t=2,p=1$c29tZXNhbHQ$9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ",
            // Fields out of order.
            "$argon2i$v=19$t=2,m=65536,p=1$c29tZXNhbHQ$9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ",
            // Trailing junk after the tag.
            "$argon2i$v=19$m=65536,t=2,p=1$c29tZXNhbHQ$9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ$",
            "$argon2i$v=19$m=65536,t=2,p=1$c29tZXNhbHQ$9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ ",
            // '=' padding is not part of the alphabet, so it ends the field.
            "$argon2i$v=19$m=65536,t=2,p=1$c29tZXNhbHQ=$9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ",
        ];
        for case in cases {
            assert_eq!(
                decode_string(case, Algorithm::Argon2i),
                Err(Error::DecodingFail),
                "expected DecodingFail for {case:?}"
            );
        }
    }

    #[test]
    fn decode_surfaces_the_c_validation_codes() {
        // A 3-byte tag: outlen is checked first.
        assert_eq!(
            decode_string(
                "$argon2i$v=19$m=65536,t=2,p=1$c29tZXNhbHQ$AAAA",
                Algorithm::Argon2i
            ),
            Err(Error::OutputTooShort)
        );
        // m_cost < ARGON2_MIN_MEMORY.
        assert_eq!(
            decode_string(
                "$argon2i$v=19$m=1,t=2,p=1$c29tZXNhbHQ$9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ",
                Algorithm::Argon2i
            ),
            Err(Error::MemoryTooLittle)
        );
        // m_cost < 8 * lanes.
        assert_eq!(
            decode_string(
                "$argon2i$v=19$m=16,t=2,p=4$c29tZXNhbHQ$9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ",
                Algorithm::Argon2i
            ),
            Err(Error::MemoryTooLittle)
        );
        // t_cost < ARGON2_MIN_TIME.
        assert_eq!(
            decode_string(
                "$argon2i$v=19$m=65536,t=0,p=1$c29tZXNhbHQ$9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ",
                Algorithm::Argon2i
            ),
            Err(Error::TimeTooSmall)
        );
        // lanes < ARGON2_MIN_LANES.
        assert_eq!(
            decode_string(
                "$argon2i$v=19$m=65536,t=2,p=0$c29tZXNhbHQ$9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ",
                Algorithm::Argon2i
            ),
            Err(Error::LanesTooFew)
        );
        // lanes > ARGON2_MAX_LANES (16777215).
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            decode_string(
                "$argon2i$v=19$m=4294967295,t=2,p=16777216$c29tZXNhbHQ$9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ",
                Algorithm::Argon2i
            ),
            Err(Error::LanesTooMany)
        );
        // On a 32-bit target ARGON2_MAX_MEMORY is 2 MiB (the C's own
        // pointer-width rule), so — exactly as the C on 32-bit — the memory
        // check fires before the lanes check gets a chance to.
        #[cfg(target_pointer_width = "32")]
        assert_eq!(
            decode_string(
                "$argon2i$v=19$m=4294967295,t=2,p=16777216$c29tZXNhbHQ$9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ",
                Algorithm::Argon2i
            ),
            Err(Error::MemoryTooMuch)
        );
    }

    #[test]
    fn validation_runs_before_the_trailing_character_check() {
        // Both wrong: the C returns SALT_TOO_SHORT because validate_inputs()
        // comes first.
        let s = "$argon2i$v=19$m=65536,t=2,p=1$\
                 $9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ!!!";
        assert_eq!(
            decode_string(s, Algorithm::Argon2i),
            Err(Error::SaltTooShort)
        );
    }

    #[test]
    fn decode_rejects_an_unrepresentable_version() {
        // Documented divergence: the C accepts this (validate_inputs never
        // looks at the version) and treats it as 0x13.
        let s = "$argon2i$v=99$m=65536,t=2,p=1$c29tZXNhbHQ\
                 $9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ";
        assert_eq!(
            decode_string(s, Algorithm::Argon2i),
            Err(Error::DecodingFail)
        );
        // …but an earlier error still wins, so the codes stay C-compatible.
        let s = "$argon2i$v=99$m=65536,t=2,p=1$$9sTbSlTio3Biev89thdrlKKiCaYsjjYVJxGAL3swxpQ";
        assert_eq!(
            decode_string(s, Algorithm::Argon2i),
            Err(Error::SaltTooShort)
        );
    }

    #[test]
    fn decode_accepts_a_long_salt_and_tag() {
        let salt: Vec<u8> = (0u8..=255).collect();
        let tag: Vec<u8> = (0u8..=200).rev().collect();
        let params = Params::new(1 << 16, 1, 1, tag.len()).unwrap();
        let encoded =
            encode_string_alloc(Algorithm::Argon2d, Version::V0x13, &params, &salt, &tag).unwrap();
        let d = decode_string(&encoded, Algorithm::Argon2d).unwrap();
        assert_eq!(d.salt, salt);
        assert_eq!(d.hash, tag);
        assert_eq!(d.params.output_len(), tag.len());
    }
}
