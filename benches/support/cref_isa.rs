//! What instruction set actually landed in the compiled `phc-winner-argon2`
//! shared library.
//!
//! Included by both `benches/argon2.rs` and `benches/micro.rs` with
//! `#[path = "support/cref_isa.rs"] mod cref_isa;`, so the two report the same
//! thing. It is not a bench target of its own (Cargo only auto-discovers
//! `benches/*.rs` and `benches/*/main.rs`, and `autobenches = false` in
//! `Cargo.toml` settles it anyway).
//!
//! # Why this exists
//!
//! `benches/argon2.rs` used to print the C reference as
//! `"(src/ref.c, scalar, -O3)"` unconditionally. That string is a *guess about
//! the build*, and the guess was wrong: it printed "scalar" next to timings
//! taken against an AVX-512 `src/opt.c`, and a whole conclusion was drawn from
//! the resulting mismatched comparison. A label must never be a guess. This
//! module reads the bytes that are actually in the library.
//!
//! # How
//!
//! 1. Parse the ELF64 header far enough to find `.symtab`/`.strtab`, look up
//!    the `fill_block` symbol, and take exactly its bytes. That is the only
//!    function whose instruction mix identifies the build, and restricting the
//!    scan to it removes every false positive from BLAKE2b, `memcpy` and the
//!    encoder.
//! 2. Count three unambiguous encoding markers in those bytes:
//!    * **EVEX** — a `0x62` lead byte whose P0 has `mm != 0` and bits 3:2 == 0,
//!      and whose P1 has bit 2 set (that bit is architecturally always 1).
//!      Only AVX-512 emits this.
//!    * **VEX.256** — `0xC5` with L set in byte 1, or `0xC4` with L set in
//!      byte 2. 256-bit vector code.
//!    * **`pshufb`** — the exact 4-byte `66 0F 38 00` opcode. SSSE3.
//! 3. Report the counts **alongside** the label, so a reader can check the
//!    classification instead of trusting it.
//!
//! On a non-ELF image (macOS Mach-O) the symbol lookup is skipped and the whole
//! file is scanned, which is noisier; the returned label says so.
//!
//! The `src/opt.o` vs `src/ref.o` object file left next to the library says
//! which *source* was compiled — `opt.c` (hand-written intrinsics) or `ref.c`
//! (portable C, which this host's gcc auto-vectorises). `make` deletes both on
//! `clean` and writes exactly one, so it is reliable for the documented
//! `make clean && make OPTTARGET=...` flow.

use std::path::Path;

/// Everything the probe learned. Every field is a measurement, not a guess.
#[derive(Debug, Clone)]
pub struct CrefIsa {
    /// Path the library was loaded from.
    pub path: String,
    /// Bytes of the `fill_block` symbol, or `None` if the symbol table had no
    /// such symbol (stripped library, or Mach-O).
    pub fill_block_bytes: Option<usize>,
    /// Whether the counts below come from `fill_block` alone or the whole file.
    pub scoped_to_fill_block: bool,
    /// EVEX (AVX-512) prefixes found.
    pub evex: usize,
    /// VEX.L=1 (256-bit) prefixes found.
    pub vex256: usize,
    /// `pshufb` (SSSE3) opcodes found.
    pub pshufb: usize,
    /// `pmuludq` / `vpmuludq` opcodes found. `opt.c` emits exactly 32 (two per
    /// BLAKE2 round, sixteen rounds); an auto-vectorised `ref.c` emits many
    /// more, so this separates the two sources even at the same ISA.
    pub pmuludq: usize,
    /// `src/opt.c` or `src/ref.c`, from the object file `make` left behind.
    pub source: &'static str,
    /// Whether this really is an x86-64 image. The markers below are x86
    /// instruction encodings, so on anything else they are noise and the ISA
    /// must not be classified from them — on the arm64 Mach-O build the
    /// whole-file byte scan happily produced enough `0xC5 ..` pairs to look
    /// like it had an opinion, and it did not.
    pub is_x86_64: bool,
}

impl CrefIsa {
    /// The ISA level, from the encoding markers.
    #[must_use]
    pub fn isa(&self) -> &'static str {
        if !self.is_x86_64 {
            "n/a (not an x86-64 image)"
        } else if self.evex >= 32 {
            "avx512"
        } else if self.vex256 >= 32 {
            "avx2"
        } else if self.pshufb >= 8 {
            "ssse3"
        } else {
            "sse2"
        }
    }

    /// One line for a report header. Carries the raw counts on purpose.
    #[must_use]
    pub fn describe(&self) -> String {
        if !self.is_x86_64 {
            // Nothing below means anything off x86-64, so do not print numbers
            // that invite being read as if they did.
            return format!("{} ({}, {})", self.path, self.source, self.isa());
        }
        let scope = if self.scoped_to_fill_block {
            match self.fill_block_bytes {
                Some(n) => format!("fill_block {n} B"),
                None => "fill_block".to_string(),
            }
        } else {
            "WHOLE FILE - not scoped to fill_block, treat with suspicion".to_string()
        };
        format!(
            "{} ({}, {}, {}: evex={} vex256={} pshufb={} pmuludq={})",
            self.path,
            self.source,
            self.isa(),
            scope,
            self.evex,
            self.vex256,
            self.pshufb,
            self.pmuludq,
        )
    }
}

/// Probe `path`. Returns `None` only if the file cannot be read.
#[must_use]
pub fn probe(path: &str) -> Option<CrefIsa> {
    let bytes = std::fs::read(path).ok()?;
    let is_x86_64 = is_elf64_x86_64(&bytes);
    let (window, scoped, size) = match fill_block_range(&bytes) {
        Some((off, len)) => (&bytes[off..off + len], true, Some(len)),
        None => (&bytes[..], false, None),
    };

    Some(CrefIsa {
        path: path.to_string(),
        fill_block_bytes: size,
        scoped_to_fill_block: scoped,
        evex: count_evex(window),
        vex256: count_vex256(window),
        pshufb: count_bytes(window, &[0x66, 0x0F, 0x38, 0x00]),
        pmuludq: count_pmuludq(window),
        source: compiled_source(path),
        is_x86_64,
    })
}

/// Which of `src/opt.c` / `src/ref.c` `make` compiled, from the object file it
/// left in the tree beside the library.
fn compiled_source(lib: &str) -> &'static str {
    let dir = Path::new(lib).parent().unwrap_or(Path::new("."));
    let opt = dir.join("src/opt.o").exists();
    let refc = dir.join("src/ref.o").exists();
    match (opt, refc) {
        (true, false) => "src/opt.c",
        (false, true) => "src/ref.c",
        // Both present means a build without `make clean`; neither means the
        // objects were removed. Either way the honest answer is "don't know".
        _ => "source unknown",
    }
}

// ---------------------------------------------------------------------------
// Minimal ELF64 symbol lookup
// ---------------------------------------------------------------------------

fn u16le(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(at..at + 2)?.try_into().ok()?))
}
fn u32le(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}
fn u64le(b: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(b.get(at..at + 8)?.try_into().ok()?))
}

/// `EM_X86_64`. Anything else and the encoding counts are meaningless.
fn is_elf64_x86_64(b: &[u8]) -> bool {
    b.len() > 0x14
        && &b[..4] == b"\x7fELF"
        && b[4] == 2
        && b[5] == 1
        && u16le(b, 0x12) == Some(62)
}

/// File offset and length of the `fill_block` symbol's bytes.
///
/// `fill_block` is `static` in both `ref.c` and `opt.c`, so it appears in
/// `.symtab` (which `make` does not strip) rather than `.dynsym`.
fn fill_block_range(b: &[u8]) -> Option<(usize, usize)> {
    // ELF64 little-endian magic.
    if b.get(..4)? != b"\x7fELF" || *b.get(4)? != 2 || *b.get(5)? != 1 {
        return None;
    }
    let e_shoff = u64le(b, 0x28)? as usize;
    let e_shentsize = u16le(b, 0x3A)? as usize;
    let e_shnum = u16le(b, 0x3C)? as usize;
    if e_shoff == 0 || e_shentsize < 64 {
        return None;
    }

    const SHT_SYMTAB: u32 = 2;
    for i in 0..e_shnum {
        let sh = e_shoff + i * e_shentsize;
        if u32le(b, sh + 0x04)? != SHT_SYMTAB {
            continue;
        }
        let sym_off = u64le(b, sh + 0x18)? as usize;
        let sym_size = u64le(b, sh + 0x20)? as usize;
        let strtab_idx = u32le(b, sh + 0x28)? as usize;
        let entsize = u64le(b, sh + 0x38)? as usize;
        if entsize < 24 {
            return None;
        }

        // The linked string table holds the symbol names.
        let strh = e_shoff + strtab_idx * e_shentsize;
        let str_off = u64le(b, strh + 0x18)? as usize;
        let str_size = u64le(b, strh + 0x20)? as usize;
        let strtab = b.get(str_off..str_off + str_size)?;

        for s in 0..(sym_size / entsize) {
            let sym = sym_off + s * entsize;
            let name_off = u32le(b, sym)? as usize;
            let value = u64le(b, sym + 0x08)? as usize;
            let size = u64le(b, sym + 0x10)? as usize;
            let shndx = u16le(b, sym + 0x06)? as usize;
            if size == 0 || shndx == 0 || shndx >= e_shnum {
                continue;
            }
            let name = strtab.get(name_off..)?;
            let end = name.iter().position(|&c| c == 0)?;
            if &name[..end] != b"fill_block" {
                continue;
            }
            // Map the virtual address back to a file offset through the
            // section the symbol lives in.
            let tsh = e_shoff + shndx * e_shentsize;
            let sh_addr = u64le(b, tsh + 0x10)? as usize;
            let sh_offset = u64le(b, tsh + 0x18)? as usize;
            let file_off = value.checked_sub(sh_addr)?.checked_add(sh_offset)?;
            if file_off.checked_add(size)? <= b.len() {
                return Some((file_off, size));
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Encoding markers
// ---------------------------------------------------------------------------

fn count_bytes(hay: &[u8], needle: &[u8]) -> usize {
    if hay.len() < needle.len() {
        return 0;
    }
    hay.windows(needle.len()).filter(|w| *w == needle).count()
}

/// EVEX: `62 P0 P1 P2`. `P0` bits 3:2 are reserved-zero and `mm` (bits 1:0) is
/// never 0; `P1` bit 2 is architecturally always 1.
fn count_evex(b: &[u8]) -> usize {
    b.windows(3)
        .filter(|w| w[0] == 0x62 && (w[1] & 0x0C) == 0 && (w[1] & 0x03) != 0 && (w[2] & 0x04) != 0)
        .count()
}

/// VEX with `L = 1`, i.e. 256-bit. Two-byte form `C5 <RvvvvLpp>`, three-byte
/// form `C4 <RXBmmmmm> <WvvvvLpp>`; `L` is bit 2 of the last prefix byte.
fn count_vex256(b: &[u8]) -> usize {
    let two = b
        .windows(2)
        .filter(|w| w[0] == 0xC5 && (w[1] & 0x04) != 0)
        .count();
    let three = b
        .windows(3)
        .filter(|w| w[0] == 0xC4 && (w[2] & 0x04) != 0)
        .count();
    two + three
}

/// `pmuludq` in every encoding this can appear in: legacy `66 0F F4`, and the
/// VEX/EVEX forms whose opcode byte `F4` follows a `0F` map selector. Counting
/// the legacy form plus "`F4` as the opcode after a VEX/EVEX prefix" is what
/// the three vector widths reduce to.
fn count_pmuludq(b: &[u8]) -> usize {
    // Legacy SSE2: 66 [REX] 0F F4 /r. The REX byte is optional and sits
    // between the mandatory 66 and the escape byte, so both spellings count.
    let legacy = count_bytes(b, &[0x66, 0x0F, 0xF4])
        + b.windows(4)
            .filter(|w| w[0] == 0x66 && (w[1] & 0xF0) == 0x40 && w[2] == 0x0F && w[3] == 0xF4)
            .count();
    // VEX 2-byte: C5 <pp=01> F4 /r
    let vex2 = b
        .windows(3)
        .filter(|w| w[0] == 0xC5 && (w[1] & 0x03) == 0x01 && w[2] == 0xF4)
        .count();
    // VEX 3-byte: C4 <RXBmmmmm, mmmmm=01> <WvvvvLpp, pp=01> F4 /r
    let vex3 = b
        .windows(4)
        .filter(|w| w[0] == 0xC4 && (w[1] & 0x1F) == 0x01 && (w[2] & 0x03) == 0x01 && w[3] == 0xF4)
        .count();
    // EVEX: 62 P0 P1 P2 F4 /r, map 01, pp 01.
    let evex = b
        .windows(5)
        .filter(|w| {
            w[0] == 0x62
                && (w[1] & 0x03) == 0x01
                && (w[1] & 0x0C) == 0
                && (w[2] & 0x07) == 0x05
                && w[4] == 0xF4
        })
        .count();
    legacy + vex2 + vex3 + evex
}
