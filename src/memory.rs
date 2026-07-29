//! The block arena, the reusable [`Workspace`] that owns one across calls, and
//! secure wiping.
//!
//! [`Arena`], [`secure_wipe`] and [`clear_internal_memory`] mirror
//! `allocate_memory`, `free_memory`, `secure_wipe_memory` and
//! `clear_internal_memory` from `phc-winner-argon2/src/core.c`. [`Workspace`]
//! has no counterpart in the C, which allocates and frees the arena on every
//! `argon2_ctx` call.
//!
//! # Why a reuse layer exists, and what it is actually worth
//!
//! Measured on this host (Apple M5 Max, 16 KiB pages, `--release`), one
//! Argon2id hash breaks down like this:
//!
//! ```text
//!  m_cost  | posix_memalign | zeroing memset |  page faults  | fill + wipe
//! ---------|----------------|----------------|---------------|-------------
//!   1 MiB  |     41 ns      |    4.04 us     |       0       |   99.6%
//!  64 MiB  |    584 ns      |     273 us     |       0       |   99.8%
//!   1 GiB  |   1.71 us      |    4.33 ms     |       0       |   99.9%
//! ```
//!
//! So "reduce allocation overhead" cannot mean "make fewer allocator calls":
//! there is already exactly **one** per hash and it costs six thousandths of
//! one percent at 1 GiB. There are also no page faults to save — `alloc_zeroed`
//! with `ARENA_ALIGN == 64` goes through `posix_memalign` + an explicit
//! `memset` (std only reaches `calloc` when `align <= 16`), and macOS
//! `libmalloc` never returns the pages to the kernel, so a steady-state process
//! faults zero times per hash.
//!
//! The one real prize is the **zeroing memset**, and reuse takes it by making
//! the security wipe do double duty: a [`Workspace`] wipes the arena when it is
//! released, so the next acquisition gets a zeroed arena for free. One zeroing
//! pass per hash instead of two. Honest size of that win: **1.4-2.4% at
//! `lanes = 1`, 3.0-5.1% at `lanes = 4`**, and roughly half that again at
//! `t_cost = 3`. Worth having. Not a breakthrough, and it must never be
//! described as removing allocations or page faults, because it removes
//! neither.
//!
//! # The zeroing contract, stated exactly
//!
//! Argon2 itself does **not** need a zeroed arena. Pass 0 writes every block
//! before anything reads it (`fill_first_blocks` writes blocks 0 and 1 of each
//! lane, `fill_segment` writes the rest, and `ref.c:185-187` passes
//! `with_xor = 0` for pass 0 so `next_block` is never read). The C reference
//! allocates with plain `malloc` (`core.c:105`). What *this* crate needs is
//! weaker than "zero" and stronger than nothing: [`Arena::as_slice`] and
//! [`Arena::as_mut_slice`] are safe `pub fn`s handing out `&[Block]` over the
//! whole arena before any block is written, so the memory must be
//! **initialised**. `alloc_zeroed` is simply the cheapest way to establish that
//! on a fresh allocation.
//!
//! Once an arena has been used once it is permanently initialised, so the slice
//! API stays sound on a pooled arena even with `zeroize-memory` off.
//!
//! # Wipe on release, never on acquire
//!
//! When a hash returns, 100% of the arena still holds material derived from
//! that password, including the final block that produced the tag. Wiping on
//! *acquire* would park all of it in the workspace for the entire idle period —
//! strictly worse than today. Wiping on *release* preserves today's property
//! exactly: the exposure window closes when the call returns.

#[cfg(feature = "bump-alloc")]
use bumpalo::Bump;

use alloc::alloc::{Layout, alloc_zeroed, dealloc};
use core::mem::ManuallyDrop;
use core::ops::{Deref, DerefMut};
use core::ptr::NonNull;
use core::sync::atomic::{Ordering, compiler_fence};

use crate::block::Block;
use crate::error::Error;

/// Alignment of the arena, in bytes. 64 keeps AVX-512 loads on a cache line.
pub const ARENA_ALIGN: usize = 64;

/// `true` when the `zeroize-memory` feature is on, as a `const` the reuse layer
/// can branch on.
///
/// [`clear_internal_memory_blocks`] compiles to nothing without the feature, so
/// the workspace has to know whether a "wipe" actually zeroed anything before
/// it may claim the arena is zero.
const WIPE_ENABLED: bool = cfg!(feature = "zeroize-memory");

// ---------------------------------------------------------------------------
// Secure wiping
// ---------------------------------------------------------------------------

/// `secure_wipe_memory()`: zero `bytes` in a way the optimiser may not remove.
///
/// Uses [`core::ptr::write_volatile`] per byte plus a `SeqCst`
/// [`compiler_fence`], instead of trusting a plain `memset` to survive dead
/// store elimination.
pub fn secure_wipe(bytes: &mut [u8]) {
    for byte in bytes.iter_mut() {
        // SAFETY: `byte` comes from a mutable slice iterator, so it is a valid,
        // aligned, uniquely-borrowed `*mut u8`.
        unsafe { core::ptr::write_volatile(byte, 0) };
    }
    compiler_fence(Ordering::SeqCst);
}

/// [`secure_wipe`] over 64-bit words — 8x fewer volatile stores.
pub fn secure_wipe_u64(words: &mut [u64]) {
    for word in words.iter_mut() {
        // SAFETY: as in `secure_wipe`, for `u64` instead of `u8`.
        unsafe { core::ptr::write_volatile(word, 0) };
    }
    compiler_fence(Ordering::SeqCst);
}

/// [`secure_wipe`] over whole blocks.
pub fn secure_wipe_blocks(blocks: &mut [Block]) {
    for block in blocks.iter_mut() {
        secure_wipe_u64(&mut block.0);
    }
}

/// [`secure_wipe`] over a raw region of arbitrary alignment.
///
/// Wipes the 8-byte-aligned interior with `u64` stores and only the ragged
/// ends byte by byte. That matters more than it looks: a pure per-byte
/// volatile wipe of the ~180 bytes a hash puts through the bump measures
/// 46.7 ns on this host, which is *more* than the ~26 ns the bump saves over
/// `Vec` in the first place. Word stores bring it under the win.
///
/// Only the bump path needs this — [`Arena`] is `Block`-aligned by
/// construction, so [`secure_wipe_blocks`] can go straight to `u64`.
///
/// # Safety
///
/// `ptr` must be valid for writes of `len` bytes, and nothing else may access
/// that region for the duration.
pub unsafe fn secure_wipe_raw(ptr: *mut u8, len: usize) {
    // Bytes before the first 8-byte boundary. `align_offset` returns
    // `usize::MAX` if alignment is impossible, which cannot happen for
    // `u8 -> u64` on any real target — treat it as "wipe it all bytewise".
    let misalign = ptr.align_offset(align_of::<u64>());
    let head = if misalign == usize::MAX {
        len
    } else {
        misalign.min(len)
    };
    let words = (len - head) / size_of::<u64>();
    let tail = head + words * size_of::<u64>();

    // SAFETY: `head <= tail <= len` and the caller guarantees `ptr` is valid
    // for `len` bytes, so every offset below is in bounds. The `u64` stores run
    // from `ptr.add(head)`, which `align_offset` just made 8-byte aligned, and
    // there are exactly `words` of them, so they stay within `tail <= len`.
    unsafe {
        for i in 0..head {
            core::ptr::write_volatile(ptr.add(i), 0u8);
        }
        let body = ptr.add(head).cast::<u64>();
        for i in 0..words {
            core::ptr::write_volatile(body.add(i), 0u64);
        }
        for i in tail..len {
            core::ptr::write_volatile(ptr.add(i), 0u8);
        }
    }
    compiler_fence(Ordering::SeqCst);
}

/// `clear_internal_memory()`: wipe only if wiping is enabled.
///
/// The `zeroize-memory` feature plays the role of `FLAG_clear_internal_memory`,
/// which defaults to 1 in `core.c`; the feature is on by default here too.
#[inline]
pub fn clear_internal_memory(bytes: &mut [u8]) {
    #[cfg(feature = "zeroize-memory")]
    secure_wipe(bytes);
    #[cfg(not(feature = "zeroize-memory"))]
    let _ = bytes;
}

/// [`clear_internal_memory`] over 64-bit words.
#[inline]
pub fn clear_internal_memory_u64(words: &mut [u64]) {
    #[cfg(feature = "zeroize-memory")]
    secure_wipe_u64(words);
    #[cfg(not(feature = "zeroize-memory"))]
    let _ = words;
}

/// [`clear_internal_memory`] over whole blocks.
#[inline]
pub fn clear_internal_memory_blocks(blocks: &mut [Block]) {
    #[cfg(feature = "zeroize-memory")]
    secure_wipe_blocks(blocks);
    #[cfg(not(feature = "zeroize-memory"))]
    let _ = blocks;
}

// ---------------------------------------------------------------------------
// Arena
// ---------------------------------------------------------------------------

/// A 64-byte-aligned array of initialised [`Block`]s.
///
/// [`Arena::new`] allocates with [`alloc_zeroed`], so a fresh arena is zeroed.
/// Do **not** replace that with `vec![Block::ZERO; n]`, which memsets one
/// kilobyte at a time.
///
/// [`Drop`] wipes the arena (subject to the `zeroize-memory` feature) and then
/// frees it, matching `free_memory()`.
///
/// # Visible length vs capacity
///
/// [`len`](Arena::len) is the *visible* length — the number of blocks
/// [`as_slice`](Arena::as_slice) and [`as_mut_slice`](Arena::as_mut_slice)
/// expose, and the only region a borrower can reach.
/// [`capacity`](Arena::capacity) is the number of blocks actually allocated.
/// [`Arena::new`] makes them equal; only [`Workspace`] ever makes the capacity
/// larger, when it hands the same allocation to a smaller hash.
///
/// # Invariants
///
/// 1. `len() <= capacity()`, and all `capacity()` blocks are allocated and
///    **initialised** — never `MaybeUninit`. This is what makes the safe slice
///    accessors sound.
/// 2. **With `zeroize-memory` on**, blocks `[len(), capacity())` are all zero.
///    Only [`Workspace`] can open that gap, and the induction is: a fresh
///    allocation is zero throughout; a borrower can only ever reach
///    `[0, len())`; and release wipes exactly `[0, len())`, restoring "all of
///    `capacity()` is zero". This is why [`Drop`] and [`Workspace::release`]
///    wipe only `[0, len())` instead of the whole capacity — an oversized
///    workspace must not make a small hash pay for blocks it never touched.
///
///    Without the feature this does **not** hold, and nothing relies on it:
///    every wipe compiles to nothing, so there is no wipe left to skip.
///    [`ensure_zeroed`](Arena::ensure_zeroed) deliberately covers the whole
///    capacity rather than leaning on this.
/// 3. [`is_known_zeroed`](Arena::is_known_zeroed) is conservative and holds in
///    **every** feature configuration: `true` implies all `capacity()` blocks
///    are zero; `false` implies nothing. This is the invariant that decides
///    whether [`Drop`] may skip its wipe, so it is the security-critical one.
pub struct Arena {
    ptr: NonNull<Block>,
    /// Visible length. See the type-level docs.
    blocks: usize,
    /// Allocated length. Always `>= blocks`.
    capacity: usize,
    /// `true` only when blocks `[0, capacity)` are known to be all zero.
    ///
    /// Set by [`alloc_zeroed`] at birth, cleared by every accessor that could
    /// let a caller write, and re-established by an actual wipe. Being wrong in
    /// the `true` direction would skip a security wipe, so every mutable
    /// accessor clears it unconditionally rather than trying to be clever.
    zeroed: bool,
}

// SAFETY: `Arena` uniquely owns a heap allocation with no thread affinity, and
// exposes no shared-mutability API, so moving it between threads is sound.
// Deliberately not `Sync`: concurrent lane writes go through raw pointers and
// need `core`'s own safety argument.
unsafe impl Send for Arena {}

impl Arena {
    /// Allocate `blocks` zeroed blocks. `capacity() == len() == blocks`.
    ///
    /// # Errors
    ///
    /// [`Error::MemoryAllocationError`] if `blocks` is 0, if
    /// `blocks * size_of::<Block>()` overflows, if the layout is invalid, or if
    /// the allocator returns null. The C reference reports the same code for
    /// all of these (`ARGON2_MEMORY_ALLOCATION_ERROR`).
    pub fn new(blocks: usize) -> Result<Arena, Error> {
        // `argon2_ctx` never asks for zero blocks (memory_blocks >= MIN_MEMORY),
        // but `alloc_zeroed` with a zero-sized layout is undefined behaviour,
        // so reject it here.
        if blocks == 0 {
            return Err(Error::MemoryAllocationError);
        }

        // Mirrors the multiplication-overflow check in `allocate_memory()`.
        let bytes = blocks
            .checked_mul(size_of::<Block>())
            .ok_or(Error::MemoryAllocationError)?;

        let layout = Layout::from_size_align(bytes, ARENA_ALIGN)
            .map_err(|_| Error::MemoryAllocationError)?;

        // SAFETY: `layout` has a non-zero size (blocks >= 1 and
        // size_of::<Block>() == 1024).
        let raw = unsafe { alloc_zeroed(layout) };

        let ptr = NonNull::new(raw.cast::<Block>()).ok_or(Error::MemoryAllocationError)?;

        Ok(Arena {
            ptr,
            blocks,
            capacity: blocks,
            // `alloc_zeroed` established invariant 3.
            zeroed: true,
        })
    }

    /// Number of blocks a borrower can see. See the type-level docs.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.blocks
    }

    /// Always `false`: [`Arena::new`] rejects a zero-block request and
    /// [`Workspace::acquire`] does too.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.blocks == 0
    }

    /// Number of blocks actually allocated, `>= len()`.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Whether all [`capacity`](Arena::capacity) blocks are known to be zero.
    ///
    /// Conservative in the safe direction: `true` guarantees the arena is
    /// zero, `false` guarantees nothing. Handing out any mutable access clears
    /// it, so this reads `false` for the whole life of a hash.
    #[inline]
    #[must_use]
    pub fn is_known_zeroed(&self) -> bool {
        self.zeroed
    }

    /// Pointer to block 0.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const Block {
        self.ptr.as_ptr()
    }

    /// Mutable pointer to block 0. This is what [`crate::block::Instance`] holds.
    ///
    /// Clears [`is_known_zeroed`](Arena::is_known_zeroed): whatever the caller
    /// does with the pointer, the arena must be treated as dirty afterwards.
    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut Block {
        self.zeroed = false;
        self.ptr.as_ptr()
    }

    /// The visible arena as a slice.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[Block] {
        // SAFETY: invariant 1 — `ptr` is a live allocation of `capacity >=
        // blocks` initialised `Block`s — and `&self` guarantees no concurrent
        // mutation.
        unsafe { core::slice::from_raw_parts(self.ptr.as_ptr(), self.blocks) }
    }

    /// The visible arena as a mutable slice. This is the Miri-checkable access
    /// path; the single-threaded fill loop should prefer it over raw pointers.
    ///
    /// Clears [`is_known_zeroed`](Arena::is_known_zeroed).
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [Block] {
        self.zeroed = false;
        self.visible_mut()
    }

    /// Zero the arena unless it is already known to be zero.
    ///
    /// A plain `memset`, **not** a secure wipe: this establishes zero for a new
    /// borrower, it does not destroy a previous borrower's secrets. That job
    /// belongs to [`Workspace::release`], which runs first.
    ///
    /// Free (a predictable branch, no stores) in the default build, where
    /// release already wiped. With `zeroize-memory` off it costs one `memset`,
    /// which is the price of asking for a guarantee the feature was turned off
    /// to avoid paying for.
    ///
    /// Covers the whole [`capacity`](Arena::capacity), not just the visible
    /// window, because it is what establishes
    /// [`is_known_zeroed`](Arena::is_known_zeroed) and that flag is a claim
    /// about the whole capacity. Zeroing only `[0, len())` here would be a
    /// silent lie the moment a later, wider acquisition exposed the tail: with
    /// `zeroize-memory` off, invariant 2 does not hold on its own, so this
    /// cannot lean on it.
    pub fn ensure_zeroed(&mut self) {
        if self.zeroed {
            return;
        }
        // SAFETY: invariant 1 — `ptr` is valid for writes of `capacity`
        // `Block`s — and `&mut self` makes the write exclusive. `Block` is
        // `Copy` with no padding-sensitive invariant, so all-zero is a valid
        // bit pattern (`Block::ZERO`).
        unsafe { core::ptr::write_bytes(self.ptr.as_ptr(), 0, self.capacity) };
        self.zeroed = true;
    }

    /// The visible arena as a mutable slice, *without* touching the
    /// known-zeroed flag.
    ///
    /// For internal paths that write only zeros ([`Drop`], the release wipe),
    /// which must not falsely mark the arena dirty.
    #[inline]
    fn visible_mut(&mut self) -> &mut [Block] {
        // SAFETY: as in `as_slice`, and `&mut self` guarantees exclusivity.
        unsafe { core::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.blocks) }
    }

    /// Secure-wipe the visible blocks and re-establish invariant 3.
    ///
    /// Only `[0, len())` is wiped: `[len(), capacity())` is already zero by
    /// invariant 2, and it is the borrower's window that holds the secrets.
    ///
    /// The flag is only claimed when the feature actually did something —
    /// [`clear_internal_memory_blocks`] compiles to nothing without
    /// `zeroize-memory`, and claiming `zeroed` after a no-op would let a later
    /// [`Drop`] skip a real wipe.
    fn wipe_visible(&mut self) {
        clear_internal_memory_blocks(self.visible_mut());
        self.zeroed |= WIPE_ENABLED;
    }

    /// Shrink or restore the visible length within the existing allocation.
    ///
    /// Returns `false` and changes nothing when `blocks > capacity`. Private:
    /// invariant 2 only survives because [`Workspace`] is the sole caller and
    /// only ever retargets onto a region it knows to be zero.
    fn retarget(&mut self, blocks: usize) -> bool {
        if blocks > self.capacity {
            return false;
        }
        self.blocks = blocks;
        true
    }

    /// The layout the arena was allocated with, needed to free it.
    ///
    /// Sized from `capacity`, not the visible `blocks` — freeing with the wrong
    /// layout would be undefined behaviour.
    #[inline]
    fn layout(&self) -> Layout {
        // `Arena::new` already validated this exact size/align pair, and
        // `capacity` has not changed since.
        match Layout::from_size_align(self.capacity * size_of::<Block>(), ARENA_ALIGN) {
            Ok(layout) => layout,
            // Unreachable: `new` succeeded with the same arguments.
            Err(_) => Layout::new::<Block>(),
        }
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        // Skip the wipe only when the arena is *known* zero — a workspace
        // release, or an arena that was never handed out. Everything else pays
        // for it, exactly as before the reuse layer existed.
        if !self.zeroed {
            clear_internal_memory_blocks(self.visible_mut());
        }
        let layout = self.layout();
        // SAFETY: `ptr` came from `alloc_zeroed` with this exact layout
        // (`capacity` blocks, `ARENA_ALIGN`) and has not been freed yet
        // (`Drop` runs once).
        unsafe { dealloc(self.ptr.as_ptr().cast::<u8>(), layout) };
    }
}

impl core::fmt::Debug for Arena {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Arena")
            .field("ptr", &self.ptr.as_ptr())
            .field("blocks", &self.blocks)
            .field("capacity", &self.capacity)
            .field("zeroed", &self.zeroed)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

/// Scratch memory one hashing worker reuses across many hashes.
///
/// Holds at most one parked [`Arena`], plus — with the non-default
/// `bump-alloc` feature — one reusable `bumpalo::Bump` for small, short-lived buffers.
/// A server hashing passwords at fixed parameters keeps one `Workspace` per
/// worker thread and pays the allocator, and the zeroing memset, once instead
/// of once per request.
///
/// ```
/// use argon2_rust::__internal::Workspace;
///
/// let mut ws = Workspace::new();
/// let first = ws.acquire(64).expect("64 blocks").as_ptr();
/// let second = ws.acquire(64).expect("reused").as_ptr();
/// assert_eq!(first, second); // same allocation, no allocator traffic
/// ```
///
/// It is [`Send`], so a worker can carry one:
///
/// ```
/// # use argon2_rust::__internal::Workspace;
/// fn needs_send<T: Send>(_: T) {}
/// needs_send(Workspace::new());
/// ```
///
/// It is **not** [`Sync`], and this must not compile — see *Threading* below:
///
/// ```compile_fail
/// # use argon2_rust::__internal::Workspace;
/// fn needs_sync<T: Sync>(_: &T) {}
/// needs_sync(&Workspace::new());
/// ```
///
/// # What reuse buys, and what it does not
///
/// It removes **one memset per hash** — 1.4-2.4% at `lanes = 1`, 3.0-5.1% at
/// `lanes = 4`. It does **not** remove allocator calls (there was only ever
/// one, at 0.0006% of a 1 GiB hash), it does **not** remove page faults (a
/// steady-state process has none), and it does **not** remove the security
/// wipe (a reused arena is still wiped, just at release instead of at free).
///
/// # Zeroing guarantee
///
/// With `zeroize-memory` on — the default — [`acquire`](Workspace::acquire)
/// **always** returns an all-zero arena, because release wiped it. With the
/// feature off, an arena that has been used before comes back holding the
/// previous hash's derived material; it is still initialised, so every safe
/// accessor stays sound, and Argon2 overwrites all of it in pass 0 before
/// reading any of it. Callers that need zero regardless should call
/// [`Arena::ensure_zeroed`], which is a no-op in the default build.
///
/// # Threading
///
/// `Workspace` is [`Send`] but not [`Sync`], so it can move between threads but
/// not be shared. That is deliberate and it is what keeps the design compatible
/// with the parallel fill: `bumpalo::Bump` is `!Sync`, and the multi-lane fill shares
/// the *arena* across [`std::thread::scope`] workers by raw pointer while the
/// bump is never touched at all. One workspace per worker, never one workspace
/// across workers.
pub struct Workspace {
    /// The parked arena. `None` before the first acquisition, while one is on
    /// loan, and after [`Workspace::clear`].
    ///
    /// While parked its blocks `[0, capacity)` are all zero, provided
    /// `zeroize-memory` is on.
    arena: Option<Arena>,

    /// A bump allocator for buffers that die inside one call.
    ///
    /// Deliberately *reused*, never constructed per call. See
    /// [`Workspace::bump`] for the measurement; the short version is that a
    /// fresh `Bump::new()` costs 18.5 ns before it serves anything, which is
    /// more than the entire `Vec` it would replace.
    #[cfg(feature = "bump-alloc")]
    bump: Bump,
}

impl Workspace {
    /// An empty workspace. Allocates nothing at all.
    ///
    /// The first [`acquire`](Workspace::acquire) allocates the arena; use
    /// [`Workspace::with_capacity`] to front-load that instead.
    ///
    /// Cannot panic even with `bump-alloc` on: `Bump::new()` is
    /// `try_with_capacity(0)`, which returns `Ok` before it touches the
    /// allocator (bumpalo 3.20.3, `src/lib.rs:810-825`), so the `oom()` arm is
    /// unreachable.
    #[must_use]
    pub fn new() -> Workspace {
        Workspace {
            arena: None,
            #[cfg(feature = "bump-alloc")]
            bump: Bump::new(),
        }
    }

    /// A workspace with room for `blocks` blocks already allocated.
    ///
    /// # Errors
    ///
    /// Whatever [`Arena::new`] returns. `blocks == 0` is *not* an error here —
    /// it just yields an empty workspace, since reserving nothing is a no-op.
    pub fn with_capacity(blocks: usize) -> Result<Workspace, Error> {
        let mut workspace = Workspace::new();
        workspace.reserve(blocks)?;
        Ok(workspace)
    }

    /// Blocks the parked arena can hold, or 0 when nothing is parked.
    ///
    /// Reads 0 while an arena is on loan — the capacity travels with the
    /// [`ArenaGuard`].
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.arena.as_ref().map_or(0, Arena::capacity)
    }

    /// Make sure the parked arena can hold `blocks` blocks, allocating if not.
    ///
    /// Growing **frees the old arena before allocating the new one**, so peak
    /// resident memory stays at one arena rather than two — at `m_cost = 1 GiB`
    /// the difference is 1 GiB of RSS. The price is that a failed growth leaves
    /// the workspace empty (the old arena is gone) and returns `Err`; the
    /// workspace stays perfectly usable, the next acquisition just allocates
    /// fresh. Size the workspace once with [`Workspace::with_capacity`] and the
    /// question never arises.
    ///
    /// # Errors
    ///
    /// Whatever [`Arena::new`] returns.
    pub fn reserve(&mut self, blocks: usize) -> Result<(), Error> {
        if blocks == 0 || self.capacity() >= blocks {
            return Ok(());
        }
        // Free first, then allocate. `Drop` wipes if the arena is not already
        // known zero, so this cannot leak the previous tenant.
        self.arena = None;
        self.arena = Some(Arena::new(blocks)?);
        Ok(())
    }

    /// Borrow an arena of `blocks` blocks, returning it to the workspace on
    /// drop.
    ///
    /// No allocator traffic when the parked arena is already big enough, which
    /// is the whole point. See [`Workspace::reserve`] for what growth costs.
    ///
    /// The guard borrows the workspace, so the bump allocator is unreachable while an
    /// arena is out. That is not a limitation in practice: hashing finishes
    /// before encoding starts. If you need both at once, use
    /// [`Workspace::acquire_owned`] plus [`Workspace::release`].
    ///
    /// # Errors
    ///
    /// [`Error::MemoryAllocationError`] if `blocks` is 0 or the arena could not
    /// be allocated.
    pub fn acquire(&mut self, blocks: usize) -> Result<ArenaGuard<'_>, Error> {
        let arena = self.acquire_owned(blocks)?;
        Ok(ArenaGuard {
            arena: ManuallyDrop::new(arena),
            workspace: self,
        })
    }

    /// [`acquire`](Workspace::acquire) without the guard: the caller owns the
    /// [`Arena`] and must hand it back with [`Workspace::release`].
    ///
    /// Forgetting to release is safe and secure — the arena's own [`Drop`]
    /// wipes and frees it — it just forfeits the reuse. Prefer
    /// [`Workspace::acquire`], which cannot forget, including on the `?` early
    /// returns that a hash is full of.
    ///
    /// # Errors
    ///
    /// [`Error::MemoryAllocationError`] if `blocks` is 0 or the arena could not
    /// be allocated.
    pub fn acquire_owned(&mut self, blocks: usize) -> Result<Arena, Error> {
        if blocks == 0 {
            return Err(Error::MemoryAllocationError);
        }
        match self.arena.take() {
            // Big enough: retarget the visible window and hand it over. No
            // allocator call, and no zeroing — invariant 2 says everything
            // outside the new window is zero, and the release wipe left
            // everything inside it zero too.
            Some(mut arena) if arena.capacity() >= blocks => {
                let fits = arena.retarget(blocks);
                debug_assert!(fits, "capacity was just checked");
                Ok(arena)
            }
            // Too small: free before allocating, as documented on `reserve`.
            Some(arena) => {
                drop(arena);
                Arena::new(blocks)
            }
            None => Arena::new(blocks),
        }
    }

    /// Wipe `arena` and park it for the next acquisition.
    ///
    /// The wipe is [`clear_internal_memory_blocks`] — `write_volatile` per
    /// `u64` plus a `SeqCst` [`compiler_fence`], gated on `zeroize-memory`
    /// exactly as [`Arena`]'s own [`Drop`] is. It covers the visible blocks,
    /// which is precisely the window the borrower could reach.
    ///
    /// If an arena is already parked, the **larger** of the two is kept and the
    /// other is freed, so a workspace never shrinks under a mixed workload.
    pub fn release(&mut self, mut arena: Arena) {
        arena.wipe_visible();

        let keep_parked = self
            .arena
            .as_ref()
            .is_some_and(|parked| parked.capacity() >= arena.capacity());

        if !keep_parked {
            // Drops (wipes if needed, frees) whatever was parked before.
            self.arena = Some(arena);
        }
        // Otherwise `arena` drops here. It was just wiped, so its `Drop` skips
        // straight to `dealloc`.
    }

    /// Return every byte this workspace holds, back to [`Workspace::new`].
    ///
    /// The arena is wiped on the way out unless it is already known to be zero,
    /// and the bump's scratch is wiped before its chunks are freed. Unlike
    /// [`reset_bump`](Workspace::reset_bump), which deliberately keeps its
    /// largest chunk so the next hash can reuse it, this keeps nothing — it is
    /// for a worker going idle, not for the per-hash path.
    pub fn clear(&mut self) {
        self.arena = None;
        #[cfg(feature = "bump-alloc")]
        {
            // Wipe first, then drop the chunks by replacing the whole `Bump`.
            self.reset_bump();
            self.bump = Bump::new();
        }
    }

    /// The reusable bump allocator, for buffers that do not outlive the call.
    ///
    /// # Use `try_alloc_*`, never `alloc_*`
    ///
    /// bumpalo's infallible `alloc_*` methods abort the process on allocation
    /// failure. This crate guarantees that nothing reachable from its safe
    /// public API panics, so only the `try_alloc_*` family is admissible here.
    ///
    /// # Is it worth it? Measured, and barely
    ///
    /// One hash's worth of small scratch is three buffers — 98 B to encode,
    /// 51 B + 33 B to decode — allocated, used and dropped inside the call.
    /// Marginal cost of all three on this host (interleaved A/B, min of 400
    /// rounds x 4096 iterations, empty-loop control subtracted):
    ///
    /// ```text
    ///   3x Vec::try_reserve + resize            24.30 ns
    ///   3x this bump + reset_bump (wiped)        7.34 ns    -17.0 ns
    ///   3x bumpalo + bare reset (no wipe)        3.43 ns    -20.9 ns
    ///   3x on a FRESH Bump::new per call        16.07 ns     -8.2 ns
    /// ```
    ///
    /// Two things to read off that table. First, the win is real but tiny:
    /// 17 ns against a hash that starts at 10.75 us and is normally 12.7 ms —
    /// 0.16% of the smallest Argon2 hash that exists, 0.00013% of an RFC 9106
    /// one, four orders of magnitude below this machine's run-to-run noise.
    /// **Do not restructure anything for it.**
    ///
    /// Second, the wipe in [`reset_bump`](Workspace::reset_bump) costs 3.9 ns
    /// of the 20.9 available, and only because it uses word-wide stores; a
    /// naive per-byte volatile wipe measured 46.7 ns and turned the whole thing
    /// into a net *loss* against `Vec`. That is why
    /// [`secure_wipe_raw`] exists.
    ///
    /// A fresh `Bump::new()` per call still beats three `Vec`s here, because
    /// its one chunk allocation amortises over three buffers — but it loses to
    /// a single `Vec` (18.5 ns against 9.0 ns) and to this reused bump by 2.2x.
    /// Reuse is the only shape worth shipping.
    #[cfg(feature = "bump-alloc")]
    #[inline]
    #[must_use]
    pub fn bump(&self) -> &Bump {
        &self.bump
    }

    /// Bytes of chunk memory the bump is holding on to, footers excluded.
    ///
    /// This is `Bump::allocated_bytes`, whose name is easy to misread: it is
    /// *reserved* capacity, not bytes currently handed out.
    /// [`reset_bump`](Workspace::reset_bump) leaves it unchanged — retaining
    /// the chunk is the entire point — while [`clear`](Workspace::clear) puts
    /// it back to zero. Diagnostic only.
    #[cfg(feature = "bump-alloc")]
    #[inline]
    #[must_use]
    pub fn bump_reserved_bytes(&self) -> usize {
        self.bump.allocated_bytes()
    }

    /// Reclaim every bump allocation, keeping the largest chunk for next time.
    ///
    /// Call this once per hash, not once per buffer. With `zeroize-memory` on,
    /// the used region of every chunk is securely wiped first, so bump scratch
    /// gets the same treatment as the arena.
    #[cfg(feature = "bump-alloc")]
    pub fn reset_bump(&mut self) {
        #[cfg(feature = "zeroize-memory")]
        {
            // SAFETY: `iter_allocated_chunks_raw` requires that no allocation
            // happens while the iterator is live and that no mutable reference
            // into previously allocated data exists. `&mut self` gives both:
            // every reference bumpalo hands out borrows the `Bump`, so an
            // exclusive borrow of the workspace proves none is live, and the
            // loop body only writes through the raw pointer — it never reads
            // the chunk, never forms a reference into it, and never allocates.
            // Each `(ptr, len)` covers exactly the `len` handed-out, therefore
            // initialised, bytes of that chunk — bumpalo bumps downward, so the
            // pair runs from the current bump finger up to the chunk footer.
            unsafe {
                for (chunk, len) in self.bump.iter_allocated_chunks_raw() {
                    secure_wipe_raw(chunk, len);
                }
            }
        }
        self.bump.reset();
    }
}

impl Default for Workspace {
    fn default() -> Workspace {
        Workspace::new()
    }
}

impl core::fmt::Debug for Workspace {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut out = f.debug_struct("Workspace");
        out.field("arena", &self.arena);
        #[cfg(feature = "bump-alloc")]
        out.field("bump_reserved_bytes", &self.bump.allocated_bytes());
        out.finish()
    }
}

// ---------------------------------------------------------------------------
// ArenaGuard
// ---------------------------------------------------------------------------

/// An [`Arena`] on loan from a [`Workspace`], returned when the guard drops.
///
/// Derefs to [`Arena`], so it is a drop-in for one: `guard.as_mut_slice()`,
/// `guard.as_mut_ptr()` and `guard.len()` all work.
///
/// Release happens in [`Drop`], so it also happens on the `?` early returns and
/// on unwind. The only way to skip it is [`core::mem::forget`], which leaks the
/// allocation without wiping it — the same caveat that has always applied to
/// forgetting an [`Arena`].
pub struct ArenaGuard<'w> {
    /// `ManuallyDrop` because [`Drop`] moves the arena out to release it,
    /// rather than letting `Arena::drop` free it.
    arena: ManuallyDrop<Arena>,
    workspace: &'w mut Workspace,
}

impl Deref for ArenaGuard<'_> {
    type Target = Arena;

    #[inline]
    fn deref(&self) -> &Arena {
        &self.arena
    }
}

impl DerefMut for ArenaGuard<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Arena {
        &mut self.arena
    }
}

impl Drop for ArenaGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: `Drop::drop` runs at most once, and `self.arena` is never
        // read again afterwards — the guard itself is being destroyed and the
        // field is private, so no other code can observe the moved-out
        // `ManuallyDrop`.
        let arena = unsafe { ManuallyDrop::take(&mut self.arena) };
        self.workspace.release(arena);
    }
}

impl core::fmt::Debug for ArenaGuard<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ArenaGuard")
            .field("arena", &*self.arena)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Only the bump wipe test needs an owned copy of the scratch bytes.
    #[cfg(all(feature = "bump-alloc", feature = "zeroize-memory"))]
    use alloc::vec::Vec;

    /// All `capacity` blocks, including the ones outside the visible window.
    ///
    /// Only invariant 2 makes this meaningful, and only the wipe tests need it.
    #[cfg(feature = "zeroize-memory")]
    fn whole_capacity(arena: &Arena) -> &[Block] {
        // SAFETY: invariant 1 — `capacity` initialised blocks are live — and
        // `&Arena` rules out concurrent mutation.
        unsafe { core::slice::from_raw_parts(arena.as_ptr(), arena.capacity()) }
    }

    fn is_all_zero(blocks: &[Block]) -> bool {
        blocks.iter().all(|b| *b == Block::ZERO)
    }

    // -----------------------------------------------------------------------
    // Arena — unchanged behaviour
    // -----------------------------------------------------------------------

    #[test]
    fn arena_is_zeroed_and_aligned() {
        let arena = Arena::new(16).expect("16 blocks");
        assert_eq!(arena.len(), 16);
        assert_eq!(arena.capacity(), 16);
        assert_eq!(arena.as_ptr() as usize % ARENA_ALIGN, 0);
        assert!(arena.as_slice().iter().all(|b| *b == Block::ZERO));
    }

    #[test]
    fn arena_rejects_zero_and_overflow() {
        assert!(matches!(
            Arena::new(0).err(),
            Some(Error::MemoryAllocationError)
        ));
        assert!(matches!(
            Arena::new(usize::MAX / 512).err(),
            Some(Error::MemoryAllocationError)
        ));
    }

    #[test]
    fn arena_is_writable_through_the_slice() {
        let mut arena = Arena::new(4).expect("4 blocks");
        arena.as_mut_slice()[3].fill(0xCD);
        assert_eq!(arena.as_slice()[3].0[0], u64::from_ne_bytes([0xCD; 8]));
        assert_eq!(arena.as_slice()[2], Block::ZERO);
    }

    #[test]
    fn wipe_zeroes_everything() {
        let mut bytes = [0xAAu8; 64];
        secure_wipe(&mut bytes);
        assert_eq!(bytes, [0u8; 64]);

        let mut words = [0xDEAD_BEEFu64; 8];
        secure_wipe_u64(&mut words);
        assert_eq!(words, [0u64; 8]);

        let mut blocks = [Block::ZERO; 2];
        blocks[1].fill(0xFF);
        secure_wipe_blocks(&mut blocks);
        assert_eq!(blocks[1], Block::ZERO);
    }

    /// `secure_wipe_raw` splits into a bytewise head, a `u64` body and a
    /// bytewise tail, so it has three off-by-one opportunities. Sweep every
    /// start alignment and length, and check the *neighbours* too — an
    /// over-wipe is as much a bug as an under-wipe, and only the neighbour
    /// check can catch it.
    #[test]
    fn secure_wipe_raw_covers_exactly_the_requested_region() {
        const N: usize = 64;
        for start in 0..16usize {
            for len in 0..=(N - 16) {
                let mut buf = [0xA5u8; N];
                // SAFETY: `start + len <= 16 + 48 == N`, so the region lies
                // inside `buf`, and `&mut buf` makes the write exclusive.
                unsafe { secure_wipe_raw(buf.as_mut_ptr().add(start), len) };

                for (i, byte) in buf.iter().enumerate() {
                    let inside = i >= start && i < start + len;
                    assert_eq!(
                        *byte == 0,
                        inside,
                        "byte {i} wrong for start={start} len={len}: {byte:#04x}"
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // The known-zeroed flag — load-bearing for the security wipe
    // -----------------------------------------------------------------------

    #[test]
    fn every_mutable_accessor_marks_the_arena_dirty() {
        let mut arena = Arena::new(2).expect("2 blocks");
        assert!(arena.is_known_zeroed(), "alloc_zeroed establishes it");

        let _ = arena.as_slice();
        let _ = arena.as_ptr();
        assert!(arena.is_known_zeroed(), "shared access cannot write");

        let _ = arena.as_mut_slice();
        assert!(!arena.is_known_zeroed());

        arena.ensure_zeroed();
        assert!(arena.is_known_zeroed());

        let _ = arena.as_mut_ptr();
        assert!(!arena.is_known_zeroed(), "a raw *mut is a write capability");
    }

    #[test]
    fn ensure_zeroed_clears_actual_bytes() {
        let mut arena = Arena::new(3).expect("3 blocks");
        for block in arena.as_mut_slice() {
            block.fill(0xA5);
        }
        assert!(!is_all_zero(arena.as_slice()));

        arena.ensure_zeroed();
        assert!(is_all_zero(arena.as_slice()));
        assert!(arena.is_known_zeroed());
    }

    // -----------------------------------------------------------------------
    // Workspace — reuse
    // -----------------------------------------------------------------------

    #[test]
    fn empty_workspace_allocates_nothing() {
        let ws = Workspace::new();
        assert_eq!(ws.capacity(), 0);
        assert_eq!(Workspace::default().capacity(), 0);
    }

    #[test]
    fn reuse_touches_the_allocator_exactly_once() {
        let mut ws = Workspace::with_capacity(32).expect("32 blocks");
        assert_eq!(ws.capacity(), 32);

        let first = {
            let guard = ws.acquire(32).expect("acquire");
            assert_eq!(guard.len(), 32);
            guard.as_ptr()
        };
        // Ten more round trips must all land on that same allocation.
        for _ in 0..10 {
            let guard = ws.acquire(32).expect("reacquire");
            assert_eq!(guard.as_ptr(), first, "reuse must not reallocate");
        }
        assert_eq!(ws.capacity(), 32, "capacity survives the round trips");
    }

    #[test]
    fn reuse_after_a_smaller_request_keeps_the_big_allocation() {
        let mut ws = Workspace::with_capacity(64).expect("64 blocks");
        let big = ws.acquire(64).expect("acquire 64").as_ptr();

        {
            let guard = ws.acquire(8).expect("acquire 8");
            assert_eq!(guard.len(), 8, "visible window shrinks");
            assert_eq!(guard.capacity(), 64, "allocation does not");
            assert_eq!(guard.as_ptr(), big, "and it is the same allocation");
        }

        // And the full-size window comes back without reallocating.
        let guard = ws.acquire(64).expect("acquire 64 again");
        assert_eq!(guard.len(), 64);
        assert_eq!(guard.as_ptr(), big);
    }

    #[test]
    fn reuse_after_a_larger_request_grows_and_still_reuses() {
        let mut ws = Workspace::with_capacity(4).expect("4 blocks");
        {
            let guard = ws.acquire(4).expect("acquire 4");
            assert_eq!(guard.capacity(), 4);
        }

        // Larger than capacity: one reallocation, then steady state again.
        let grown = {
            let guard = ws.acquire(48).expect("grow to 48");
            assert_eq!(guard.len(), 48);
            assert!(guard.capacity() >= 48);
            guard.as_ptr()
        };
        assert!(ws.capacity() >= 48);

        for _ in 0..4 {
            let guard = ws.acquire(48).expect("reacquire 48");
            assert_eq!(guard.as_ptr(), grown, "growth happens once");
        }
    }

    #[test]
    fn release_keeps_the_larger_of_two_arenas() {
        let mut ws = Workspace::with_capacity(4).expect("4 blocks");
        let small = ws.acquire_owned(4).expect("owned 4");
        let large = Arena::new(64).expect("64 blocks");

        ws.release(large);
        assert_eq!(ws.capacity(), 64);
        ws.release(small);
        assert_eq!(ws.capacity(), 64, "a smaller arena must not evict a larger");
    }

    #[test]
    fn acquire_rejects_zero_blocks() {
        let mut ws = Workspace::new();
        assert!(matches!(
            ws.acquire(0).err(),
            Some(Error::MemoryAllocationError)
        ));
        assert!(matches!(
            ws.acquire_owned(0).err(),
            Some(Error::MemoryAllocationError)
        ));
    }

    #[test]
    fn reserve_is_idempotent_and_never_shrinks() {
        let mut ws = Workspace::new();
        ws.reserve(0).expect("reserving nothing is a no-op");
        assert_eq!(ws.capacity(), 0);

        ws.reserve(32).expect("reserve 32");
        assert_eq!(ws.capacity(), 32);
        ws.reserve(8).expect("reserve 8");
        assert_eq!(ws.capacity(), 32, "reserve never shrinks");
        ws.reserve(32).expect("reserve 32 again");
        assert_eq!(ws.capacity(), 32);
    }

    #[test]
    fn clear_drops_the_parked_arena() {
        let mut ws = Workspace::with_capacity(16).expect("16 blocks");
        assert_eq!(ws.capacity(), 16);
        ws.clear();
        assert_eq!(ws.capacity(), 0);
        // Still usable afterwards.
        assert_eq!(ws.acquire(2).expect("acquire after clear").len(), 2);
    }

    #[test]
    fn owned_acquisition_round_trips_by_hand() {
        let mut ws = Workspace::with_capacity(16).expect("16 blocks");
        let arena = ws.acquire_owned(16).expect("owned");
        let ptr = arena.as_ptr();
        assert_eq!(ws.capacity(), 0, "on loan, so nothing is parked");
        ws.release(arena);
        assert_eq!(ws.capacity(), 16);
        assert_eq!(ws.acquire(16).expect("reacquire").as_ptr(), ptr);
    }

    #[test]
    fn dropping_an_owned_arena_instead_of_releasing_it_is_safe() {
        let mut ws = Workspace::with_capacity(8).expect("8 blocks");
        drop(ws.acquire_owned(8).expect("owned"));
        assert_eq!(ws.capacity(), 0, "reuse forfeited, nothing else");
        // The workspace still works; it just allocates again.
        assert_eq!(ws.acquire(8).expect("acquire").len(), 8);
    }

    // -----------------------------------------------------------------------
    // Workspace — the security properties
    // -----------------------------------------------------------------------

    /// The headline invariant: what one borrower wrote must never be visible to
    /// the next one.
    #[test]
    #[cfg(feature = "zeroize-memory")]
    fn a_reused_arena_cannot_leak_the_previous_tenants_bytes() {
        let mut ws = Workspace::with_capacity(32).expect("32 blocks");

        for round in 0u8..4 {
            let mut guard = ws.acquire(32).expect("acquire");
            assert!(
                is_all_zero(guard.as_slice()),
                "round {round} started dirty — release did not wipe"
            );
            // Stand in for a hash: fill every block with a distinctive pattern.
            for (i, block) in guard.as_mut_slice().iter_mut().enumerate() {
                block.fill(0xC0u8.wrapping_add(round).wrapping_add(i as u8));
            }
            assert!(!is_all_zero(guard.as_slice()), "the pattern must land");
        }

        // And the very last release wiped too, so nothing is parked dirty.
        let parked = ws.arena.as_ref().expect("parked");
        assert!(is_all_zero(whole_capacity(parked)));
        assert!(parked.is_known_zeroed());
    }

    /// Release wipes the *whole* window the borrower had, not a prefix of it,
    /// and the blocks outside a later, smaller window stay zero — invariant 2.
    #[test]
    #[cfg(feature = "zeroize-memory")]
    fn release_wipes_the_whole_borrowed_window() {
        let mut ws = Workspace::with_capacity(64).expect("64 blocks");
        {
            let mut guard = ws.acquire(64).expect("acquire 64");
            for block in guard.as_mut_slice() {
                block.fill(0xEE);
            }
        }

        let parked = ws.arena.as_ref().expect("parked");
        assert_eq!(parked.capacity(), 64);
        assert!(is_all_zero(whole_capacity(parked)), "all 64 blocks wiped");

        // A smaller acquisition must not resurrect the tail.
        let guard = ws.acquire(8).expect("acquire 8");
        assert!(is_all_zero(guard.as_slice()));
        assert!(is_all_zero(whole_capacity(&guard)), "the tail stays zero");
    }

    /// The tail beyond a *small* window must be zero even though release only
    /// ever wipes the visible blocks. This is the induction step of invariant 2
    /// and the one that would break if `release` wiped a prefix instead.
    #[test]
    #[cfg(feature = "zeroize-memory")]
    fn a_small_borrower_cannot_dirty_the_tail() {
        let mut ws = Workspace::with_capacity(64).expect("64 blocks");
        {
            let mut guard = ws.acquire(4).expect("acquire 4");
            for block in guard.as_mut_slice() {
                block.fill(0xB7);
            }
            assert_eq!(guard.len(), 4);
        }
        // Now take the whole thing: blocks 4..64 were never handed out, so they
        // must still be the zeros `alloc_zeroed` produced.
        let guard = ws.acquire(64).expect("acquire 64");
        assert!(is_all_zero(guard.as_slice()));
    }

    #[test]
    #[cfg(feature = "zeroize-memory")]
    fn growth_does_not_carry_bytes_over() {
        let mut ws = Workspace::with_capacity(8).expect("8 blocks");
        {
            let mut guard = ws.acquire(8).expect("acquire 8");
            for block in guard.as_mut_slice() {
                block.fill(0x5C);
            }
        }
        let guard = ws.acquire(96).expect("grow to 96");
        assert!(is_all_zero(guard.as_slice()), "a grown arena is zeroed");
    }

    /// Without `zeroize-memory` the workspace makes no zeroing promise, but
    /// `ensure_zeroed` still does, and the arena is always *initialised*.
    #[test]
    fn ensure_zeroed_holds_regardless_of_the_wipe_feature() {
        let mut ws = Workspace::with_capacity(16).expect("16 blocks");
        {
            let mut guard = ws.acquire(16).expect("acquire");
            for block in guard.as_mut_slice() {
                block.fill(0x93);
            }
        }
        let mut guard = ws.acquire(16).expect("reacquire");
        guard.ensure_zeroed();
        assert!(is_all_zero(guard.as_slice()));

        // Reading every block is sound either way: initialised is the real
        // requirement, zero is not.
        let sum: u64 = guard.as_slice().iter().map(|b| b.0[0]).sum();
        assert_eq!(sum, 0);
    }

    /// Invariant 3 is the security-critical one: `is_known_zeroed()` decides
    /// whether `Drop` may skip a wipe, so it must never be `true` while any
    /// block in the *capacity* is dirty — not just the visible window.
    ///
    /// The trap this guards is real and was live during development: with
    /// `zeroize-memory` off, invariant 2 does not hold, so an `ensure_zeroed`
    /// that only covered `[0, len())` would set the flag while the tail beyond
    /// the window still held the previous hash's bytes. Widening the window
    /// then exposes them under a `true` flag.
    #[test]
    fn known_zeroed_never_lies_about_the_capacity() {
        let mut ws = Workspace::with_capacity(64).expect("64 blocks");

        // Dirty the whole 64-block capacity.
        {
            let mut guard = ws.acquire(64).expect("acquire 64");
            for block in guard.as_mut_slice() {
                block.fill(0x88);
            }
        }
        // Now take a narrow window and ask for the zero guarantee.
        {
            let mut guard = ws.acquire(4).expect("acquire 4");
            guard.ensure_zeroed();
            assert!(guard.is_known_zeroed());
        }
        // Widen again. The flag survived the release (release cannot dirty
        // anything), so if it still claims zero, every block must be zero.
        let guard = ws.acquire(64).expect("acquire 64 again");
        if guard.is_known_zeroed() {
            assert!(
                is_all_zero(guard.as_slice()),
                "is_known_zeroed() claimed zero while the tail was dirty"
            );
        }
    }

    #[test]
    fn a_dropped_workspace_wipes_what_it_parked() {
        // Not directly observable after the free, so assert the flag that
        // decides it: `Drop` wipes exactly when the arena is not known zero.
        let mut arena = Arena::new(4).expect("4 blocks");
        arena.as_mut_slice()[0].fill(0x11);
        assert!(!arena.is_known_zeroed(), "Drop will wipe this one");

        let mut ws = Workspace::with_capacity(4).expect("4 blocks");
        {
            let mut guard = ws.acquire(4).expect("acquire");
            guard.as_mut_slice()[0].fill(0x22);
        }
        assert_eq!(
            ws.arena.as_ref().map(Arena::is_known_zeroed),
            Some(WIPE_ENABLED),
            "release wipes iff the feature is on"
        );
    }

    // -----------------------------------------------------------------------
    // Thread-safety shape
    // -----------------------------------------------------------------------

    /// Everything here must be movable between threads, so a server can hand a
    /// workspace to whichever worker picks the request up.
    ///
    /// The matching negative — that none of them is `Sync` — is the
    /// `compile_fail` doctest on [`Workspace`]. A `Sync` workspace would let a
    /// `Bump` be shared across threads, which is exactly the design the brief
    /// forbids, so that doctest is a real guard and not decoration.
    #[test]
    fn send_but_not_sync() {
        const fn assert_send<T: Send>() {}
        assert_send::<Arena>();
        assert_send::<Workspace>();
        assert_send::<ArenaGuard<'static>>();
    }

    // -----------------------------------------------------------------------
    // Bump
    // -----------------------------------------------------------------------

    #[test]
    #[cfg(feature = "bump-alloc")]
    fn bump_serves_and_recycles_small_buffers() {
        let mut ws = Workspace::new();
        assert_eq!(ws.bump_reserved_bytes(), 0, "Bump::new allocates nothing");

        let first = {
            let buf = ws
                .bump()
                .try_alloc_slice_fill_copy(98usize, 0u8)
                .expect("98 bytes");
            assert_eq!(buf.len(), 98);
            assert!(buf.iter().all(|b| *b == 0));
            buf.as_ptr()
        };
        assert!(ws.bump_reserved_bytes() >= 98);

        // Reset recycles the chunk rather than returning it to the allocator —
        // that is what makes the second hash cheaper than the first, and it is
        // why `reserved` must NOT drop back to zero here.
        ws.reset_bump();
        assert!(
            ws.bump_reserved_bytes() >= 98,
            "reset keeps its chunk; only `clear` gives it back"
        );

        let second = ws
            .bump()
            .try_alloc_slice_fill_copy(98usize, 0u8)
            .expect("98 bytes again")
            .as_ptr();
        assert_eq!(first, second, "reset must reuse the chunk");
    }

    /// The wipe has to be checked by *reading back the same bytes*, not by
    /// allocating a fresh zeroed buffer over them — that would pass whether or
    /// not the wipe ran. So: write a pattern, reset, then re-claim the exact
    /// same region with `try_alloc_layout`, which allocates without writing,
    /// and read what is actually there.
    #[test]
    #[cfg(all(feature = "bump-alloc", feature = "zeroize-memory"))]
    fn reset_bump_wipes_the_scratch() {
        const LEN: usize = 64;

        let mut ws = Workspace::new();
        let written = {
            let buf = ws
                .bump()
                .try_alloc_slice_fill_copy(LEN, 0xABu8)
                .expect("64 bytes");
            assert!(buf.iter().all(|b| *b == 0xAB), "the pattern must land");
            buf.as_ptr()
        };

        ws.reset_bump();

        let layout = Layout::from_size_align(LEN, 1).expect("valid layout");
        let reclaimed = ws.bump().try_alloc_layout(layout).expect("same region");
        assert_eq!(
            reclaimed.as_ptr().cast_const(),
            written,
            "reset must recycle the chunk, or this test proves nothing"
        );

        // SAFETY: `reclaimed` is a fresh, exclusive `LEN`-byte allocation from
        // the bump, and every one of those bytes was initialised above (0xAB,
        // then whatever `reset_bump` left). The chunk is still owned by `ws`,
        // which outlives this read.
        let bytes: Vec<u8> =
            unsafe { core::slice::from_raw_parts(reclaimed.as_ptr(), LEN) }.to_vec();
        assert!(
            bytes.iter().all(|b| *b == 0),
            "reset_bump must wipe scratch before recycling it, found {bytes:02x?}"
        );
    }

    #[test]
    #[cfg(feature = "bump-alloc")]
    fn clear_resets_both_halves() {
        let mut ws = Workspace::with_capacity(8).expect("8 blocks");
        let _ = ws
            .bump()
            .try_alloc_slice_fill_copy(32usize, 0u8)
            .expect("32 bytes");
        assert!(ws.bump_reserved_bytes() >= 32);
        assert_eq!(ws.capacity(), 8);

        ws.clear();
        assert_eq!(ws.capacity(), 0, "the arena went back to the allocator");
        assert_eq!(ws.bump_reserved_bytes(), 0, "and so did the bump chunks");

        // Still usable afterwards, both halves.
        assert_eq!(ws.acquire(8).expect("acquire after clear").len(), 8);
        assert_eq!(
            ws.bump()
                .try_alloc_slice_fill_copy(32usize, 0u8)
                .expect("32 bytes after clear")
                .len(),
            32
        );
    }
}
