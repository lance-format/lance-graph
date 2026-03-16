// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Bridge types for lance-graph <-> ndarray interop.
//!
//! Provides zero-copy conversion between lance-graph's `BitVec` and ndarray's
//! `Fingerprint<256>`, and wraps ndarray's SIMD-dispatched hamming/popcount
//! operations for use by the semiring algebra.
//!
//! When the `ndarray-hpc` feature is enabled, these bridges use the actual
//! ndarray types. Otherwise, they provide compatible standalone implementations.

use super::types::{BitVec, VECTOR_WORDS};

/// Number of bytes in a 16384-bit vector.
const VECTOR_BYTES: usize = VECTOR_WORDS * 8;

// ---------------------------------------------------------------------------
// NdarrayFingerprint — standalone mirror of ndarray's Fingerprint<256>
// ---------------------------------------------------------------------------

/// A newtype wrapping `[u64; 256]` that matches ndarray's `Fingerprint<256>` API.
///
/// When ndarray is available as a dependency, this can be converted to/from
/// `Fingerprint<256>` via `From` impls. In standalone mode it provides the
/// same SIMD-dispatched hamming and popcount operations.
#[derive(Clone, PartialEq, Eq)]
pub struct NdarrayFingerprint {
    /// Raw storage: 256 x 64-bit words = 16384 bits.
    /// Public to match ndarray's `Fingerprint { pub words }` layout.
    pub words: [u64; VECTOR_WORDS],
}

impl std::fmt::Debug for NdarrayFingerprint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pop = self.popcount();
        write!(f, "NdarrayFingerprint {{ popcount: {} }}", pop)
    }
}

impl Default for NdarrayFingerprint {
    fn default() -> Self {
        Self {
            words: [0u64; VECTOR_WORDS],
        }
    }
}

impl NdarrayFingerprint {
    /// Create a zeroed fingerprint.
    pub fn zero() -> Self {
        Self::default()
    }

    /// Create a fingerprint with all bits set.
    pub fn ones() -> Self {
        Self {
            words: [u64::MAX; VECTOR_WORDS],
        }
    }

    /// Construct from a word array.
    pub fn from_words(words: [u64; VECTOR_WORDS]) -> Self {
        Self { words }
    }

    /// Return a byte-level view of the fingerprint.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: [u64; 256] and [u8; 2048] have the same size and alignment
        // requirements are satisfied (u8 has alignment 1).
        unsafe {
            std::slice::from_raw_parts(self.words.as_ptr() as *const u8, VECTOR_BYTES)
        }
    }

    /// Population count (number of set bits).
    pub fn popcount(&self) -> u64 {
        dispatch_popcount(self.as_bytes())
    }

    /// Hamming distance to another fingerprint.
    pub fn hamming_distance(&self, other: &NdarrayFingerprint) -> u64 {
        dispatch_hamming(self.as_bytes(), other.as_bytes())
    }
}

// ---------------------------------------------------------------------------
// From conversions: BitVec <-> NdarrayFingerprint
// ---------------------------------------------------------------------------

impl From<&BitVec> for NdarrayFingerprint {
    /// Zero-copy-semantics conversion from `BitVec` to `NdarrayFingerprint`.
    ///
    /// Copies the word array (stack copy, no heap allocation).
    fn from(bv: &BitVec) -> Self {
        NdarrayFingerprint {
            words: *bv.words(),
        }
    }
}

impl From<&NdarrayFingerprint> for BitVec {
    /// Convert an `NdarrayFingerprint` back to a `BitVec`.
    fn from(fp: &NdarrayFingerprint) -> Self {
        BitVec::from_words(&fp.words)
    }
}

// ---------------------------------------------------------------------------
// SIMD dispatch — 4-tier fallback matching ndarray's bitwise.rs pattern
// ---------------------------------------------------------------------------

/// SIMD-dispatched Hamming distance between two byte slices.
///
/// Computes `popcount(a XOR b)` using the best available instruction set:
///
/// 1. **VPOPCNTDQ** (AVX-512 VPOPCNTDQ) — 512-bit popcount in one instruction
/// 2. **AVX-512BW** — 512-bit XOR + byte-level popcount via shuffle LUT
/// 3. **AVX2** — 256-bit XOR + byte-level popcount via shuffle LUT
/// 4. **Scalar** — word-by-word `count_ones()`
///
/// Both slices must have the same length. Panics otherwise.
pub fn dispatch_hamming(a: &[u8], b: &[u8]) -> u64 {
    assert_eq!(a.len(), b.len(), "hamming: slices must have equal length");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512f") {
            // SAFETY: feature detection guarantees VPOPCNTDQ is available.
            return unsafe { hamming_avx512_vpopcntdq(a, b) };
        }
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512f") {
            // SAFETY: feature detection guarantees AVX-512BW is available.
            return unsafe { hamming_avx512bw(a, b) };
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: feature detection guarantees AVX2 is available.
            return unsafe { hamming_avx2(a, b) };
        }
    }

    hamming_scalar(a, b)
}

/// SIMD-dispatched population count over a byte slice.
///
/// Uses the same 4-tier fallback as `dispatch_hamming`:
/// VPOPCNTDQ -> AVX-512BW -> AVX2 -> scalar.
pub fn dispatch_popcount(a: &[u8]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512f") {
            return unsafe { popcount_avx512_vpopcntdq(a) };
        }
        if is_x86_feature_detected!("avx512bw") && is_x86_feature_detected!("avx512f") {
            return unsafe { popcount_avx512bw(a) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { popcount_avx2(a) };
        }
    }

    popcount_scalar(a)
}

// ---------------------------------------------------------------------------
// Scalar fallback — always available
// ---------------------------------------------------------------------------

/// Scalar Hamming distance: XOR each u64 pair and sum `count_ones()`.
fn hamming_scalar(a: &[u8], b: &[u8]) -> u64 {
    // Process 8 bytes at a time as u64 words for efficiency.
    let chunks = a.len() / 8;
    let mut total = 0u64;

    for i in 0..chunks {
        let off = i * 8;
        let wa = u64::from_le_bytes(a[off..off + 8].try_into().unwrap());
        let wb = u64::from_le_bytes(b[off..off + 8].try_into().unwrap());
        total += (wa ^ wb).count_ones() as u64;
    }

    // Handle remaining bytes (< 8).
    let remainder = a.len() % 8;
    if remainder > 0 {
        let off = chunks * 8;
        for i in 0..remainder {
            total += (a[off + i] ^ b[off + i]).count_ones() as u64;
        }
    }

    total
}

/// Scalar popcount: process as u64 words then handle remainder.
fn popcount_scalar(a: &[u8]) -> u64 {
    let chunks = a.len() / 8;
    let mut total = 0u64;

    for i in 0..chunks {
        let off = i * 8;
        let w = u64::from_le_bytes(a[off..off + 8].try_into().unwrap());
        total += w.count_ones() as u64;
    }

    let remainder = a.len() % 8;
    if remainder > 0 {
        let off = chunks * 8;
        for i in 0..remainder {
            total += a[off + i].count_ones() as u64;
        }
    }

    total
}

// ---------------------------------------------------------------------------
// AVX2 — 256-bit SIMD via shuffle-based popcount LUT
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hamming_avx2(a: &[u8], b: &[u8]) -> u64 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 32;
    let mut total = 0u64;

    // Nibble lookup table for popcount: LUT[nibble] = popcount(nibble)
    let lut = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    );
    let mask_lo = _mm256_set1_epi8(0x0f);
    let mut acc = _mm256_setzero_si256();

    for i in 0..chunks {
        let off = i * 32;
        let va = _mm256_loadu_si256(a.as_ptr().add(off) as *const __m256i);
        let vb = _mm256_loadu_si256(b.as_ptr().add(off) as *const __m256i);
        let xored = _mm256_xor_si256(va, vb);

        // Popcount via nibble shuffle LUT
        let lo_nibbles = _mm256_and_si256(xored, mask_lo);
        let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(xored, 4), mask_lo);
        let pop_lo = _mm256_shuffle_epi8(lut, lo_nibbles);
        let pop_hi = _mm256_shuffle_epi8(lut, hi_nibbles);
        let pop = _mm256_add_epi8(pop_lo, pop_hi);

        // Horizontal sum: sad against zero accumulates byte sums into u64 lanes
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(pop, _mm256_setzero_si256()));
    }

    // Extract the 4 u64 lanes from the accumulator
    let mut buf = [0u64; 4];
    _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, acc);
    total += buf[0] + buf[1] + buf[2] + buf[3];

    // Handle remainder bytes with scalar
    let remainder_start = chunks * 32;
    if remainder_start < len {
        total += hamming_scalar(&a[remainder_start..], &b[remainder_start..]);
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn popcount_avx2(a: &[u8]) -> u64 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 32;
    let mut total = 0u64;

    let lut = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    );
    let mask_lo = _mm256_set1_epi8(0x0f);
    let mut acc = _mm256_setzero_si256();

    for i in 0..chunks {
        let off = i * 32;
        let va = _mm256_loadu_si256(a.as_ptr().add(off) as *const __m256i);

        let lo_nibbles = _mm256_and_si256(va, mask_lo);
        let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(va, 4), mask_lo);
        let pop_lo = _mm256_shuffle_epi8(lut, lo_nibbles);
        let pop_hi = _mm256_shuffle_epi8(lut, hi_nibbles);
        let pop = _mm256_add_epi8(pop_lo, pop_hi);

        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(pop, _mm256_setzero_si256()));
    }

    let mut buf = [0u64; 4];
    _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, acc);
    total += buf[0] + buf[1] + buf[2] + buf[3];

    let remainder_start = chunks * 32;
    if remainder_start < len {
        total += popcount_scalar(&a[remainder_start..]);
    }

    total
}

// ---------------------------------------------------------------------------
// AVX-512BW — 512-bit SIMD via shuffle-based popcount LUT
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn hamming_avx512bw(a: &[u8], b: &[u8]) -> u64 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 64;
    let mut total = 0u64;

    let lut = _mm512_set4_epi32(
        0x04030302_i32, 0x03020201_i32, 0x03020201_i32, 0x02010100_i32,
    );
    let mask_lo = _mm512_set1_epi8(0x0f);
    let mut acc = _mm512_setzero_si512();

    for i in 0..chunks {
        let off = i * 64;
        let va = _mm512_loadu_si512(a.as_ptr().add(off) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(off) as *const __m512i);
        let xored = _mm512_xor_si512(va, vb);

        let lo_nibbles = _mm512_and_si512(xored, mask_lo);
        let hi_nibbles = _mm512_and_si512(_mm512_srli_epi16(xored, 4), mask_lo);
        let pop_lo = _mm512_shuffle_epi8(lut, lo_nibbles);
        let pop_hi = _mm512_shuffle_epi8(lut, hi_nibbles);
        let pop = _mm512_add_epi8(pop_lo, pop_hi);

        acc = _mm512_add_epi64(acc, _mm512_sad_epu8(pop, _mm512_setzero_si512()));
    }

    total += _mm512_reduce_add_epi64(acc) as u64;

    let remainder_start = chunks * 64;
    if remainder_start < len {
        total += hamming_scalar(&a[remainder_start..], &b[remainder_start..]);
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn popcount_avx512bw(a: &[u8]) -> u64 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 64;
    let mut total = 0u64;

    let lut = _mm512_set4_epi32(
        0x04030302_i32, 0x03020201_i32, 0x03020201_i32, 0x02010100_i32,
    );
    let mask_lo = _mm512_set1_epi8(0x0f);
    let mut acc = _mm512_setzero_si512();

    for i in 0..chunks {
        let off = i * 64;
        let va = _mm512_loadu_si512(a.as_ptr().add(off) as *const __m512i);

        let lo_nibbles = _mm512_and_si512(va, mask_lo);
        let hi_nibbles = _mm512_and_si512(_mm512_srli_epi16(va, 4), mask_lo);
        let pop_lo = _mm512_shuffle_epi8(lut, lo_nibbles);
        let pop_hi = _mm512_shuffle_epi8(lut, hi_nibbles);
        let pop = _mm512_add_epi8(pop_lo, pop_hi);

        acc = _mm512_add_epi64(acc, _mm512_sad_epu8(pop, _mm512_setzero_si512()));
    }

    total += _mm512_reduce_add_epi64(acc) as u64;

    let remainder_start = chunks * 64;
    if remainder_start < len {
        total += popcount_scalar(&a[remainder_start..]);
    }

    total
}

// ---------------------------------------------------------------------------
// AVX-512 VPOPCNTDQ — native 512-bit popcount
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512vpopcntdq")]
unsafe fn hamming_avx512_vpopcntdq(a: &[u8], b: &[u8]) -> u64 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 64;
    let mut total = 0u64;
    let mut acc = _mm512_setzero_si512();

    for i in 0..chunks {
        let off = i * 64;
        let va = _mm512_loadu_si512(a.as_ptr().add(off) as *const __m512i);
        let vb = _mm512_loadu_si512(b.as_ptr().add(off) as *const __m512i);
        let xored = _mm512_xor_si512(va, vb);
        let popcnt = _mm512_popcnt_epi64(xored);
        acc = _mm512_add_epi64(acc, popcnt);
    }

    total += _mm512_reduce_add_epi64(acc) as u64;

    let remainder_start = chunks * 64;
    if remainder_start < len {
        total += hamming_scalar(&a[remainder_start..], &b[remainder_start..]);
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f", enable = "avx512vpopcntdq")]
unsafe fn popcount_avx512_vpopcntdq(a: &[u8]) -> u64 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 64;
    let mut total = 0u64;
    let mut acc = _mm512_setzero_si512();

    for i in 0..chunks {
        let off = i * 64;
        let va = _mm512_loadu_si512(a.as_ptr().add(off) as *const __m512i);
        let popcnt = _mm512_popcnt_epi64(va);
        acc = _mm512_add_epi64(acc, popcnt);
    }

    total += _mm512_reduce_add_epi64(acc) as u64;

    let remainder_start = chunks * 64;
    if remainder_start < len {
        total += popcount_scalar(&a[remainder_start..]);
    }

    total
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_default_is_zero() {
        let fp = NdarrayFingerprint::default();
        assert_eq!(fp.popcount(), 0);
    }

    #[test]
    fn test_fingerprint_ones() {
        let fp = NdarrayFingerprint::ones();
        assert_eq!(fp.popcount(), (VECTOR_WORDS * 64) as u64);
    }

    #[test]
    fn test_fingerprint_from_bitvec_roundtrip() {
        let bv = BitVec::random(42);
        let fp = NdarrayFingerprint::from(&bv);
        let bv2 = BitVec::from(&fp);
        assert_eq!(bv, bv2);
    }

    #[test]
    fn test_fingerprint_hamming_self_is_zero() {
        let fp = NdarrayFingerprint::from(&BitVec::random(99));
        assert_eq!(fp.hamming_distance(&fp), 0);
    }

    #[test]
    fn test_fingerprint_hamming_complement() {
        let fp = NdarrayFingerprint::from(&BitVec::random(99));
        let mut inv = NdarrayFingerprint::zero();
        for i in 0..VECTOR_WORDS {
            inv.words[i] = !fp.words[i];
        }
        assert_eq!(fp.hamming_distance(&inv), (VECTOR_WORDS * 64) as u64);
    }

    #[test]
    fn test_dispatch_hamming_matches_bitvec() {
        let a = BitVec::random(1);
        let b = BitVec::random(2);
        let expected = a.hamming_distance(&b) as u64;

        let fa = NdarrayFingerprint::from(&a);
        let fb = NdarrayFingerprint::from(&b);
        let got = dispatch_hamming(fa.as_bytes(), fb.as_bytes());

        assert_eq!(got, expected);
    }

    #[test]
    fn test_dispatch_popcount_matches_bitvec() {
        let bv = BitVec::random(12345);
        let expected = bv.popcount() as u64;

        let fp = NdarrayFingerprint::from(&bv);
        let got = dispatch_popcount(fp.as_bytes());

        assert_eq!(got, expected);
    }

    #[test]
    fn test_scalar_hamming_small() {
        let a = [0xFFu8, 0x00, 0xAA];
        let b = [0x00u8, 0xFF, 0x55];
        // 0xFF ^ 0x00 = 0xFF = 8 bits, 0x00 ^ 0xFF = 8 bits,
        // 0xAA ^ 0x55 = 0xFF = 8 bits => total 24
        assert_eq!(hamming_scalar(&a, &b), 24);
    }

    #[test]
    fn test_scalar_popcount_small() {
        let a = [0xFFu8, 0x00, 0x0F];
        // 8 + 0 + 4 = 12
        assert_eq!(popcount_scalar(&a), 12);
    }

    #[test]
    fn test_dispatch_hamming_empty() {
        assert_eq!(dispatch_hamming(&[], &[]), 0);
    }

    #[test]
    fn test_dispatch_popcount_empty() {
        assert_eq!(dispatch_popcount(&[]), 0);
    }

    #[test]
    fn test_fingerprint_words_preserved() {
        let bv = BitVec::random(777);
        let fp = NdarrayFingerprint::from(&bv);
        assert_eq!(&fp.words, bv.words());
    }
}
