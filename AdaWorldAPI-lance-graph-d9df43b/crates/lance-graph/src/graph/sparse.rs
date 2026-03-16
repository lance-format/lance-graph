// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Sparse bitmap operations for SPO fingerprint packing.
//!
//! Uses `[u64; BITMAP_WORDS]` for fixed-width bitmaps that can be
//! packed into Lance vector columns for ANN search.

/// Number of u64 words in a bitmap.
///
/// Previously hardcoded as `[u64; 2]` which truncated fingerprints.
/// Now matches the fingerprint width for full coverage.
pub const BITMAP_WORDS: usize = 8;

/// A fixed-width bitmap for sparse set encoding.
pub type Bitmap = [u64; BITMAP_WORDS];

/// Create an empty bitmap (all zeros).
pub const fn bitmap_zero() -> Bitmap {
    [0u64; BITMAP_WORDS]
}

/// OR two bitmaps together.
pub fn bitmap_or(a: &Bitmap, b: &Bitmap) -> Bitmap {
    let mut result = [0u64; BITMAP_WORDS];
    for i in 0..BITMAP_WORDS {
        result[i] = a[i] | b[i];
    }
    result
}

/// AND two bitmaps together.
pub fn bitmap_and(a: &Bitmap, b: &Bitmap) -> Bitmap {
    let mut result = [0u64; BITMAP_WORDS];
    for i in 0..BITMAP_WORDS {
        result[i] = a[i] & b[i];
    }
    result
}

/// XOR two bitmaps (used for Hamming distance).
pub fn bitmap_xor(a: &Bitmap, b: &Bitmap) -> Bitmap {
    let mut result = [0u64; BITMAP_WORDS];
    for i in 0..BITMAP_WORDS {
        result[i] = a[i] ^ b[i];
    }
    result
}

/// Count set bits in a bitmap.
pub fn bitmap_popcount(bm: &Bitmap) -> u32 {
    bm.iter().map(|w| w.count_ones()).sum()
}

/// Hamming distance between two bitmaps.
pub fn bitmap_hamming(a: &Bitmap, b: &Bitmap) -> u32 {
    bitmap_popcount(&bitmap_xor(a, b))
}

/// Check if a bitmap is all zeros.
pub fn bitmap_is_zero(bm: &Bitmap) -> bool {
    bm.iter().all(|&w| w == 0)
}

/// Set a specific bit position (0..BITMAP_WORDS*64).
pub fn bitmap_set_bit(bm: &mut Bitmap, pos: usize) {
    let word = pos / 64;
    let bit = pos % 64;
    if word < BITMAP_WORDS {
        bm[word] |= 1u64 << bit;
    }
}

/// Pack three fingerprints into a combined bitmap for SPO encoding.
///
/// The packed result is the OR of all three, used as the search vector.
/// Individual components can be recovered via AND with the original fingerprints.
pub fn pack_axes(
    s: &[u64; BITMAP_WORDS],
    p: &[u64; BITMAP_WORDS],
    o: &[u64; BITMAP_WORDS],
) -> Bitmap {
    let sp = bitmap_or(s, p);
    bitmap_or(&sp, o)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitmap_zero() {
        let bm = bitmap_zero();
        assert!(bitmap_is_zero(&bm));
        assert_eq!(bitmap_popcount(&bm), 0);
    }

    #[test]
    fn test_bitmap_or() {
        let a = [1u64, 0, 0, 0, 0, 0, 0, 0];
        let b = [0u64, 1, 0, 0, 0, 0, 0, 0];
        let c = bitmap_or(&a, &b);
        assert_eq!(c[0], 1);
        assert_eq!(c[1], 1);
    }

    #[test]
    fn test_bitmap_hamming() {
        let a = [0xFFu64, 0, 0, 0, 0, 0, 0, 0];
        let b = [0x00u64, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(bitmap_hamming(&a, &b), 8);
    }

    #[test]
    fn test_pack_axes() {
        let s = [1u64, 0, 0, 0, 0, 0, 0, 0];
        let p = [2u64, 0, 0, 0, 0, 0, 0, 0];
        let o = [4u64, 0, 0, 0, 0, 0, 0, 0];
        let packed = pack_axes(&s, &p, &o);
        assert_eq!(packed[0], 7); // 1|2|4 = 7
    }

    #[test]
    fn test_bitmap_words_matches_fingerprint() {
        // BITMAP_WORDS must match FINGERPRINT_WORDS
        assert_eq!(BITMAP_WORDS, super::super::fingerprint::FINGERPRINT_WORDS);
    }
}
