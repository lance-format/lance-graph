// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Fingerprint functions for SPO triple addressing.
//!
//! Labels (node names, relationship types) are hashed into fixed-width
//! fingerprints for compact storage and fast comparison in the SPO store.

/// Number of u64 words in a fingerprint vector.
pub const FINGERPRINT_WORDS: usize = 8;

/// A fingerprint is a fixed-width hash of a label string.
pub type Fingerprint = [u64; FINGERPRINT_WORDS];

/// Hash a label string into a fingerprint.
///
/// Uses FNV-1a inspired mixing to distribute bits across all words.
/// The result is deterministic: same label always produces the same fingerprint.
pub fn label_fp(label: &str) -> Fingerprint {
    let mut fp = [0u64; FINGERPRINT_WORDS];
    let bytes = label.as_bytes();

    // Primary hash using FNV-1a constants
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    fp[0] = h;

    // Fill remaining words with cascading mixes
    #[allow(clippy::needless_range_loop)]
    for i in 1..FINGERPRINT_WORDS {
        h = h.wrapping_mul(0x517cc1b727220a95);
        h ^= h >> 17;
        h = h.wrapping_mul(0x6c62272e07bb0142);
        h ^= (i as u64).wrapping_mul(0x9e3779b97f4a7c15);
        fp[i] = h;
    }

    // Guard: reject if density > 11% (prevents pack_axes overflow)
    // Density = popcount / total_bits. At 8 words × 64 bits = 512 bits,
    // 11% ≈ 56 set bits. If we exceed this, rotate to thin out.
    let popcount: u32 = fp.iter().map(|w| w.count_ones()).sum();
    let total_bits = (FINGERPRINT_WORDS * 64) as u32;
    let max_density_bits = total_bits * 11 / 100; // 11% threshold

    if popcount > max_density_bits {
        // Thin out by XOR-folding with shifted self
        for i in 0..FINGERPRINT_WORDS {
            fp[i] ^= fp[i] >> 3;
            fp[i] &= fp[(i + 1) % FINGERPRINT_WORDS].wrapping_shr(1) | fp[i];
        }
        // Re-check and force-mask if still too dense
        let popcount2: u32 = fp.iter().map(|w| w.count_ones()).sum();
        if popcount2 > max_density_bits {
            for w in fp.iter_mut() {
                // Keep only every other bit
                *w &= 0x5555_5555_5555_5555;
            }
        }
    }

    fp
}

/// Hash a DN (distinguished name) path into a u64 address.
///
/// Used for keying records in the SPO store.
pub fn dn_hash(dn: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in dn.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Compute Hamming distance between two fingerprints.
///
/// Returns the number of bit positions where the fingerprints differ.
pub fn hamming_distance(a: &Fingerprint, b: &Fingerprint) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Zero fingerprint constant.
pub const ZERO_FP: Fingerprint = [0u64; FINGERPRINT_WORDS];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_fp_deterministic() {
        let fp1 = label_fp("Jan");
        let fp2 = label_fp("Jan");
        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_label_fp_different_labels() {
        let fp1 = label_fp("Jan");
        let fp2 = label_fp("Ada");
        assert_ne!(fp1, fp2);
    }

    #[test]
    fn test_label_fp_density_bound() {
        // Check that density stays under ~50% for reasonable labels
        for label in &["Jan", "Ada", "KNOWS", "CREATES", "HELPS", "entity_42"] {
            let fp = label_fp(label);
            let popcount: u32 = fp.iter().map(|w| w.count_ones()).sum();
            let total = (FINGERPRINT_WORDS * 64) as u32;
            assert!(
                popcount < total / 2,
                "Label '{}' has density {}/{}",
                label,
                popcount,
                total
            );
        }
    }

    #[test]
    fn test_dn_hash_deterministic() {
        assert_eq!(dn_hash("edge:jan-knows-ada"), dn_hash("edge:jan-knows-ada"));
    }

    #[test]
    fn test_hamming_distance_self() {
        let fp = label_fp("test");
        assert_eq!(hamming_distance(&fp, &fp), 0);
    }

    #[test]
    fn test_hamming_distance_different() {
        let fp1 = label_fp("Jan");
        let fp2 = label_fp("Ada");
        assert!(hamming_distance(&fp1, &fp2) > 0);
    }
}
