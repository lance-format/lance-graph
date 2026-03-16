// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # Types for BlasGraph
//!
//! Core types used throughout the BlasGraph module, including the self-contained
//! 16384-bit [`BitVec`] type for hyperdimensional computing and all operator enums.

use std::fmt;

/// Number of u64 words in a 16384-bit vector.
pub const VECTOR_WORDS: usize = 256;

/// Number of bits in a vector.
pub const VECTOR_BITS: usize = VECTOR_WORDS * 64;

/// A 16384-bit binary vector for hyperdimensional computing.
///
/// This is the fundamental data type for the BlasGraph algebra. Each vector
/// occupies 2 KiB and supports bitwise operations, Hamming distance, and
/// majority-vote bundling.
#[derive(Clone, PartialEq, Eq)]
pub struct BitVec {
    /// The raw storage: 256 × 64-bit words = 16384 bits.
    words: [u64; VECTOR_WORDS],
}

impl fmt::Debug for BitVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pop = self.popcount();
        let density = self.density();
        write!(f, "BitVec {{ popcount: {}, density: {:.4} }}", pop, density)
    }
}

impl Default for BitVec {
    fn default() -> Self {
        Self::zero()
    }
}

/// SplitMix64 pseudo-random number generator.
///
/// Unlike xorshift64, SplitMix64 is **non-linear**, which prevents the
/// property `random(a ^ b) == random(a) ^ random(b)` that causes XOR
/// chains to collapse to zero prematurely.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

impl BitVec {
    /// Create a zero vector (all bits cleared).
    pub fn zero() -> Self {
        Self {
            words: [0u64; VECTOR_WORDS],
        }
    }

    /// Create a ones vector (all bits set).
    pub fn ones() -> Self {
        Self {
            words: [u64::MAX; VECTOR_WORDS],
        }
    }

    /// Create a random vector from the given seed using xorshift64.
    ///
    /// The same seed always produces the same vector, which is essential
    /// for reproducible symbol assignment in HDC.
    pub fn random(seed: u64) -> Self {
        let mut state = seed;
        let mut words = [0u64; VECTOR_WORDS];
        for w in words.iter_mut() {
            *w = splitmix64(&mut state);
        }
        Self { words }
    }

    /// Create a `BitVec` from raw words. Panics if slice length != 256.
    pub fn from_words(words: &[u64]) -> Self {
        assert_eq!(
            words.len(),
            VECTOR_WORDS,
            "Expected {} words, got {}",
            VECTOR_WORDS,
            words.len()
        );
        let mut arr = [0u64; VECTOR_WORDS];
        arr.copy_from_slice(words);
        Self { words: arr }
    }

    /// Return a reference to the raw words.
    pub fn words(&self) -> &[u64; VECTOR_WORDS] {
        &self.words
    }

    /// Return a mutable reference to the raw words.
    pub fn words_mut(&mut self) -> &mut [u64; VECTOR_WORDS] {
        &mut self.words
    }

    /// Bitwise XOR of two vectors.
    pub fn xor(&self, other: &BitVec) -> BitVec {
        let mut result = BitVec::zero();
        for i in 0..VECTOR_WORDS {
            result.words[i] = self.words[i] ^ other.words[i];
        }
        result
    }

    /// Bitwise AND of two vectors.
    pub fn and(&self, other: &BitVec) -> BitVec {
        let mut result = BitVec::zero();
        for i in 0..VECTOR_WORDS {
            result.words[i] = self.words[i] & other.words[i];
        }
        result
    }

    /// Bitwise OR of two vectors.
    pub fn or(&self, other: &BitVec) -> BitVec {
        let mut result = BitVec::zero();
        for i in 0..VECTOR_WORDS {
            result.words[i] = self.words[i] | other.words[i];
        }
        result
    }

    /// Bitwise NOT of this vector.
    pub fn not(&self) -> BitVec {
        let mut result = BitVec::zero();
        for i in 0..VECTOR_WORDS {
            result.words[i] = !self.words[i];
        }
        result
    }

    /// Count the number of set bits (population count).
    pub fn popcount(&self) -> u32 {
        let mut count = 0u32;
        for &w in self.words.iter() {
            count += w.count_ones();
        }
        count
    }

    /// Density: fraction of bits that are set, in `[0.0, 1.0]`.
    pub fn density(&self) -> f32 {
        self.popcount() as f32 / VECTOR_BITS as f32
    }

    /// Hamming distance to another vector (number of differing bits).
    pub fn hamming_distance(&self, other: &BitVec) -> u32 {
        let mut dist = 0u32;
        for i in 0..VECTOR_WORDS {
            dist += (self.words[i] ^ other.words[i]).count_ones();
        }
        dist
    }

    /// Bundle (majority vote) over a slice of vectors.
    ///
    /// For each bit position, count the number of vectors that have a 1.
    /// If more than half do, the result bit is 1; otherwise 0.
    /// Ties (even number of vectors, exactly half set) resolve to 0.
    pub fn bundle(vectors: &[&BitVec]) -> BitVec {
        if vectors.is_empty() {
            return BitVec::zero();
        }
        if vectors.len() == 1 {
            return vectors[0].clone();
        }

        let threshold = vectors.len() / 2;
        let mut result = BitVec::zero();

        for word_idx in 0..VECTOR_WORDS {
            let mut result_word = 0u64;
            for bit in 0..64 {
                let mask = 1u64 << bit;
                let mut count = 0usize;
                for v in vectors.iter() {
                    if v.words[word_idx] & mask != 0 {
                        count += 1;
                    }
                }
                if count > threshold {
                    result_word |= mask;
                }
            }
            result.words[word_idx] = result_word;
        }

        result
    }

    /// Check if this vector is the zero vector.
    pub fn is_zero(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Cyclic left-shift (permutation) by `n` bits.
    pub fn permute(&self, n: u32) -> BitVec {
        if n == 0 {
            return self.clone();
        }
        let shift = (n as usize) % VECTOR_BITS;
        if shift == 0 {
            return self.clone();
        }

        let word_shift = shift / 64;
        let bit_shift = shift % 64;
        let mut result = BitVec::zero();

        if bit_shift == 0 {
            for i in 0..VECTOR_WORDS {
                let src = (i + VECTOR_WORDS - word_shift) % VECTOR_WORDS;
                result.words[i] = self.words[src];
            }
        } else {
            for i in 0..VECTOR_WORDS {
                let src_hi = (i + VECTOR_WORDS - word_shift) % VECTOR_WORDS;
                let src_lo = (i + VECTOR_WORDS - word_shift - 1) % VECTOR_WORDS;
                result.words[i] =
                    (self.words[src_hi] << bit_shift) | (self.words[src_lo] >> (64 - bit_shift));
            }
        }

        result
    }
}

/// Convert Hamming distance to a similarity score in `[0.0, 1.0]`.
///
/// `similarity = 1.0 - distance / VECTOR_BITS`
pub fn hamming_to_similarity(distance: u32) -> f32 {
    1.0 - (distance as f32 / VECTOR_BITS as f32)
}

/// Scalar type that can inhabit a matrix or vector element position.
#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum HdrScalar {
    /// A hyperdimensional binary vector.
    Vector(BitVec),
    /// A floating-point scalar (e.g. Hamming distance, similarity).
    Float(f32),
    /// A boolean value.
    Bool(bool),
    /// Absence of a value (structural zero).
    Empty,
}

impl HdrScalar {
    /// Try to extract the inner `BitVec`.
    pub fn as_vector(&self) -> Option<&BitVec> {
        match self {
            HdrScalar::Vector(v) => Some(v),
            _ => None,
        }
    }

    /// Try to extract the inner float.
    pub fn as_float(&self) -> Option<f32> {
        match self {
            HdrScalar::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Try to extract the inner bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            HdrScalar::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Check if this is an empty (structural zero) value.
    pub fn is_empty(&self) -> bool {
        matches!(self, HdrScalar::Empty)
    }
}

/// Type descriptor for elements in a GrB container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrBType {
    /// 16384-bit hyperdimensional vector.
    HdrVector,
    /// 32-bit floating-point scalar.
    Float32,
    /// Boolean.
    Bool,
}

/// Unary operators that can be applied element-wise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Bitwise NOT.
    BitwiseNot,
    /// Identity (no-op).
    Identity,
    /// Negate a float.
    Negate,
    /// Convert vector to its density (float).
    Density,
    /// Cyclic permutation by 1 bit.
    Permute,
}

/// Binary operators used in semiring multiply or element-wise operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// Bitwise XOR (bind operation in HDC).
    Xor,
    /// Bitwise AND.
    And,
    /// Bitwise OR.
    Or,
    /// Majority-vote bundle of two vectors.
    Bundle,
    /// Return the first operand.
    First,
    /// Return the second operand.
    Second,
    /// Hamming distance (result is Float).
    HammingDistance,
    /// Cosine-like similarity via Hamming (result is Float).
    Similarity,
    /// Floating-point addition.
    FloatAdd,
    /// Floating-point multiplication.
    FloatMul,
    /// Floating-point minimum.
    FloatMin,
    /// Floating-point maximum.
    FloatMax,
    /// Boolean AND.
    BoolAnd,
    /// Boolean OR.
    BoolOr,
}

/// Monoid for the additive part of a semiring.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonoidOp {
    /// XOR monoid (identity = zero vector).
    Xor,
    /// OR monoid (identity = zero vector / false).
    Or,
    /// AND monoid (identity = ones vector / true).
    And,
    /// Bundle (majority vote) monoid.
    BundleMonoid,
    /// First non-empty monoid.
    First,
    /// Min monoid for floats (identity = +inf).
    Min,
    /// Max monoid for floats (identity = -inf).
    Max,
    /// Float addition monoid (identity = 0.0).
    FloatAdd,
    /// Best density: keep the vector with density closest to 0.5.
    BestDensity,
    /// Keep the vector with the smallest popcount (min Hamming weight).
    MinPopcount,
    /// XOR for GF(2) field.
    XorField,
}

/// Select which elements participate in a masked operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectOp {
    /// Select non-zero elements.
    NonZero,
    /// Select elements equal to a given value.
    Equal,
    /// Select elements not equal to a given value.
    NotEqual,
    /// Select elements greater than a threshold (for floats).
    GreaterThan,
    /// Select elements less than a threshold (for floats).
    LessThan,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_vector() {
        let v = BitVec::zero();
        assert_eq!(v.popcount(), 0);
        assert!(v.is_zero());
        assert_eq!(v.density(), 0.0);
    }

    #[test]
    fn test_ones_vector() {
        let v = BitVec::ones();
        assert_eq!(v.popcount(), VECTOR_BITS as u32);
        assert!(!v.is_zero());
        assert_eq!(v.density(), 1.0);
    }

    #[test]
    fn test_random_deterministic() {
        let a = BitVec::random(42);
        let b = BitVec::random(42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_random_different_seeds() {
        let a = BitVec::random(1);
        let b = BitVec::random(2);
        assert_ne!(a, b);
    }

    #[test]
    fn test_random_density_near_half() {
        let v = BitVec::random(12345);
        let d = v.density();
        // A random vector should have density near 0.5
        assert!(d > 0.4 && d < 0.6, "density was {}", d);
    }

    #[test]
    fn test_xor_self_is_zero() {
        let v = BitVec::random(99);
        let z = v.xor(&v);
        assert!(z.is_zero());
    }

    #[test]
    fn test_xor_inverse() {
        let a = BitVec::random(10);
        let b = BitVec::random(20);
        let c = a.xor(&b);
        let recovered = c.xor(&b);
        assert_eq!(recovered, a);
    }

    #[test]
    fn test_and_with_ones() {
        let v = BitVec::random(7);
        let result = v.and(&BitVec::ones());
        assert_eq!(result, v);
    }

    #[test]
    fn test_and_with_zero() {
        let v = BitVec::random(7);
        let result = v.and(&BitVec::zero());
        assert!(result.is_zero());
    }

    #[test]
    fn test_or_with_zero() {
        let v = BitVec::random(7);
        let result = v.or(&BitVec::zero());
        assert_eq!(result, v);
    }

    #[test]
    fn test_not_inverse() {
        let v = BitVec::random(8);
        let double_not = v.not().not();
        assert_eq!(double_not, v);
    }

    #[test]
    fn test_hamming_distance_self() {
        let v = BitVec::random(50);
        assert_eq!(v.hamming_distance(&v), 0);
    }

    #[test]
    fn test_hamming_distance_complement() {
        let v = BitVec::random(50);
        let inv = v.not();
        assert_eq!(v.hamming_distance(&inv), VECTOR_BITS as u32);
    }

    #[test]
    fn test_hamming_to_similarity() {
        assert_eq!(hamming_to_similarity(0), 1.0);
        assert_eq!(hamming_to_similarity(VECTOR_BITS as u32), 0.0);
        let half = hamming_to_similarity(VECTOR_BITS as u32 / 2);
        assert!((half - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_bundle_single() {
        let v = BitVec::random(1);
        let result = BitVec::bundle(&[&v]);
        assert_eq!(result, v);
    }

    #[test]
    fn test_bundle_three_identical() {
        let v = BitVec::random(2);
        let result = BitVec::bundle(&[&v, &v, &v]);
        assert_eq!(result, v);
    }

    #[test]
    fn test_bundle_majority() {
        let a = BitVec::random(100);
        let b = BitVec::random(200);
        let result = BitVec::bundle(&[&a, &a, &b]);
        // Result should be closer to a than to b
        let dist_a = result.hamming_distance(&a);
        let dist_b = result.hamming_distance(&b);
        assert!(dist_a < dist_b);
    }

    #[test]
    fn test_permute_zero() {
        let v = BitVec::random(3);
        assert_eq!(v.permute(0), v);
    }

    #[test]
    fn test_permute_full_cycle() {
        let v = BitVec::random(3);
        let rotated = v.permute(VECTOR_BITS as u32);
        assert_eq!(rotated, v);
    }

    #[test]
    fn test_from_words() {
        let v = BitVec::random(77);
        let v2 = BitVec::from_words(v.words());
        assert_eq!(v, v2);
    }

    #[test]
    fn test_hdr_scalar_variants() {
        let sv = HdrScalar::Vector(BitVec::random(1));
        assert!(sv.as_vector().is_some());
        assert!(sv.as_float().is_none());

        let sf = HdrScalar::Float(0.5);
        assert!(sf.as_float().is_some());

        let sb = HdrScalar::Bool(true);
        assert_eq!(sb.as_bool(), Some(true));

        let se = HdrScalar::Empty;
        assert!(se.is_empty());
    }
}
