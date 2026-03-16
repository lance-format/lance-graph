// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # Semiring Algebra for Hyperdimensional Vectors
//!
//! Defines the [`Semiring`] trait and concrete [`HdrSemiring`] variants used
//! by matrix and vector operations. Each semiring pairs a multiplicative
//! operator (applied during inner products) with an additive monoid
//! (used to accumulate partial results).

use crate::graph::blasgraph::ndarray_bridge;
use crate::graph::blasgraph::types::{
    hamming_to_similarity, BinaryOp, BitVec, HdrScalar, MonoidOp,
};

/// A semiring over [`HdrScalar`] values.
///
/// The two fundamental operations are:
/// - **multiply** (`otimes`): combines two scalars during an inner product step.
/// - **add** (`oplus`): accumulates partial products into a single result.
///
/// The `zero` element is the additive identity.
pub trait Semiring: Send + Sync {
    /// The multiplicative operator.
    fn multiply(&self, a: &HdrScalar, b: &HdrScalar) -> HdrScalar;

    /// The additive (accumulation) operator.
    fn add(&self, a: &HdrScalar, b: &HdrScalar) -> HdrScalar;

    /// The additive identity element.
    fn zero(&self) -> HdrScalar;

    /// Human-readable name of this semiring.
    fn name(&self) -> &str;
}

/// Pre-defined semirings for hyperdimensional graph computations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrSemiring {
    /// XOR multiply, majority-vote bundle add. Path composition.
    XorBundle,
    /// XOR multiply, first non-empty add. BFS traversal.
    BindFirst,
    /// Hamming distance multiply, min add. Shortest path.
    HammingMin,
    /// Similarity multiply, max add. Best match.
    SimilarityMax,
    /// XOR bind multiply, best-density add. Query expansion.
    Resonance,
    /// AND multiply, OR add. Boolean reachability.
    Boolean,
    /// XOR multiply, XOR add. GF(2) field algebra.
    XorField,
}

impl HdrSemiring {
    /// Get the [`BinaryOp`] used for the multiplicative step.
    pub fn multiply_op(&self) -> BinaryOp {
        match self {
            HdrSemiring::XorBundle => BinaryOp::Xor,
            HdrSemiring::BindFirst => BinaryOp::Xor,
            HdrSemiring::HammingMin => BinaryOp::Xor,
            HdrSemiring::SimilarityMax => BinaryOp::Xor,
            HdrSemiring::Resonance => BinaryOp::Xor,
            HdrSemiring::Boolean => BinaryOp::And,
            HdrSemiring::XorField => BinaryOp::Xor,
        }
    }

    /// Get the [`MonoidOp`] used for the additive accumulation.
    pub fn add_op(&self) -> MonoidOp {
        match self {
            HdrSemiring::XorBundle => MonoidOp::BundleMonoid,
            HdrSemiring::BindFirst => MonoidOp::First,
            HdrSemiring::HammingMin => MonoidOp::MinPopcount,
            HdrSemiring::SimilarityMax => MonoidOp::MinPopcount,
            HdrSemiring::Resonance => MonoidOp::BestDensity,
            HdrSemiring::Boolean => MonoidOp::Or,
            HdrSemiring::XorField => MonoidOp::XorField,
        }
    }
}

impl Semiring for HdrSemiring {
    fn multiply(&self, a: &HdrScalar, b: &HdrScalar) -> HdrScalar {
        match self {
            HdrSemiring::XorBundle | HdrSemiring::BindFirst | HdrSemiring::XorField => {
                match (a, b) {
                    (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => HdrScalar::Vector(va.xor(vb)),
                    _ => HdrScalar::Empty,
                }
            }
            HdrSemiring::Resonance => match (a, b) {
                (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => HdrScalar::Vector(va.xor(vb)),
                _ => HdrScalar::Empty,
            },
            HdrSemiring::HammingMin | HdrSemiring::SimilarityMax => match (a, b) {
                (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                    HdrScalar::Vector(va.xor(vb))
                }
                _ => HdrScalar::Empty,
            },
            HdrSemiring::Boolean => match (a, b) {
                (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => HdrScalar::Vector(va.and(vb)),
                (HdrScalar::Bool(ba), HdrScalar::Bool(bb)) => HdrScalar::Bool(*ba && *bb),
                _ => HdrScalar::Empty,
            },
        }
    }

    fn add(&self, a: &HdrScalar, b: &HdrScalar) -> HdrScalar {
        // If either operand is empty, return the other.
        if a.is_empty() {
            return b.clone();
        }
        if b.is_empty() {
            return a.clone();
        }

        match self {
            HdrSemiring::XorBundle => match (a, b) {
                (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                    HdrScalar::Vector(BitVec::bundle(&[va, vb]))
                }
                _ => a.clone(),
            },
            HdrSemiring::BindFirst => {
                // Return first non-empty; `a` is non-empty at this point.
                a.clone()
            }
            HdrSemiring::HammingMin | HdrSemiring::SimilarityMax => match (a, b) {
                (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                    // Keep the vector with the smallest popcount (min Hamming
                    // weight). For HammingMin this is "shortest path"; for
                    // SimilarityMax it is "highest similarity" (least
                    // difference).
                    // Uses SIMD-dispatched popcount via ndarray bridge.
                    let fa = ndarray_bridge::NdarrayFingerprint::from(va);
                    let fb = ndarray_bridge::NdarrayFingerprint::from(vb);
                    if fa.popcount() <= fb.popcount() {
                        a.clone()
                    } else {
                        b.clone()
                    }
                }
                _ => a.clone(),
            },
            HdrSemiring::Resonance => match (a, b) {
                (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                    let da = (va.density() - 0.5).abs();
                    let db = (vb.density() - 0.5).abs();
                    if da <= db {
                        a.clone()
                    } else {
                        b.clone()
                    }
                }
                _ => a.clone(),
            },
            HdrSemiring::Boolean => match (a, b) {
                (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => HdrScalar::Vector(va.or(vb)),
                (HdrScalar::Bool(ba), HdrScalar::Bool(bb)) => HdrScalar::Bool(*ba || *bb),
                _ => a.clone(),
            },
            HdrSemiring::XorField => match (a, b) {
                (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => HdrScalar::Vector(va.xor(vb)),
                _ => a.clone(),
            },
        }
    }

    fn zero(&self) -> HdrScalar {
        match self {
            HdrSemiring::XorBundle
            | HdrSemiring::BindFirst
            | HdrSemiring::HammingMin
            | HdrSemiring::SimilarityMax
            | HdrSemiring::Resonance
            | HdrSemiring::XorField => HdrScalar::Empty,
            HdrSemiring::Boolean => HdrScalar::Bool(false),
        }
    }

    fn name(&self) -> &str {
        match self {
            HdrSemiring::XorBundle => "XOR_BUNDLE",
            HdrSemiring::BindFirst => "BIND_FIRST",
            HdrSemiring::HammingMin => "HAMMING_MIN",
            HdrSemiring::SimilarityMax => "SIMILARITY_MAX",
            HdrSemiring::Resonance => "RESONANCE",
            HdrSemiring::Boolean => "BOOLEAN",
            HdrSemiring::XorField => "XOR_FIELD",
        }
    }
}

/// Apply a binary operator to two scalars.
pub fn apply_binary_op(op: BinaryOp, a: &HdrScalar, b: &HdrScalar) -> HdrScalar {
    match op {
        BinaryOp::Xor => match (a, b) {
            (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => HdrScalar::Vector(va.xor(vb)),
            _ => HdrScalar::Empty,
        },
        BinaryOp::And => match (a, b) {
            (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => HdrScalar::Vector(va.and(vb)),
            (HdrScalar::Bool(ba), HdrScalar::Bool(bb)) => HdrScalar::Bool(*ba && *bb),
            _ => HdrScalar::Empty,
        },
        BinaryOp::Or => match (a, b) {
            (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => HdrScalar::Vector(va.or(vb)),
            (HdrScalar::Bool(ba), HdrScalar::Bool(bb)) => HdrScalar::Bool(*ba || *bb),
            _ => HdrScalar::Empty,
        },
        BinaryOp::Bundle => match (a, b) {
            (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                HdrScalar::Vector(BitVec::bundle(&[va, vb]))
            }
            _ => HdrScalar::Empty,
        },
        BinaryOp::First => {
            if a.is_empty() {
                b.clone()
            } else {
                a.clone()
            }
        }
        BinaryOp::Second => {
            if b.is_empty() {
                a.clone()
            } else {
                b.clone()
            }
        }
        BinaryOp::HammingDistance => match (a, b) {
            (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                HdrScalar::Float(va.hamming_distance(vb) as f32)
            }
            _ => HdrScalar::Empty,
        },
        BinaryOp::Similarity => match (a, b) {
            (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                let dist = va.hamming_distance(vb);
                HdrScalar::Float(hamming_to_similarity(dist))
            }
            _ => HdrScalar::Empty,
        },
        BinaryOp::FloatAdd => match (a, b) {
            (HdrScalar::Float(fa), HdrScalar::Float(fb)) => HdrScalar::Float(fa + fb),
            _ => HdrScalar::Empty,
        },
        BinaryOp::FloatMul => match (a, b) {
            (HdrScalar::Float(fa), HdrScalar::Float(fb)) => HdrScalar::Float(fa * fb),
            _ => HdrScalar::Empty,
        },
        BinaryOp::FloatMin => match (a, b) {
            (HdrScalar::Float(fa), HdrScalar::Float(fb)) => HdrScalar::Float(fa.min(*fb)),
            _ => HdrScalar::Empty,
        },
        BinaryOp::FloatMax => match (a, b) {
            (HdrScalar::Float(fa), HdrScalar::Float(fb)) => HdrScalar::Float(fa.max(*fb)),
            _ => HdrScalar::Empty,
        },
        BinaryOp::BoolAnd => match (a, b) {
            (HdrScalar::Bool(ba), HdrScalar::Bool(bb)) => HdrScalar::Bool(*ba && *bb),
            _ => HdrScalar::Empty,
        },
        BinaryOp::BoolOr => match (a, b) {
            (HdrScalar::Bool(ba), HdrScalar::Bool(bb)) => HdrScalar::Bool(*ba || *bb),
            _ => HdrScalar::Empty,
        },
    }
}

/// Apply a monoid accumulation to two scalars.
pub fn apply_monoid(op: MonoidOp, a: &HdrScalar, b: &HdrScalar) -> HdrScalar {
    if a.is_empty() {
        return b.clone();
    }
    if b.is_empty() {
        return a.clone();
    }

    match op {
        MonoidOp::Xor => match (a, b) {
            (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => HdrScalar::Vector(va.xor(vb)),
            _ => a.clone(),
        },
        MonoidOp::Or => match (a, b) {
            (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => HdrScalar::Vector(va.or(vb)),
            (HdrScalar::Bool(ba), HdrScalar::Bool(bb)) => HdrScalar::Bool(*ba || *bb),
            _ => a.clone(),
        },
        MonoidOp::And => match (a, b) {
            (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => HdrScalar::Vector(va.and(vb)),
            (HdrScalar::Bool(ba), HdrScalar::Bool(bb)) => HdrScalar::Bool(*ba && *bb),
            _ => a.clone(),
        },
        MonoidOp::BundleMonoid => match (a, b) {
            (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                HdrScalar::Vector(BitVec::bundle(&[va, vb]))
            }
            _ => a.clone(),
        },
        MonoidOp::First => a.clone(),
        MonoidOp::Min => match (a, b) {
            (HdrScalar::Float(fa), HdrScalar::Float(fb)) => HdrScalar::Float(fa.min(*fb)),
            _ => a.clone(),
        },
        MonoidOp::Max => match (a, b) {
            (HdrScalar::Float(fa), HdrScalar::Float(fb)) => HdrScalar::Float(fa.max(*fb)),
            _ => a.clone(),
        },
        MonoidOp::FloatAdd => match (a, b) {
            (HdrScalar::Float(fa), HdrScalar::Float(fb)) => HdrScalar::Float(fa + fb),
            _ => a.clone(),
        },
        MonoidOp::MinPopcount => match (a, b) {
            (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                if va.popcount() <= vb.popcount() {
                    a.clone()
                } else {
                    b.clone()
                }
            }
            _ => a.clone(),
        },
        MonoidOp::BestDensity => match (a, b) {
            (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => {
                let da = (va.density() - 0.5).abs();
                let db = (vb.density() - 0.5).abs();
                if da <= db {
                    a.clone()
                } else {
                    b.clone()
                }
            }
            _ => a.clone(),
        },
        MonoidOp::XorField => match (a, b) {
            (HdrScalar::Vector(va), HdrScalar::Vector(vb)) => HdrScalar::Vector(va.xor(vb)),
            _ => a.clone(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vec(seed: u64) -> HdrScalar {
        HdrScalar::Vector(BitVec::random(seed))
    }

    #[test]
    fn test_xor_bundle_semiring() {
        let sr = HdrSemiring::XorBundle;
        assert_eq!(sr.name(), "XOR_BUNDLE");

        let a = make_vec(1);
        let b = make_vec(2);
        let prod = sr.multiply(&a, &b);
        assert!(prod.as_vector().is_some());

        // Adding zero (Empty) should return the other
        let z = sr.zero();
        let sum = sr.add(&z, &a);
        assert_eq!(sum, a);
    }

    #[test]
    fn test_bind_first_semiring() {
        let sr = HdrSemiring::BindFirst;
        let a = make_vec(10);
        let b = make_vec(20);

        // add returns first non-empty
        let sum = sr.add(&a, &b);
        assert_eq!(sum, a);

        // if first is empty, returns second
        let sum2 = sr.add(&HdrScalar::Empty, &b);
        assert_eq!(sum2, b);
    }

    #[test]
    fn test_hamming_min_semiring() {
        let sr = HdrSemiring::HammingMin;
        let a = make_vec(1);
        let b = make_vec(2);

        // Multiply produces XOR vector, not Float.
        let prod = sr.multiply(&a, &b);
        assert!(prod.as_vector().is_some());

        // Add keeps the vector with smallest popcount (shortest path).
        let sparse = HdrScalar::Vector(BitVec::zero()); // popcount 0
        let dense = HdrScalar::Vector(BitVec::ones()); // popcount 16384
        let sum = sr.add(&sparse, &dense);
        assert_eq!(sum.as_vector().unwrap().popcount(), 0);
    }

    #[test]
    fn test_similarity_max_semiring() {
        let sr = HdrSemiring::SimilarityMax;
        let a = make_vec(1);

        // Same vector XOR => zero vector (perfect similarity).
        let prod = sr.multiply(&a, &a);
        assert!(prod.as_vector().unwrap().is_zero());

        // Add keeps the vector with min popcount (max similarity).
        let close = HdrScalar::Vector(BitVec::zero());
        let far = HdrScalar::Vector(BitVec::ones());
        let sum = sr.add(&close, &far);
        assert_eq!(sum.as_vector().unwrap().popcount(), 0);
    }

    #[test]
    fn test_boolean_semiring() {
        let sr = HdrSemiring::Boolean;
        let t = HdrScalar::Bool(true);
        let f = HdrScalar::Bool(false);

        assert_eq!(sr.multiply(&t, &f), HdrScalar::Bool(false));
        assert_eq!(sr.add(&t, &f), HdrScalar::Bool(true));
        assert_eq!(sr.zero(), HdrScalar::Bool(false));
    }

    #[test]
    fn test_xor_field_semiring() {
        let sr = HdrSemiring::XorField;
        let a = make_vec(5);
        let b = make_vec(6);

        let prod = sr.multiply(&a, &b);
        assert!(prod.as_vector().is_some());

        // XOR add: a + a = zero-ish (XOR of identical vectors)
        let sum = sr.add(&a, &a);
        if let HdrScalar::Vector(v) = &sum {
            assert!(v.is_zero());
        } else {
            panic!("expected vector");
        }
    }

    #[test]
    fn test_resonance_semiring() {
        let sr = HdrSemiring::Resonance;
        let a = make_vec(1);
        let b = make_vec(2);
        let prod = sr.multiply(&a, &b);
        assert!(prod.as_vector().is_some());
    }

    #[test]
    fn test_apply_binary_op_xor() {
        let a = HdrScalar::Vector(BitVec::random(1));
        let b = HdrScalar::Vector(BitVec::random(2));
        let result = apply_binary_op(BinaryOp::Xor, &a, &b);
        assert!(result.as_vector().is_some());
    }

    #[test]
    fn test_apply_binary_op_float() {
        let a = HdrScalar::Float(3.0);
        let b = HdrScalar::Float(4.0);
        assert_eq!(
            apply_binary_op(BinaryOp::FloatAdd, &a, &b),
            HdrScalar::Float(7.0)
        );
        assert_eq!(
            apply_binary_op(BinaryOp::FloatMul, &a, &b),
            HdrScalar::Float(12.0)
        );
        assert_eq!(
            apply_binary_op(BinaryOp::FloatMin, &a, &b),
            HdrScalar::Float(3.0)
        );
        assert_eq!(
            apply_binary_op(BinaryOp::FloatMax, &a, &b),
            HdrScalar::Float(4.0)
        );
    }

    #[test]
    fn test_apply_monoid_empty_identity() {
        let a = HdrScalar::Vector(BitVec::random(1));
        let e = HdrScalar::Empty;
        assert_eq!(apply_monoid(MonoidOp::Xor, &e, &a), a);
        assert_eq!(apply_monoid(MonoidOp::Xor, &a, &e), a);
    }
}
