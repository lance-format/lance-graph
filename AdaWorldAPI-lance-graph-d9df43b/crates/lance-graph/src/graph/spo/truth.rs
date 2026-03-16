// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! NARS-style truth values and gates for SPO edge confidence.
//!
//! Each SPO edge carries a `TruthValue` with frequency (how often the relation
//! holds) and confidence (how certain we are). `TruthGate` thresholds filter
//! query results by minimum truth strength.

/// A NARS-style truth value: (frequency, confidence).
///
/// - `frequency` ∈ [0.0, 1.0]: proportion of positive evidence
/// - `confidence` ∈ [0.0, 1.0]: amount of evidence relative to total possible
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TruthValue {
    pub frequency: f32,
    pub confidence: f32,
}

impl TruthValue {
    /// Create a new truth value with validation.
    pub fn new(frequency: f32, confidence: f32) -> Self {
        Self {
            frequency: frequency.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Full truth: frequency=1.0, confidence=1.0.
    pub fn certain() -> Self {
        Self {
            frequency: 1.0,
            confidence: 1.0,
        }
    }

    /// Unknown truth: frequency=0.5, confidence=0.0.
    pub fn unknown() -> Self {
        Self {
            frequency: 0.5,
            confidence: 0.0,
        }
    }

    /// Expectation: e = c * (f - 0.5) + 0.5
    ///
    /// This is the "expected truth" — a single scalar combining frequency and confidence.
    pub fn expectation(&self) -> f32 {
        self.confidence * (self.frequency - 0.5) + 0.5
    }

    /// Strength: f * c (simple product, used for ranking).
    pub fn strength(&self) -> f32 {
        self.frequency * self.confidence
    }

    /// Revision: combine two truth values with independent evidence.
    pub fn revision(&self, other: &TruthValue) -> TruthValue {
        let k = 1.0; // evidence horizon
        let w1 = self.confidence / (1.0 - self.confidence + f32::EPSILON);
        let w2 = other.confidence / (1.0 - other.confidence + f32::EPSILON);
        let w = w1 + w2;

        let f = if w > f32::EPSILON {
            (w1 * self.frequency + w2 * other.frequency) / w
        } else {
            0.5
        };
        let c = w / (w + k);

        TruthValue::new(f, c)
    }
}

impl Default for TruthValue {
    fn default() -> Self {
        Self::unknown()
    }
}

/// Gate thresholds for filtering SPO query results by truth strength.
///
/// Named thresholds control the minimum expectation required for an edge
/// to pass through a query filter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TruthGate {
    /// Minimum expectation to pass the gate.
    pub min_expectation: f32,
}

impl TruthGate {
    /// Open gate: everything passes (min_expectation = 0.0).
    pub const OPEN: TruthGate = TruthGate {
        min_expectation: 0.0,
    };

    /// Weak gate: expectation > 0.4.
    pub const WEAK: TruthGate = TruthGate {
        min_expectation: 0.4,
    };

    /// Normal gate: expectation > 0.6.
    pub const NORMAL: TruthGate = TruthGate {
        min_expectation: 0.6,
    };

    /// Strong gate: expectation > 0.75.
    pub const STRONG: TruthGate = TruthGate {
        min_expectation: 0.75,
    };

    /// Certain gate: expectation > 0.9.
    pub const CERTAIN: TruthGate = TruthGate {
        min_expectation: 0.9,
    };

    /// Check if a truth value passes this gate.
    pub fn passes(&self, tv: &TruthValue) -> bool {
        tv.expectation() >= self.min_expectation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truth_value_clamp() {
        let tv = TruthValue::new(1.5, -0.3);
        assert_eq!(tv.frequency, 1.0);
        assert_eq!(tv.confidence, 0.0);
    }

    #[test]
    fn test_expectation() {
        let tv = TruthValue::new(0.9, 0.8);
        let e = tv.expectation();
        // e = 0.8 * (0.9 - 0.5) + 0.5 = 0.8 * 0.4 + 0.5 = 0.82
        assert!((e - 0.82).abs() < 0.001);
    }

    #[test]
    fn test_gate_open() {
        let tv = TruthValue::new(0.1, 0.1);
        assert!(TruthGate::OPEN.passes(&tv));
    }

    #[test]
    fn test_gate_strong() {
        let high = TruthValue::new(0.9, 0.8);
        let low = TruthValue::new(0.3, 0.2);
        assert!(TruthGate::STRONG.passes(&high));
        assert!(!TruthGate::STRONG.passes(&low));
    }

    #[test]
    fn test_gate_certain() {
        let very_high = TruthValue::new(0.95, 0.95);
        let high = TruthValue::new(0.9, 0.8);
        assert!(TruthGate::CERTAIN.passes(&very_high));
        // 0.8*(0.9-0.5)+0.5 = 0.82 < 0.9
        assert!(!TruthGate::CERTAIN.passes(&high));
    }

    #[test]
    fn test_revision() {
        let a = TruthValue::new(0.8, 0.5);
        let b = TruthValue::new(0.6, 0.5);
        let revised = a.revision(&b);
        // Combined should have higher confidence
        assert!(revised.confidence > a.confidence);
        // Frequency should be between a and b
        assert!(revised.frequency >= 0.6 && revised.frequency <= 0.8);
    }
}
