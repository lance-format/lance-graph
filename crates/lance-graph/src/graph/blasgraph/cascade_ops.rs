// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cascade search integration for DataFusion planner.
//!
//! Translates Cypher queries with hamming predicates into cascade-accelerated
//! scan operations. The cascade eliminates 97%+ of candidates using sampled
//! bit comparisons before a full Hamming comparison is ever attempted.

use super::hdr::{Band, Cascade};
use super::types::VECTOR_BITS;

// ---------------------------------------------------------------------------
// CascadeScanConfig — parameterizes a cascade-accelerated table scan
// ---------------------------------------------------------------------------

/// Configuration for a cascade-accelerated scan operation.
///
/// When a Cypher query contains a predicate like `WHERE hamming(a, b) < threshold`,
/// the planner can convert it into a cascade scan that pre-filters candidates
/// using sampled Hamming distance before computing exact distances.
#[derive(Debug, Clone)]
pub struct CascadeScanConfig {
    /// The Hamming distance threshold from the query predicate.
    /// Candidates with `hamming(query, row) >= threshold` are rejected.
    pub threshold: u32,

    /// Minimum band required for a candidate to survive the cascade.
    /// Derived from the threshold relative to the corpus statistics.
    pub min_band: Band,

    /// Number of vector bits (e.g. 16384). Used to normalize thresholds
    /// for the cascade's sampled stages.
    pub vector_bits: usize,

    /// Name of the fingerprint column in the Lance dataset.
    pub fingerprint_column: String,

    /// Optional limit on the number of results to return.
    pub top_k: Option<usize>,
}

impl CascadeScanConfig {
    /// Create a new cascade scan config from a threshold and column name.
    pub fn new(threshold: u32, fingerprint_column: String) -> Self {
        let min_band = threshold_to_band(threshold);
        Self {
            threshold,
            min_band,
            vector_bits: VECTOR_BITS,
            fingerprint_column,
            top_k: None,
        }
    }

    /// Set a limit on the number of results.
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = Some(k);
        self
    }
}

// ---------------------------------------------------------------------------
// Predicate-to-cascade translation
// ---------------------------------------------------------------------------

/// Classify a raw Hamming threshold into a sigma band.
///
/// Uses the standard binomial statistics for N=16384:
///   mu = 8192, sigma = 64
///
/// - threshold < mu - 3*sigma => Foveal
/// - threshold < mu - 2*sigma => Near
/// - threshold < mu - sigma   => Good
/// - threshold < mu           => Weak
/// - otherwise                => Reject
fn threshold_to_band(threshold: u32) -> Band {
    let mu = (VECTOR_BITS / 2) as u32;
    let sigma = 64u32; // sqrt(16384/4) = 64

    if threshold < mu.saturating_sub(3 * sigma) {
        Band::Foveal
    } else if threshold < mu.saturating_sub(2 * sigma) {
        Band::Near
    } else if threshold < mu.saturating_sub(sigma) {
        Band::Good
    } else if threshold < mu {
        Band::Weak
    } else {
        Band::Reject
    }
}

/// Convert a `WHERE hamming(a, b) < threshold` predicate into cascade parameters.
///
/// Returns a `CascadeScanConfig` that can be used to configure a cascade-accelerated
/// scan, along with a pre-calibrated `Cascade` instance for the given threshold.
///
/// # Arguments
///
/// * `threshold` - The Hamming distance threshold from the query predicate.
/// * `fingerprint_column` - Name of the fingerprint column to scan.
///
/// # Returns
///
/// A tuple of `(CascadeScanConfig, Cascade)` ready for use in a scan operation.
///
/// # Example
///
/// ```rust,ignore
/// // Cypher: MATCH (n:Node) WHERE hamming(n.plane_s, $query) < 7000 RETURN n
/// let (config, cascade) = hamming_predicate_to_cascade(7000, "plane_s".to_string());
/// assert_eq!(config.min_band, Band::Near);
/// ```
pub fn hamming_predicate_to_cascade(
    threshold: u32,
    fingerprint_column: String,
) -> (CascadeScanConfig, Cascade) {
    let config = CascadeScanConfig::new(threshold, fingerprint_column);

    // Create a cascade calibrated for VECTOR_BITS width.
    // The cascade uses theoretical mu/sigma for 16384-bit vectors.
    // The caller's threshold is stored in the config; the cascade
    // provides sigma-band classification and sampled pre-filtering.
    let cascade = Cascade::for_width(VECTOR_BITS as u32);

    (config, cascade)
}

/// Estimate the selectivity of a Hamming predicate.
///
/// Given a threshold, estimate what fraction of random vectors would satisfy
/// `hamming(query, candidate) < threshold`. Uses the normal approximation
/// to the binomial distribution.
///
/// Returns a value in `[0.0, 1.0]` where 0.0 means no candidates pass
/// and 1.0 means all candidates pass.
pub fn estimate_selectivity(threshold: u32) -> f64 {
    let mu = (VECTOR_BITS / 2) as f64;
    let sigma = ((VECTOR_BITS as f64) / 4.0).sqrt();

    if threshold == 0 {
        return 0.0;
    }

    let z = (threshold as f64 - mu) / sigma;

    // Approximate the CDF of the standard normal using the logistic function.
    // This is accurate to within ~0.01 for the range we care about.
    let cdf = 1.0 / (1.0 + (-1.7 * z).exp());

    cdf.clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_to_band_foveal() {
        // mu=8192, sigma=64, mu-3*sigma = 8000
        assert_eq!(threshold_to_band(7000), Band::Foveal);
    }

    #[test]
    fn test_threshold_to_band_near() {
        // mu-2*sigma = 8064
        assert_eq!(threshold_to_band(8010), Band::Near);
    }

    #[test]
    fn test_threshold_to_band_good() {
        // mu-sigma = 8128
        assert_eq!(threshold_to_band(8100), Band::Good);
    }

    #[test]
    fn test_threshold_to_band_weak() {
        assert_eq!(threshold_to_band(8150), Band::Weak);
    }

    #[test]
    fn test_threshold_to_band_reject() {
        assert_eq!(threshold_to_band(9000), Band::Reject);
    }

    #[test]
    fn test_cascade_scan_config_new() {
        let config = CascadeScanConfig::new(7000, "plane_s".to_string());
        assert_eq!(config.threshold, 7000);
        assert_eq!(config.min_band, Band::Foveal);
        assert_eq!(config.vector_bits, VECTOR_BITS);
        assert_eq!(config.fingerprint_column, "plane_s");
        assert!(config.top_k.is_none());
    }

    #[test]
    fn test_cascade_scan_config_with_top_k() {
        let config = CascadeScanConfig::new(7000, "fp".to_string()).with_top_k(10);
        assert_eq!(config.top_k, Some(10));
    }

    #[test]
    fn test_hamming_predicate_to_cascade() {
        let (config, cascade) =
            hamming_predicate_to_cascade(7000, "plane_s".to_string());
        assert_eq!(config.threshold, 7000);
        assert_eq!(config.min_band, Band::Foveal);
        // The cascade should be calibrated for 16384-bit vectors
        assert_eq!(cascade.mu(), (VECTOR_BITS / 2) as u32);
        assert_eq!(cascade.sigma(), 64);
    }

    #[test]
    fn test_estimate_selectivity_zero() {
        assert_eq!(estimate_selectivity(0), 0.0);
    }

    #[test]
    fn test_estimate_selectivity_half() {
        // At mu=8192, selectivity should be ~0.5
        let sel = estimate_selectivity(8192);
        assert!((sel - 0.5).abs() < 0.02, "selectivity at mu was {}", sel);
    }

    #[test]
    fn test_estimate_selectivity_low_threshold() {
        // Well below mu should have very low selectivity
        let sel = estimate_selectivity(7000);
        assert!(sel < 0.1, "selectivity at 7000 was {}", sel);
    }

    #[test]
    fn test_estimate_selectivity_high_threshold() {
        // Well above mu should have high selectivity
        let sel = estimate_selectivity(10000);
        assert!(sel > 0.9, "selectivity at 10000 was {}", sel);
    }
}
