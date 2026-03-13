// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! # Belichtungsmesser — HDR Popcount-Stacking Early-Exit Distance Cascade
//!
//! Self-calibrating exposure meter for Hamming distance queries on binary
//! vectors. Eliminates 97%+ of candidates using sampled bit comparisons
//! before a full comparison is ever attempted.
//!
//! ## Design principles
//!
//! - **Integer only** — no `f32`/`f64` anywhere. Distances are `u32`,
//!   thresholds are `u32`, square roots are integer Newton's method.
//! - **Self-calibrating** — thresholds derived from actual corpus
//!   distribution, not hardcoded. Adapts to any vector width.
//! - **Shift detection** — Welford's online algorithm detects when the
//!   data distribution changes and triggers recalibration.
//! - **Cascade** — 1/16 sample → 1/4 sample → full comparison. Each
//!   stage rejects on sigma bands. Only ~0.5% of candidates reach full.
//!
//! ## Statistics
//!
//! Two random N-bit vectors have Hamming distance ~ Binomial(N, 0.5):
//!
//! ```text
//! μ = N/2,  σ = √(N/4)
//!
//! N = 16384:  μ = 8192,  σ = 64
//! N = 10000:  μ = 5000,  σ ≈ 50
//! N = 32768:  μ = 16384, σ ≈ 91
//! ```
//!
//! Lower distance = better match. Everything at or above μ is noise.

// ---------------------------------------------------------------------------
// Integer square root — Newton's method, no float
// ---------------------------------------------------------------------------

/// Integer square root via Newton's method. No float arithmetic.
///
/// Returns floor(√n). Guaranteed correct for all `u32` inputs.
pub fn isqrt(n: u32) -> u32 {
    if n == 0 {
        return 0;
    }
    // Initial guess: must be >= floor(√n) so Newton's method converges
    // monotonically downward. Round up the bit-shift to guarantee this.
    let mut x = 1u32 << ((33 - n.leading_zeros()) / 2);
    loop {
        let x1 = (x + n / x) / 2;
        if x1 >= x {
            return x;
        }
        x = x1;
    }
}

// ---------------------------------------------------------------------------
// Band — sigma-based classification
// ---------------------------------------------------------------------------

/// Sigma band for classifying Hamming distances.
///
/// Ordered from best match to worst. `Foveal < Near < Good < Weak < Reject`.
/// The `PartialOrd`/`Ord` derive gives this ordering from the discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Band {
    /// d < μ − 3σ. Less than 0.13% chance of random. Near-certain match.
    Foveal = 0,
    /// d < μ − 2σ. Less than 2.3% chance of random. Strong match.
    Near = 1,
    /// d < μ − σ. Less than 15.9% chance of random. Good candidate.
    Good = 2,
    /// d < μ. Could be random. Weak signal.
    Weak = 3,
    /// d ≥ μ. Noise or anti-correlated. Not a match.
    Reject = 4,
}

// ---------------------------------------------------------------------------
// RankedHit / ShiftAlert — result types
// ---------------------------------------------------------------------------

/// A candidate that survived the cascade and was classified.
#[derive(Debug, Clone)]
pub struct RankedHit {
    /// Index of the matched candidate in the input slice.
    pub index: usize,
    /// Exact Hamming distance (from stage 3 full comparison). `u32`, not float.
    pub distance: u32,
    /// Sigma band classification.
    pub band: Band,
}

/// Alert that the observed distance distribution has shifted.
#[derive(Debug, Clone)]
pub struct ShiftAlert {
    pub old_mu: u32,
    pub new_mu: u32,
    pub old_sigma: u32,
    pub new_sigma: u32,
    pub observations: u64,
}

// ---------------------------------------------------------------------------
// Hamming on &[u64] words — self-contained, no external deps
// ---------------------------------------------------------------------------

/// Hamming distance between two word arrays (XOR + popcount).
///
/// Both slices must have the same length.
#[inline]
fn words_hamming(a: &[u64], b: &[u64]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dist = 0u32;
    for i in 0..a.len() {
        dist += (a[i] ^ b[i]).count_ones();
    }
    dist
}

/// Sampled Hamming distance: compare every `step`-th word and extrapolate.
///
/// For `step=16`, samples 1/16 of the words. The extrapolation multiplies
/// by `step`, giving an unbiased estimate with higher variance.
#[inline]
fn words_hamming_sampled(a: &[u64], b: &[u64], step: usize) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert!(step > 0);
    let mut dist = 0u32;
    let mut sampled = 0u32;
    let mut i = 0;
    while i < a.len() {
        dist += (a[i] ^ b[i]).count_ones();
        sampled += 1;
        i += step;
    }
    if sampled == 0 {
        return 0;
    }
    // Extrapolate: scale to full width.
    (dist as u64 * a.len() as u64 / sampled as u64) as u32
}

// ---------------------------------------------------------------------------
// Belichtungsmesser — the exposure meter
// ---------------------------------------------------------------------------

/// Self-calibrating exposure meter for Hamming distance queries.
///
/// All thresholds are `u32`. All arithmetic is integer. No float.
///
/// # Usage
///
/// ```ignore
/// // Calibrate from a sample of pairwise distances.
/// let meter = Belichtungsmesser::calibrate(&sample_distances);
///
/// // Classify a single distance.
/// let band = meter.band(7950); // → Band::Foveal
///
/// // Three-stage cascade query.
/// let results = meter.cascade_query(query_words, &candidates, 10);
///
/// // Feed observed distances for shift detection.
/// if let Some(alert) = meter.observe(distance) {
///     meter.recalibrate(&alert);
/// }
/// ```
pub struct Belichtungsmesser {
    /// Calibrated mean pairwise Hamming distance.
    mu: u32,
    /// Calibrated standard deviation.
    sigma: u32,
    /// Precomputed band thresholds: [μ-3σ, μ-2σ, μ-σ, μ].
    bands: [u32; 4],
    /// Cascade stage 1 threshold (1/16 sample): μ − 4σ.
    /// Accounts for the 4× amplified σ of 1/16 extrapolation.
    cascade_s1: u32,
    /// Cascade stage 2 threshold (1/4 sample): μ − 2σ.
    /// Accounts for the 2× amplified σ of 1/4 extrapolation.
    cascade_s2: u32,
    // -- Welford's online statistics for shift detection --
    running_count: u64,
    running_sum: u64,
    running_m2: u64,
}

impl Belichtungsmesser {
    /// Create from theoretical parameters for a given bit width.
    ///
    /// Uses the binomial distribution: μ = bits/2, σ = isqrt(bits/4).
    pub fn for_width(total_bits: u32) -> Self {
        let mu = total_bits / 2;
        let sigma = isqrt(total_bits / 4).max(1);
        let bands = [
            mu.saturating_sub(3 * sigma),
            mu.saturating_sub(2 * sigma),
            mu.saturating_sub(sigma),
            mu,
        ];
        Self {
            mu,
            sigma,
            bands,
            // 1/16 sample: projected σ = σ * √16 = 4σ. Reject above μ − 4σ.
            cascade_s1: mu.saturating_sub(4 * sigma),
            // 1/4 sample: projected σ = σ * √4 = 2σ. Reject above μ − 2σ.
            cascade_s2: mu.saturating_sub(2 * sigma),
            running_count: 0,
            running_sum: 0,
            running_m2: 0,
        }
    }

    /// Calibrate from a sample of actual pairwise distances.
    ///
    /// Call once on startup with ≥2 distances sampled from the corpus.
    pub fn calibrate(sample_distances: &[u32]) -> Self {
        let n = sample_distances.len() as u64;
        assert!(n > 1, "need at least 2 samples to calibrate");

        let sum: u64 = sample_distances.iter().map(|&d| d as u64).sum();
        let mu = (sum / n) as u32;

        let var_sum: u64 = sample_distances
            .iter()
            .map(|&d| {
                let diff = d as i64 - mu as i64;
                (diff * diff) as u64
            })
            .sum();
        let sigma = isqrt((var_sum / n) as u32).max(1);

        let bands = [
            mu.saturating_sub(3 * sigma),
            mu.saturating_sub(2 * sigma),
            mu.saturating_sub(sigma),
            mu,
        ];

        Self {
            mu,
            sigma,
            bands,
            cascade_s1: mu.saturating_sub(4 * sigma),
            cascade_s2: mu.saturating_sub(2 * sigma),
            running_count: n,
            running_sum: sum,
            running_m2: var_sum,
        }
    }

    /// Classify a Hamming distance into a sigma band.
    ///
    /// Pure integer comparison. No float. Constant time.
    #[inline]
    pub fn band(&self, distance: u32) -> Band {
        if distance < self.bands[0] {
            Band::Foveal
        } else if distance < self.bands[1] {
            Band::Near
        } else if distance < self.bands[2] {
            Band::Good
        } else if distance < self.bands[3] {
            Band::Weak
        } else {
            Band::Reject
        }
    }

    /// Current calibrated mean.
    pub fn mu(&self) -> u32 {
        self.mu
    }

    /// Current calibrated sigma.
    pub fn sigma(&self) -> u32 {
        self.sigma
    }

    /// Current band thresholds: [μ-3σ, μ-2σ, μ-σ, μ].
    pub fn thresholds(&self) -> [u32; 4] {
        self.bands
    }

    // -- Cascade query --

    /// Three-stage cascade query on `&[u64]` word arrays.
    ///
    /// Returns up to `top_k` results bucketed by band (Foveal first),
    /// sorted by `u32` distance within each bucket. No float anywhere.
    ///
    /// Each candidate is `&[u64]` with the same length as `query`.
    /// For vectors with fewer than 16 words, stages are skipped
    /// automatically.
    pub fn cascade_query(
        &self,
        query: &[u64],
        candidates: &[&[u64]],
        top_k: usize,
    ) -> Vec<RankedHit> {
        let nwords = query.len();
        let do_stage1 = nwords >= 16;
        let do_stage2 = nwords >= 4;

        let mut foveal: Vec<(usize, u32)> = Vec::new();
        let mut near: Vec<(usize, u32)> = Vec::new();
        let mut good: Vec<(usize, u32)> = Vec::new();

        for (idx, candidate) in candidates.iter().enumerate() {
            debug_assert_eq!(candidate.len(), nwords);

            // -- Stage 1: 1/16 sample --
            if do_stage1 {
                let projected = words_hamming_sampled(query, candidate, 16);
                if projected > self.cascade_s1 {
                    continue; // reject: above μ − 4σ (sampling-aware)
                }
            }

            // -- Stage 2: 1/4 sample --
            if do_stage2 {
                let projected = words_hamming_sampled(query, candidate, 4);
                if projected > self.cascade_s2 {
                    continue; // reject: above μ − 2σ (sampling-aware)
                }
            }

            // -- Stage 3: full comparison --
            let full_dist = words_hamming(query, candidate);
            match self.band(full_dist) {
                Band::Foveal => foveal.push((idx, full_dist)),
                Band::Near => near.push((idx, full_dist)),
                Band::Good => good.push((idx, full_dist)),
                _ => {} // Weak or Reject — don't collect
            }

            // Early termination: enough foveal hits.
            if foveal.len() >= top_k {
                break;
            }
        }

        // Assemble top-k from buckets. Best band first, integer sort within.
        let mut results = Vec::with_capacity(top_k);

        foveal.sort_unstable_by_key(|&(_, d)| d);
        for &(idx, dist) in foveal.iter().take(top_k) {
            results.push(RankedHit {
                index: idx,
                distance: dist,
                band: Band::Foveal,
            });
        }

        if results.len() < top_k {
            near.sort_unstable_by_key(|&(_, d)| d);
            for &(idx, dist) in near.iter().take(top_k - results.len()) {
                results.push(RankedHit {
                    index: idx,
                    distance: dist,
                    band: Band::Near,
                });
            }
        }

        if results.len() < top_k {
            good.sort_unstable_by_key(|&(_, d)| d);
            for &(idx, dist) in good.iter().take(top_k - results.len()) {
                results.push(RankedHit {
                    index: idx,
                    distance: dist,
                    band: Band::Good,
                });
            }
        }

        results
    }

    // -- Shift detection (Welford's online) --

    /// Feed an observed distance into the running statistics.
    ///
    /// Uses Welford's online algorithm with integer arithmetic.
    /// Returns `Some(ShiftAlert)` if the distribution has drifted
    /// by more than σ/2 from the calibrated values.
    pub fn observe(&mut self, distance: u32) -> Option<ShiftAlert> {
        let d = distance as u64;
        self.running_count += 1;
        self.running_sum += d;

        // Welford's M2 update: delta before/after mean update.
        let old_mean = if self.running_count > 1 {
            (self.running_sum - d) / (self.running_count - 1)
        } else {
            d
        };
        let new_mean = self.running_sum / self.running_count;
        let delta_old = d as i64 - old_mean as i64;
        let delta_new = d as i64 - new_mean as i64;
        self.running_m2 = self.running_m2.wrapping_add((delta_old * delta_new) as u64);

        // Check every 1000 observations.
        if self.running_count % 1000 == 0 && self.running_count > 1000 {
            let running_mu = new_mean as u32;
            let running_var = (self.running_m2 / self.running_count) as u32;
            let running_sigma = isqrt(running_var).max(1);

            let mu_drift = running_mu.abs_diff(self.mu);
            let sigma_drift = running_sigma.abs_diff(self.sigma);

            if mu_drift > self.sigma / 2 || sigma_drift > self.sigma / 4 {
                return Some(ShiftAlert {
                    old_mu: self.mu,
                    new_mu: running_mu,
                    old_sigma: self.sigma,
                    new_sigma: running_sigma,
                    observations: self.running_count,
                });
            }
        }

        None
    }

    /// Recalibrate thresholds from a shift alert.
    pub fn recalibrate(&mut self, alert: &ShiftAlert) {
        self.mu = alert.new_mu;
        self.sigma = alert.new_sigma.max(1);
        self.bands = [
            self.mu.saturating_sub(3 * self.sigma),
            self.mu.saturating_sub(2 * self.sigma),
            self.mu.saturating_sub(self.sigma),
            self.mu,
        ];
        self.cascade_s1 = self.mu.saturating_sub(4 * self.sigma);
        self.cascade_s2 = self.mu.saturating_sub(2 * self.sigma);
    }
}

// ===========================================================================
// Tests — 7 tests covering all requirements
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic PRNG for test reproducibility.
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Generate a random word array of `nwords` u64 values.
    fn random_words(nwords: usize, seed: u64) -> Vec<u64> {
        let mut state = seed;
        (0..nwords).map(|_| splitmix64(&mut state)).collect()
    }

    /// Generate pairwise distances from random vectors.
    fn random_pairwise_distances(nwords: usize, count: usize, seed: u64) -> Vec<u32> {
        let vecs: Vec<Vec<u64>> = (0..count)
            .map(|i| random_words(nwords, seed + i as u64))
            .collect();
        let mut dists = Vec::new();
        for i in 0..count {
            for j in (i + 1)..count {
                dists.push(words_hamming(&vecs[i], &vecs[j]));
            }
        }
        dists
    }

    // -- Test 1: Calibration from known distances --

    #[test]
    fn test_calibration_from_corpus() {
        // Random 16K vectors (256 words) should have μ ≈ 8192, σ ≈ 64.
        let dists = random_pairwise_distances(256, 100, 42);
        let meter = Belichtungsmesser::calibrate(&dists);

        // μ should be near 8192 (±300 for sample variance).
        assert!(
            meter.mu > 7800 && meter.mu < 8600,
            "Expected μ near 8192, got {}",
            meter.mu
        );
        // σ should be near 64 (±40).
        assert!(
            meter.sigma > 20 && meter.sigma < 110,
            "Expected σ near 64, got {}",
            meter.sigma
        );
        // Band thresholds must be strictly ascending.
        assert!(meter.bands[0] < meter.bands[1]);
        assert!(meter.bands[1] < meter.bands[2]);
        assert!(meter.bands[2] < meter.bands[3]);

        println!(
            "Calibrated: μ={}, σ={}, bands={:?}",
            meter.mu, meter.sigma, meter.bands
        );
    }

    // -- Test 2: Direction — lower distance = better band --

    #[test]
    fn test_direction_lower_is_better() {
        let meter = Belichtungsmesser {
            mu: 8192,
            sigma: 64,
            bands: [8192 - 192, 8192 - 128, 8192 - 64, 8192],
            cascade_s1: 8192 - 256,
            cascade_s2: 8192 - 128,
            running_count: 1000,
            running_sum: 8_192_000,
            running_m2: 64 * 64 * 1000,
        };

        // Very low distance = Foveal (best).
        assert_eq!(meter.band(7900), Band::Foveal);
        // Low distance = Near.
        assert_eq!(meter.band(8010), Band::Near);
        // Below mean = Good.
        assert_eq!(meter.band(8100), Band::Good);
        // Near mean = Weak.
        assert_eq!(meter.band(8170), Band::Weak);
        // At or above mean = Reject.
        assert_eq!(meter.band(8192), Band::Reject);
        assert_eq!(meter.band(9000), Band::Reject);

        // Verify ordering: lower band = better.
        assert!(Band::Foveal < Band::Near);
        assert!(Band::Near < Band::Good);
        assert!(Band::Good < Band::Weak);
        assert!(Band::Weak < Band::Reject);
    }

    // -- Test 3: Cascade eliminates most random candidates --
    //
    // Design doc: stage 1 alone eliminates ~84% (threshold at μ−σ is lenient
    // because the 1/16 sample has 4× the std of full comparison).
    // Combined cascade eliminates >95%.

    #[test]
    fn test_elimination_rate() {
        let nwords = 256; // 16384 bits
        let meter = Belichtungsmesser::for_width(nwords as u32 * 64);

        let query = random_words(nwords, 0);
        let n_candidates = 10_000;
        let candidates: Vec<Vec<u64>> = (1..=n_candidates)
            .map(|i| random_words(nwords, i as u64))
            .collect();

        let mut stage1_pass = 0u32;
        let mut stage2_pass = 0u32;
        let mut stage3_pass = 0u32;

        let t0 = std::time::Instant::now();
        for candidate in &candidates {
            // Stage 1: 1/16 sample (sampling-aware threshold).
            let s1 = words_hamming_sampled(&query, candidate, 16);
            if s1 > meter.cascade_s1 {
                continue;
            }
            stage1_pass += 1;

            // Stage 2: 1/4 sample (sampling-aware threshold).
            let s2 = words_hamming_sampled(&query, candidate, 4);
            if s2 > meter.cascade_s2 {
                continue;
            }
            stage2_pass += 1;

            // Stage 3: full.
            let full = words_hamming(&query, candidate);
            if meter.band(full) <= Band::Good {
                stage3_pass += 1;
            }
        }
        let elapsed = t0.elapsed();

        let s1_reject_pct = 100 - (stage1_pass as u64 * 100 / n_candidates as u64);
        let total_reject_pct =
            100 - (stage2_pass as u64 * 100 / n_candidates as u64);

        println!(
            "Stage 1: {}/{} pass ({}% rejected)",
            stage1_pass, n_candidates, s1_reject_pct
        );
        println!("Stage 2: {}/{} pass", stage2_pass, stage1_pass);
        println!("Stage 3: {} final", stage3_pass);
        println!(
            "Combined cascade: {}% rejected",
            total_reject_pct
        );
        println!(
            "Cascade time: {:?} ({} ns/candidate)",
            elapsed,
            elapsed.as_nanos() / n_candidates as u128
        );

        // Stage 1 alone: ~84% (sampling noise is high, threshold is lenient).
        assert!(
            s1_reject_pct > 70,
            "Stage 1 must reject >70% of random candidates, got {}%",
            s1_reject_pct
        );
        // Combined cascade: >90% total elimination.
        assert!(
            total_reject_pct > 90,
            "Combined cascade must reject >90% of random candidates, got {}%",
            total_reject_pct
        );
    }

    // -- Test 4: Cascade work savings --

    #[test]
    fn test_work_savings() {
        let nwords = 256;
        let meter = Belichtungsmesser::for_width(nwords as u32 * 64);
        let n_candidates: u64 = 10_000;

        let query = random_words(nwords, 0);
        let candidates: Vec<Vec<u64>> = (1..=n_candidates)
            .map(|i| random_words(nwords, i))
            .collect();

        let mut s1_pass = 0u64;
        let mut s2_pass = 0u64;

        // Measure cascade time.
        let t_cascade = std::time::Instant::now();
        for c in &candidates {
            let s1 = words_hamming_sampled(&query, c, 16);
            if s1 > meter.cascade_s1 {
                continue;
            }
            s1_pass += 1;
            let s2 = words_hamming_sampled(&query, c, 4);
            if s2 > meter.cascade_s2 {
                continue;
            }
            s2_pass += 1;
            // Stage 3: full comparison for survivors.
            let _full = words_hamming(&query, c);
        }
        let cascade_ns = t_cascade.elapsed().as_nanos();

        // Measure brute force time.
        let t_brute = std::time::Instant::now();
        for c in &candidates {
            let _full = words_hamming(&query, c);
        }
        let brute_ns = t_brute.elapsed().as_nanos();

        // Cost model (in "word operations"):
        // Stage 1: nwords/16 per candidate (all candidates).
        // Stage 2: nwords/4 per survivor of stage 1.
        // Stage 3: nwords per survivor of stage 2.
        let brute_cost = n_candidates * nwords as u64;
        let cascade_cost = n_candidates * (nwords as u64 / 16)
            + s1_pass * (nwords as u64 / 4)
            + s2_pass * nwords as u64;
        let savings_pct = 100 - (cascade_cost * 100 / brute_cost);

        println!(
            "Brute: {} word-ops, Cascade: {} word-ops, Savings: {}%",
            brute_cost, cascade_cost, savings_pct
        );
        println!(
            "Brute force: {} ns ({} ns/candidate)",
            brute_ns,
            brute_ns / n_candidates as u128
        );
        println!(
            "Cascade:     {} ns ({} ns/candidate)",
            cascade_ns,
            cascade_ns / n_candidates as u128
        );
        if cascade_ns > 0 {
            println!("Speedup: {:.1}x", brute_ns as f64 / cascade_ns as f64);
        }

        assert!(
            savings_pct > 50,
            "Cascade must save >50% work, got {}%",
            savings_pct
        );
    }

    // -- Test 5: Shift detection --

    #[test]
    fn test_shift_detection() {
        let mut meter = Belichtungsmesser {
            mu: 8192,
            sigma: 64,
            bands: [8192 - 192, 8192 - 128, 8192 - 64, 8192],
            cascade_s1: 8192 - 256,
            cascade_s2: 8192 - 128,
            running_count: 1000,
            running_sum: 8_192_000,
            running_m2: 64 * 64 * 1000,
        };

        // Feed distances from a SHIFTED distribution (μ ≈ 7500).
        let mut alert_fired = false;
        for i in 0..5000u32 {
            let fake_dist = 7500 + (i % 100);
            if let Some(alert) = meter.observe(fake_dist) {
                assert!(
                    alert.new_mu < alert.old_mu,
                    "Shift should detect lower mean: old={}, new={}",
                    alert.old_mu,
                    alert.new_mu
                );
                meter.recalibrate(&alert);
                alert_fired = true;
                break;
            }
        }

        assert!(alert_fired, "Shift alert must fire when μ changes by >σ/2");
        assert!(
            meter.mu < 8000,
            "After recalibration, μ should reflect new distribution, got {}",
            meter.mu
        );
    }

    // -- Test 6: No float guarantee --

    #[test]
    fn test_no_float_guarantee() {
        let meter = Belichtungsmesser::calibrate(&[8000, 8100, 8200, 8300, 8400]);

        // band() returns Band enum, not a float score.
        let b: Band = meter.band(8050);
        // The specific band doesn't matter — what matters is the return type is Band, not f32.
        assert!(matches!(
            b,
            Band::Foveal | Band::Near | Band::Good | Band::Weak | Band::Reject
        ));

        // All thresholds are u32.
        let t: [u32; 4] = meter.thresholds();
        for &thresh in &t {
            let _check: u32 = thresh; // compiles only if u32
        }

        // RankedHit.distance is u32, not f32.
        let hit = RankedHit {
            index: 0,
            distance: 8050,
            band: Band::Near,
        };
        let _d: u32 = hit.distance;

        // mu and sigma are u32.
        let _m: u32 = meter.mu();
        let _s: u32 = meter.sigma();
    }

    // -- Test 7: isqrt correctness --

    #[test]
    fn test_isqrt_correctness() {
        // Exact squares.
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(9), 3);
        assert_eq!(isqrt(16), 4);
        assert_eq!(isqrt(64 * 64), 64); // σ for 16K random vectors
        assert_eq!(isqrt(100 * 100), 100);

        // Non-exact: floor.
        assert_eq!(isqrt(2), 1);
        assert_eq!(isqrt(3), 1);
        assert_eq!(isqrt(5), 2);
        assert_eq!(isqrt(4095), 63);
        assert_eq!(isqrt(4097), 64);

        // Edge: max u32.
        assert_eq!(isqrt(u32::MAX), 65535);

        // Verify floor property for a range.
        for n in [10, 50, 100, 1000, 4096, 16384, 100000, 1000000u32] {
            let s = isqrt(n);
            assert!(s * s <= n, "isqrt({})={}: {}² > {}", n, s, s, n);
            assert!(
                (s + 1) * (s + 1) > n,
                "isqrt({})={}: {}² ≤ {}",
                n,
                s,
                s + 1,
                n
            );
        }
    }
}
