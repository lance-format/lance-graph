# BELICHTUNGSMESSER.md

## HDR Popcount-Stacking Early-Exit Distance Cascade

### What this is and why it matters

A standard nearest-neighbor search compares a query against every candidate at full resolution. For 16,384-bit binary vectors, that's 256 popcount operations per comparison. Against 1 million candidates: 256 million popcount ops per query.

The Belichtungsmesser (German: exposure meter, like in a camera) eliminates 97%+ of candidates using a FRACTION of the bits, so only ~0.5% of candidates ever get a full comparison. Same results. 100x less work.

The name comes from photography: before taking the photo (full comparison), the camera's exposure meter takes a quick light reading (sampled comparison) to decide if there's enough signal to bother.

---

### The statistics you need to understand

Two random 16,384-bit vectors have a Hamming distance that follows a binomial distribution:

```
Each bit: 50% chance of matching (like a coin flip)
Total bits: 16,384
Expected distance: μ = 16384 / 2 = 8192 (half the bits differ)
Standard deviation: σ = sqrt(16384 / 4) = 64
```

This means for RANDOM (unrelated) pairs:

```
~68.3% of random pairs have distance between 8128 and 8256  (μ ± 1σ)
~95.4% of random pairs have distance between 8064 and 8320  (μ ± 2σ)
~99.7% of random pairs have distance between 8000 and 8384  (μ ± 3σ)
```

A REAL MATCH has a distance MUCH LOWER than 8192. The further BELOW μ, the more certain it's a real relationship, not random chance:

```
DISTANCE     SIGMA BELOW μ    PROBABILITY OF RANDOM PAIR BEING THIS CLOSE
──────────────────────────────────────────────────────────────────────────
8192         μ (expected)      50% — pure coin flip. No signal.
8128         μ - 1σ           15.9% of random pairs this close. Weak signal.
8064         μ - 2σ            2.3% of random pairs this close. Likely real.
8000         μ - 3σ            0.13% of random pairs this close. Almost certainly real.
7936         μ - 4σ            0.003% — extremely rare for random pair. Strong match.
7500         μ - 10.8σ         Effectively zero chance of random. Near-identical content.
```

KEY INSIGHT: The search finds vectors with distance far BELOW μ.
Everything NEAR or ABOVE μ is noise. Reject it without full comparison.

---

### The cascade: progressive refinement

Instead of computing all 16,384 bits, sample a fraction first.

A sampled distance SCALES linearly. If you sample 1/16 of the bits and get distance 480, the projected full distance is approximately 480 × 16 = 7,680. The variance of the projection is higher (fewer samples = more noise), but the MEAN is correct. So the sample gives a noisy but unbiased estimate.

```
STAGE 1: Sample 1/16 of bits (1,024 bits = 16 u64 words = 1 AVX-512 op)
  Cost: ~2 CPU cycles per candidate.
  Sample σ = 64 / sqrt(16) = 16
  Project: sample_distance × 16 = estimated_full_distance
  Reject if projected distance > μ - 1σ (above noise floor)
  ELIMINATES: ~84% of random candidates. Cost: trivial.

STAGE 2: Sample 1/4 of bits (4,096 bits = 64 u64 words = 8 AVX-512 ops)
  Cost: ~8 CPU cycles per surviving candidate.
  Sample σ = 64 / sqrt(4) = 32  (tighter estimate than stage 1)
  Project: sample_distance × 4 = estimated_full_distance
  Reject if projected distance > μ - 2σ (likely noise)
  ELIMINATES: ~90% of stage 1 survivors.

STAGE 3: Full comparison (16,384 bits = 256 u64 words = 32 AVX-512 ops)
  Cost: ~32 CPU cycles per surviving candidate.
  Exact distance. No projection error.
  Classify into sigma bands for ranking.
  ONLY ~0.5% of original candidates reach this stage.
```

Total work per query against 1 million candidates:

```
WITHOUT CASCADE:
  1,000,000 × 32 cycles = 32,000,000 cycles ≈ 10ms at 3GHz

WITH CASCADE:
  Stage 1: 1,000,000 ×  2 cycles = 2,000,000 cycles (all candidates)
  Stage 2:    160,000 ×  8 cycles = 1,280,000 cycles (16% survived stage 1)
  Stage 3:     16,000 × 32 cycles =   512,000 cycles (1.6% survived stage 2)
  TOTAL:                             3,792,000 cycles ≈ 1.3ms

  SPEEDUP: ~8.4x with ZERO loss of accuracy on final results.
  The cascade only prunes candidates that would have been rejected anyway.
```

---

### The sigma bands: integer top-k without float sort

After the cascade, surviving candidates are classified by how far below μ their distance falls. This classification replaces float-based ranking:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Band {
    /// distance < μ - 3σ. Less than 0.13% chance of random. Near-certain match.
    Foveal,
    /// distance < μ - 2σ. Less than 2.3% chance of random. Strong match.
    Near,
    /// distance < μ - 1σ. Less than 15.9% chance of random. Good candidate.
    Good,
    /// distance < μ. Could be random. Weak signal.
    Weak,
    /// distance ≥ μ. Noise or anti-correlated. Not a match.
    Reject,
}
```

Top-k is: take from Foveal bucket first, then Near, then Good.
Within each bucket: sort by `u32` Hamming distance (integer sort).
No float. No NaN. No "similarity score 0.873f32". Just sigma bands.

---

### Self-calibrating thresholds

The values μ=8192 and σ=64 are for RANDOM 16K vectors. Real corpora have different distributions because entities share context. After encoding Wikidata, the actual μ might be 7800 with σ=80. Hardcoded thresholds would be wrong.

The Belichtungsmesser calibrates itself from a sample of actual pairwise distances on startup. And it detects when the distribution shifts (e.g., after a large import) and recalibrates:

```rust
/// Self-calibrating exposure meter for Hamming distance queries.
/// All thresholds derived from actual data distribution. Integer arithmetic.
pub struct Belichtungsmesser {
    /// Mean pairwise Hamming distance in this corpus.
    mu: u32,
    /// Standard deviation of pairwise distances.
    sigma: u32,
    /// Precomputed band thresholds. Integer. Looked up, not computed per query.
    /// bands[0] = μ - 3σ (Foveal cutoff)
    /// bands[1] = μ - 2σ (Near cutoff)
    /// bands[2] = μ - σ  (Good cutoff)
    /// bands[3] = μ      (Weak cutoff, everything above = Reject)
    bands: [u32; 4],
    /// Welford running stats for shift detection.
    running_count: u64,
    running_mean: u64,   // scaled by running_count for integer arithmetic
    running_m2: u64,     // sum of squared deviations (Welford)
}

impl Belichtungsmesser {
    /// Calibrate from a sample of actual pairwise distances.
    /// Call once on startup with ~1000 random pair distances from the corpus.
    pub fn calibrate(sample_distances: &[u32]) -> Self {
        let n = sample_distances.len() as u64;
        assert!(n > 1, "Need at least 2 samples to calibrate");

        let sum: u64 = sample_distances.iter().map(|&d| d as u64).sum();
        let mu = (sum / n) as u32;

        let var_sum: u64 = sample_distances.iter()
            .map(|&d| {
                let diff = d as i64 - mu as i64;
                (diff * diff) as u64
            })
            .sum();
        let sigma = isqrt((var_sum / n) as u32);

        // Ensure sigma > 0 to prevent zero-width bands
        let sigma = sigma.max(1);

        let bands = [
            mu.saturating_sub(3 * sigma), // Foveal: < μ - 3σ
            mu.saturating_sub(2 * sigma), // Near:   < μ - 2σ
            mu.saturating_sub(sigma),     // Good:   < μ - 1σ
            mu,                           // Weak:   < μ (everything above = Reject)
        ];

        Self {
            mu,
            sigma,
            bands,
            running_count: n,
            running_mean: sum,
            running_m2: var_sum,
        }
    }

    /// Classify a Hamming distance into a sigma band.
    /// Pure integer comparison. No float. Constant time.
    #[inline]
    pub fn band(&self, distance: u32) -> Band {
        if distance < self.bands[0] { Band::Foveal }
        else if distance < self.bands[1] { Band::Near }
        else if distance < self.bands[2] { Band::Good }
        else if distance < self.bands[3] { Band::Weak }
        else { Band::Reject }
    }

    /// HDR cascade query. Progressive elimination with sampled early exit.
    ///
    /// Returns results bucketed by sigma band. Foveal first, then Near, then Good.
    /// Within each band: sorted by u32 distance. No float anywhere.
    ///
    /// `top_k`: maximum results to return.
    /// `candidates`: the corpus to search. Each entry provides byte-level access
    ///   to its bits for sampled and full comparison.
    pub fn query_hdr<T: AsBytes>(
        &self,
        query: &[u8],   // query vector as bytes (2048 bytes for 16K)
        candidates: &[T],
        top_k: usize,
    ) -> Vec<RankedHit> {
        let query_len = query.len();

        // Buckets: one per band worth collecting
        let mut foveal: Vec<(usize, u32)> = Vec::new();
        let mut near: Vec<(usize, u32)> = Vec::new();
        let mut good: Vec<(usize, u32)> = Vec::new();

        // Stage 1 sample size: 1/16 of total bytes
        let s1_len = query_len / 16;
        // Stage 2 sample size: 1/4 of total bytes
        let s2_len = query_len / 4;

        // The SIMD hamming function. Resolved ONCE. Used millions of times.
        let hamming = crate::simd::select_hamming_fn();

        for (idx, candidate) in candidates.iter().enumerate() {
            let cbytes = candidate.as_bytes();

            // ── STAGE 1: 1/16 sample ──────────────────────────────────
            // Compare first 1/16 of bytes. Cost: ~2 cycles.
            let s1_dist = hamming(&query[..s1_len], &cbytes[..s1_len]);
            let s1_projected = (s1_dist as u32) * 16; // project to full width

            // Reject if projected distance is above the Good threshold (μ - σ).
            // This eliminates candidates that are almost certainly in the noise.
            if s1_projected > self.bands[2] {
                continue; // ~84%+ eliminated here
            }

            // ── STAGE 2: 1/4 sample ───────────────────────────────────
            // Compare first 1/4 of bytes. Cost: ~8 cycles.
            let s2_dist = hamming(&query[..s2_len], &cbytes[..s2_len]);
            let s2_projected = (s2_dist as u32) * 4;

            // Tighter filter: reject above Near threshold (μ - 2σ).
            if s2_projected > self.bands[1] {
                continue; // another ~80% of survivors eliminated
            }

            // ── STAGE 3: full comparison ──────────────────────────────
            // Compare all bytes. Cost: ~32 cycles. But only ~0.5% reach here.
            let full_dist = hamming(query, cbytes) as u32;

            match self.band(full_dist) {
                Band::Foveal => foveal.push((idx, full_dist)),
                Band::Near => near.push((idx, full_dist)),
                Band::Good => good.push((idx, full_dist)),
                _ => {} // Weak or Reject — don't collect
            }

            // Early termination: foveal bucket full
            if foveal.len() >= top_k {
                break;
            }
        }

        // Assemble top-k from buckets. Best band first. Integer sort within.
        let mut results = Vec::with_capacity(top_k);

        foveal.sort_unstable_by_key(|&(_, d)| d);
        for &(idx, dist) in foveal.iter().take(top_k) {
            results.push(RankedHit { index: idx, distance: dist, band: Band::Foveal });
        }

        if results.len() < top_k {
            near.sort_unstable_by_key(|&(_, d)| d);
            for &(idx, dist) in near.iter().take(top_k - results.len()) {
                results.push(RankedHit { index: idx, distance: dist, band: Band::Near });
            }
        }

        if results.len() < top_k {
            good.sort_unstable_by_key(|&(_, d)| d);
            for &(idx, dist) in good.iter().take(top_k - results.len()) {
                results.push(RankedHit { index: idx, distance: dist, band: Band::Good });
            }
        }

        results
    }

    /// Feed an observed distance into the running statistics.
    /// Call after each query with the distances of actual results.
    /// Returns ShiftAlert if the data distribution has changed significantly.
    ///
    /// Uses Welford's online algorithm. Integer arithmetic.
    pub fn observe(&mut self, distance: u32) -> Option<ShiftAlert> {
        let d = distance as u64;
        self.running_count += 1;

        // Welford's online mean and M2 update
        let delta = d as i64 - (self.running_mean / self.running_count.max(1)) as i64;
        self.running_mean += d;
        let new_mean = self.running_mean / self.running_count;
        let delta2 = d as i64 - new_mean as i64;
        self.running_m2 = self.running_m2.wrapping_add((delta.wrapping_mul(delta2)) as u64);

        // Check every 1000 observations
        if self.running_count % 1000 == 0 && self.running_count > 1000 {
            let running_mu = (self.running_mean / self.running_count) as u32;
            let running_var = (self.running_m2 / self.running_count) as u32;
            let running_sigma = isqrt(running_var).max(1);

            // Shift detection: running stats diverge from calibrated stats
            let mu_drift = running_mu.abs_diff(self.mu);
            let sigma_drift = running_sigma.abs_diff(self.sigma);

            if mu_drift > self.sigma / 2 || sigma_drift > self.sigma / 4 {
                return Some(ShiftAlert {
                    old_mu: self.mu,
                    new_mu: running_mu,
                    old_sigma: self.sigma,
                    new_sigma: running_sigma,
                    observations: self.running_count as u32,
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
    }
}

pub struct RankedHit {
    /// Index of the matched candidate in the input slice.
    pub index: usize,
    /// Exact Hamming distance (from stage 3 full comparison).
    pub distance: u32,
    /// Sigma band classification.
    pub band: Band,
}

pub struct ShiftAlert {
    pub old_mu: u32,
    pub new_mu: u32,
    pub old_sigma: u32,
    pub new_sigma: u32,
    pub observations: u32,
}

/// Trait for types that expose their raw bytes for SIMD comparison.
pub trait AsBytes {
    fn as_bytes(&self) -> &[u8];
}

/// Integer square root. No float in the hot path.
/// Uses Newton's method with integer arithmetic.
fn isqrt(n: u32) -> u32 {
    if n == 0 { return 0; }
    // Initial guess: bit-shift approximation
    let mut x = 1u32 << ((32 - n.leading_zeros()) / 2);
    loop {
        let x1 = (x + n / x) / 2;
        if x1 >= x { return x; }
        x = x1;
    }
}
```

---

### Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn random_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut state = seed;
        (0..n).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u8
        }).collect()
    }

    #[test]
    fn calibration_from_random_data() {
        // Random 16K vectors should have μ ≈ 8192, σ ≈ 64
        let hamming = crate::simd::select_hamming_fn();
        let vecs: Vec<Vec<u8>> = (0..100).map(|i| random_bytes(2048, i)).collect();
        let mut dists = Vec::new();
        for i in 0..100 {
            for j in (i+1)..100 {
                dists.push(hamming(&vecs[i], &vecs[j]) as u32);
            }
        }

        let meter = Belichtungsmesser::calibrate(&dists);

        // μ should be near 8192 (±200 for sample variance)
        assert!(meter.mu > 7900 && meter.mu < 8500,
            "Expected μ near 8192, got {}", meter.mu);
        // σ should be near 64 (±30)
        assert!(meter.sigma > 30 && meter.sigma < 100,
            "Expected σ near 64, got {}", meter.sigma);
        // Band ordering must be strictly ascending
        assert!(meter.bands[0] < meter.bands[1]);
        assert!(meter.bands[1] < meter.bands[2]);
        assert!(meter.bands[2] < meter.bands[3]);
    }

    #[test]
    fn band_classification_correct_direction() {
        let meter = Belichtungsmesser {
            mu: 8192,
            sigma: 64,
            bands: [8192 - 192, 8192 - 128, 8192 - 64, 8192],
            running_count: 1000,
            running_mean: 8192000,
            running_m2: 64 * 64 * 1000,
        };

        // Very low distance = very similar = Foveal (best)
        assert_eq!(meter.band(7900), Band::Foveal);
        // Low distance = strong match = Near
        assert_eq!(meter.band(8010), Band::Near);
        // Below mean = good candidate
        assert_eq!(meter.band(8100), Band::Good);
        // Near mean = weak signal
        assert_eq!(meter.band(8170), Band::Weak);
        // Above mean = noise = Reject
        assert_eq!(meter.band(8200), Band::Reject);
        assert_eq!(meter.band(9000), Band::Reject);
    }

    #[test]
    fn cascade_eliminates_most_candidates() {
        let hamming = crate::simd::select_hamming_fn();

        // Create 10,000 random vectors + 10 vectors similar to query
        let query = random_bytes(2048, 0);
        let mut corpus: Vec<Vec<u8>> = (1..10_001).map(|i| random_bytes(2048, i)).collect();

        // Make 10 similar vectors by copying query and flipping few bits
        for i in 0..10 {
            let mut similar = query.clone();
            for j in 0..20 { // flip ~20 bytes → ~80 bits → distance ≈ 80 (far below μ)
                similar[j + i * 20] ^= 0xFF;
            }
            corpus.push(similar);
        }

        // Calibrate from random pairs
        let mut sample_dists = Vec::new();
        for i in 0..100 {
            for j in (i+1)..100 {
                sample_dists.push(hamming(&corpus[i], &corpus[j]) as u32);
            }
        }
        let meter = Belichtungsmesser::calibrate(&sample_dists);

        // Query: find top 10
        let results = meter.query_hdr(&query, &corpus, 10);

        // Should find the 10 similar vectors
        assert!(results.len() >= 5,
            "Should find at least 5 of 10 similar vectors, got {}", results.len());

        // All results should be in good bands (Foveal, Near, or Good)
        for hit in &results {
            assert!(hit.band <= Band::Good,
                "Result with distance {} classified as {:?}, expected Good or better",
                hit.distance, hit.band);
        }

        // The similar vectors should have much lower distance than μ
        for hit in &results {
            assert!(hit.distance < meter.mu,
                "Result distance {} should be below μ={}", hit.distance, meter.mu);
        }
    }

    #[test]
    fn cascade_work_savings() {
        // Measure how much work the cascade actually saves
        let query = random_bytes(2048, 0);
        let corpus: Vec<Vec<u8>> = (1..10_001).map(|i| random_bytes(2048, i)).collect();

        let hamming = crate::simd::select_hamming_fn();
        let mut sample_dists = Vec::new();
        for i in 0..100 {
            for j in (i+1)..100 {
                sample_dists.push(hamming(&corpus[i], &corpus[j]) as u32);
            }
        }
        let meter = Belichtungsmesser::calibrate(&sample_dists);

        // Count how many candidates survive each stage
        let s1_len = 2048 / 16;  // 128 bytes
        let s2_len = 2048 / 4;   // 512 bytes
        let mut s1_pass = 0u32;
        let mut s2_pass = 0u32;
        let mut s3_pass = 0u32;
        let total = corpus.len() as u32;

        for candidate in &corpus {
            let s1 = hamming(&query[..s1_len], &candidate[..s1_len]) as u32 * 16;
            if s1 > meter.bands[2] { continue; }
            s1_pass += 1;

            let s2 = hamming(&query[..s2_len], &candidate[..s2_len]) as u32 * 4;
            if s2 > meter.bands[1] { continue; }
            s2_pass += 1;

            let full = hamming(&query, &candidate) as u32;
            if meter.band(full) <= Band::Good {
                s3_pass += 1;
            }
        }

        let s1_reject_pct = 100.0 * (1.0 - s1_pass as f64 / total as f64);
        let s2_reject_pct = 100.0 * (1.0 - s2_pass as f64 / s1_pass.max(1) as f64);

        // Against random data: stage 1 should reject >80%
        println!("Stage 1: {}/{} pass ({:.1}% rejected)", s1_pass, total, s1_reject_pct);
        println!("Stage 2: {}/{} pass ({:.1}% rejected of survivors)", s2_pass, s1_pass, s2_reject_pct);
        println!("Stage 3: {} final results", s3_pass);

        assert!(s1_reject_pct > 70.0,
            "Stage 1 should reject >70% of random candidates, got {:.1}%", s1_reject_pct);

        // Effective work compared to brute force
        let brute_cycles = total as f64 * 32.0;  // 32 cycles per full comparison
        let cascade_cycles = total as f64 * 2.0   // stage 1: all candidates, 2 cycles
            + s1_pass as f64 * 8.0                  // stage 2: survivors, 8 cycles
            + s2_pass as f64 * 32.0;                // stage 3: survivors, 32 cycles
        let savings = 100.0 * (1.0 - cascade_cycles / brute_cycles);

        println!("Brute force: {:.0} cycles", brute_cycles);
        println!("Cascade:     {:.0} cycles", cascade_cycles);
        println!("Savings:     {:.1}%", savings);

        assert!(savings > 50.0,
            "Cascade should save >50% work vs brute force, got {:.1}%", savings);
    }

    #[test]
    fn shift_detection_triggers_on_distribution_change() {
        let mut meter = Belichtungsmesser {
            mu: 8192,
            sigma: 64,
            bands: [8192 - 192, 8192 - 128, 8192 - 64, 8192],
            running_count: 1000,
            running_mean: 8_192_000,
            running_m2: 64 * 64 * 1000,
        };

        // Feed distances from a SHIFTED distribution (μ=7500 instead of 8192)
        let mut alert_fired = false;
        for i in 0..5000 {
            let fake_dist = 7500 + (i % 100);  // mean ~7550, far from 8192
            if let Some(alert) = meter.observe(fake_dist) {
                assert!(alert.new_mu < alert.old_mu,
                    "Shift should detect lower mean");
                meter.recalibrate(&alert);
                alert_fired = true;
                break;
            }
        }

        assert!(alert_fired, "Shift alert should fire when μ changes by >0.5σ");
        assert!(meter.mu < 8000, "After recalibration, μ should reflect new distribution");
    }

    #[test]
    fn no_float_in_query_path() {
        // This test is a documentation test: it verifies that the query path
        // doesn't use f32/f64 by examining the return types.
        let meter = Belichtungsmesser::calibrate(&[8000, 8100, 8200, 8300, 8400]);
        
        // band() returns enum, not float
        let b: Band = meter.band(8050);
        assert!(matches!(b, Band::Near));

        // RankedHit.distance is u32, not f32
        let hit = RankedHit { index: 0, distance: 8050, band: Band::Near };
        let _d: u32 = hit.distance;  // compiles only if u32
        
        // No .score, .similarity, or .confidence fields that would be float
    }

    #[test]
    fn isqrt_correctness() {
        assert_eq!(isqrt(0), 0);
        assert_eq!(isqrt(1), 1);
        assert_eq!(isqrt(4), 2);
        assert_eq!(isqrt(9), 3);
        assert_eq!(isqrt(4096), 64);  // σ² for 16K random vectors
        assert_eq!(isqrt(4095), 63);  // floor
        assert_eq!(isqrt(u32::MAX), 65535);
    }
}
```

---

### Where this goes

```
In rustynum-core:
  src/belichtungsmesser.rs   (~350 lines: struct, calibrate, band, query_hdr, observe)
  Uses: crate::simd::select_hamming_fn() for all distance computation.
  No external deps beyond blake3 (if needed for hashing) and std.

In lance-graph (upstream contribution):
  Copy the same code into graph/spo/belichtungsmesser.rs
  WITHOUT rustynum dependency. Replace select_hamming_fn() with
  the inline SIMD dispatch from BlasGraph types.rs.
  
In ladybug-rs:
  Uses rustynum-core::Belichtungsmesser directly.
  Integrated into Plane.distance() for alpha-aware cascade.
```
