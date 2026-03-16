# SESSION_D_LENS_CORRECTION.md

## Gamma + Cushion Lens Correction + Self-Organizing Boundary Fold

**Repo:** rustynum (WRITE), lance-graph (WRITE)
**Prereq:** Session C completed (hdr::Cascade has ReservoirSample, skewness, kurtosis, Welford)
**Scope:** lens correction pipeline, boundary folding, adaptive healing. Nothing else.
**Stop when:** `cargo test --workspace` passes in both repos, correction tests pass.

---

## CONTEXT

Session C ported ReservoirSample with skewness and kurtosis detection.
The current system uses these to AUTO-SWITCH between parametric (σ bands)
and empirical (quantile bands). That's a binary choice: normal or not.

This session replaces the binary switch with a continuous lens correction
that MAKES parametric bands work for any distribution shape. The empirical
mode becomes a diagnostic check, not an alternative code path.

Three corrections applied in order:
1. **Gamma:** fixes skewness (left/right asymmetry)
2. **Cushion:** fixes kurtosis (heavy/light tails)
3. **Fold:** moves boundaries to density troughs (minimum pressure)

After all three: σ-based bands work on any distribution.

---

## VERIFY SESSION C COMPLETED

```bash
cd rustynum
cargo test -p rustynum-core -- hdr::reservoir  # must pass
cargo test -p rustynum-core -- hdr::shift      # must pass
# Confirm these exist in hdr.rs:
grep "fn skewness" rustynum-core/src/hdr.rs     # must find it
grep "fn kurtosis" rustynum-core/src/hdr.rs     # must find it
grep "struct ReservoirSample" rustynum-core/src/hdr.rs  # must find it
```

---

## STEP 1: Add gamma field + LUT to Cascade

In `rustynum-core/src/hdr.rs`, add to the Cascade struct:

```rust
pub struct Cascade {
    // ...existing fields from Session C...
    
    /// Gamma correction for skew. Fixed-point: 100 = γ=1.0 (no correction).
    /// γ > 100: right-skewed data, stretches left tail.
    /// γ < 100: left-skewed data, stretches right tail.
    /// Derived from reservoir skewness. Updated on fold.
    gamma: u16,
    
    /// Kurtosis correction. Fixed-point: 100 = normal (κ=3.0).
    /// > 100: heavy tails, widen extreme bands.
    /// < 100: light tails, narrow extreme bands.
    /// Derived from reservoir kurtosis. Updated on fold.
    kappa: u16,
    
    /// Gamma lookup table. 256 entries. Integer.
    /// Maps normalized distance [0..255] to corrected distance.
    /// Rebuilt on calibrate/recalibrate/fold. Not on every query.
    gamma_lut: Box<[u32; 256]>,
    
    /// Boundary pressure: density at each band boundary.
    /// Updated on fold. Used for adaptive healing decisions.
    pressure: [u32; 4],
}
```

Initialize in `calibrate()`:

```rust
impl Cascade {
    pub fn calibrate(sample: &[u32]) -> Self {
        let mut c = Self {
            // ...existing initialization...
            gamma: 100,                        // no correction initially
            kappa: 100,                        // no correction initially
            gamma_lut: Box::new([0u32; 256]),  // identity initially
            pressure: [0; 4],                  // unknown initially
        };
        // Identity LUT: gamma_lut[i] = i * μ / 255
        c.rebuild_gamma_lut();
        c
    }
}
```

---

## STEP 2: Gamma correction (skewness → power-law LUT)

```rust
impl Cascade {
    /// Derive gamma from reservoir skewness.
    /// Pearson's second skewness maps linearly to gamma:
    ///   skew = 0  → γ = 1.0 (symmetric)
    ///   skew > 0  → γ > 1.0 (right-skewed, stretch compressed left side)
    ///   skew < 0  → γ < 1.0 (left-skewed, stretch compressed right side)
    fn calibrate_gamma(&mut self) {
        let skew = self.reservoir.skewness(self.mu, self.sigma);
        // γ = 1.0 + skew × 0.15, clamped to [0.5, 2.0]
        // In fixed-point: gamma = 100 + skew * 15, clamped to [50, 200]
        self.gamma = (100 + skew * 15).clamp(50, 200) as u16;
        self.rebuild_gamma_lut();
    }
    
    /// Build the 256-entry gamma LUT. Float used HERE ONLY (on calibration).
    /// Hot path uses integer table lookup.
    fn rebuild_gamma_lut(&mut self) {
        let g = self.gamma as f64 / 100.0;
        for i in 0..256 {
            let x = (i as f64 + 0.5) / 256.0;  // normalized [0, 1], avoid x=0
            let corrected = x.powf(g);
            self.gamma_lut[i] = (corrected * self.mu as f64) as u32;
        }
    }
    
    /// Apply gamma correction to a raw distance. Integer LUT lookup.
    /// Cost: 1 multiply + 1 table lookup. ~2 cycles.
    #[inline]
    fn gamma_correct(&self, distance: u32) -> u32 {
        if self.gamma == 100 { return distance; }  // fast path: no correction
        // Normalize distance to [0, 255] relative to μ
        let idx = ((distance as u64 * 255) / self.mu.max(1) as u64) as usize;
        self.gamma_lut[idx.min(255)]
    }
}
```

---

## STEP 3: Cushion correction (kurtosis → cubic tail adjustment)

```rust
impl Cascade {
    /// Derive kappa from reservoir kurtosis.
    /// Reservoir kurtosis is scaled ×100: normal = 300.
    /// We normalize: kappa = kurtosis × 100 / 300.
    /// So kappa 100 = normal, >100 = heavy tails, <100 = light tails.
    fn calibrate_kappa(&mut self) {
        let kurt = self.reservoir.kurtosis(self.mu, self.sigma);
        self.kappa = (kurt * 100 / 300).clamp(30, 300) as u16;
    }
    
    /// Apply cushion correction to fix tail weight.
    /// Normal tails (kappa=100): no correction.
    /// Heavy tails (kappa>100): push extremes outward (widen tail bands).
    /// Light tails (kappa<100): push extremes inward (narrow tail bands).
    ///
    /// Uses cubic term: correction grows with cube of deviation from μ.
    /// At center (d ≈ μ): correction ≈ 0 (center is always stable).
    /// At tails: correction dominates.
    ///
    /// Cost: ~5 integer ops. No float.
    #[inline]
    fn cushion_correct(&self, distance: u32) -> u32 {
        if self.kappa == 100 { return distance; }  // fast path: normal tails
        
        let d = distance as i64;
        let mu = self.mu as i64;
        let deviation = d - mu;
        
        // kappa_norm: 100 = normal
        let kappa_norm = self.kappa as i64;
        
        // Cubic correction scaled to prevent overflow:
        //   correction = deviation³ / μ² × (1 - 100/κ) / 100
        //
        // For heavy tails (κ>100): (1 - 100/κ) > 0 → pushes tails OUT
        // For light tails (κ<100): (1 - 100/κ) < 0 → pushes tails IN
        // For normal (κ=100): (1 - 100/κ) = 0 → no correction
        let mu_sq = (mu * mu).max(1);
        let kappa_factor = 100 - (10000 / kappa_norm.max(1)); // ×100 scaled
        let correction = deviation * deviation / mu_sq.max(1) * deviation
            * kappa_factor / 10000;
        
        (d + correction).clamp(0, mu * 3) as u32
    }
}
```

---

## STEP 4: Compose corrections in expose()

```rust
impl Cascade {
    /// Band classification with full lens correction pipeline.
    /// Raw distance → gamma (skew) → cushion (kurtosis) → band lookup.
    /// Cost: ~8 cycles total (2 gamma + 5 cushion + 1 compare).
    #[inline]
    pub fn expose(&self, distance: u32) -> Band {
        let d = self.cushion_correct(self.gamma_correct(distance));
        // NOTE: gamma first, then cushion. Gamma fixes the asymmetry
        // so the cubic cushion term operates on a symmetric-ish distribution.
        
        let bands = if self.use_empirical { &self.empirical_bands } else { &self.bands };
        if d < bands[0] { Band::Foveal }
        else if d < bands[1] { Band::Near }
        else if d < bands[2] { Band::Good }
        else if d < bands[3] { Band::Weak }
        else { Band::Reject }
    }
}
```

---

## STEP 5: Self-organizing boundary fold

The boundaries MOVE toward density troughs. Like a soap film finding
minimum energy. Reduces the need for healing by eliminating ambiguity.

```rust
impl Cascade {
    /// Move each boundary toward the nearest density trough.
    /// Reduces classification pressure: fewer candidates fall on boundaries.
    ///
    /// Called during periodic maintenance (every ~5000 observations),
    /// NOT on every query. The fold is expensive (sorts reservoir).
    pub fn fold_boundaries(&mut self) {
        if self.reservoir.len() < 500 { return; }
        
        let mut sorted = self.reservoir.samples.clone();
        sorted.sort_unstable();
        
        let window = (self.sigma / 4).max(1);
        
        for i in 0..4 {
            let current = self.bands[i];
            let search_lo = current.saturating_sub(self.sigma);
            let search_hi = current.saturating_add(self.sigma);
            
            // Slide a window across [lo, hi], find position with minimum density
            let mut min_density = u32::MAX;
            let mut best_pos = current;
            
            let mut pos = search_lo;
            while pos <= search_hi {
                // Count samples within ±window of pos
                let count = sorted.iter()
                    .filter(|&&d| d >= pos.saturating_sub(window)
                                && d <= pos.saturating_add(window))
                    .count() as u32;
                
                if count < min_density {
                    min_density = count;
                    best_pos = pos;
                }
                pos += window / 2;  // slide by half-window for resolution
            }
            
            self.bands[i] = best_pos;
            self.pressure[i] = min_density;
        }
        
        // Ensure strict monotonicity: b₀ < b₁ < b₂ < b₃
        for i in 1..4 {
            if self.bands[i] <= self.bands[i - 1] {
                self.bands[i] = self.bands[i - 1] + 1;
            }
        }
    }
}
```

---

## STEP 6: Adaptive healing using boundary pressure

The pressure from fold_boundaries tells the cascade WHERE to spend bytes.

```rust
impl Cascade {
    /// Determine how many bytes to read based on boundary pressure.
    /// High pressure boundary nearby → read more bytes for certainty.
    /// Low pressure (clean gap) → accept the projection, save bytes.
    ///
    /// Returns: number of bytes to read total (including already read).
    pub fn healing_target(
        &self,
        projected: u32,
        sigma_est: u32,
        bytes_read: usize,
        total_bytes: usize,
    ) -> usize {
        // Find nearest boundary and its pressure
        let mut nearest_dist = u32::MAX;
        let mut nearest_pressure = 0u32;
        
        for i in 0..4 {
            let dist_to_boundary = (projected as i64 - self.bands[i] as i64)
                .unsigned_abs() as u32;
            if dist_to_boundary < nearest_dist {
                nearest_dist = dist_to_boundary;
                nearest_pressure = self.pressure[i];
            }
        }
        
        // Far from any boundary (> 3σ): fully confident, no healing
        if nearest_dist > sigma_est * 3 {
            return bytes_read;
        }
        
        // Urgency: pressure × proximity (both high = maximum healing)
        let urgency = nearest_pressure as u64 
            * sigma_est as u64 
            / nearest_dist.max(1) as u64;
        
        let additional = match urgency {
            0..=50 => 0,                    // low urgency, accept uncertainty
            51..=200 => 64,                 // one cache line
            201..=500 => 256,               // moderate healing
            501..=1000 => 512,              // significant healing
            _ => total_bytes - bytes_read,  // go to full
        };
        
        // SIMD-align to 64 bytes
        ((bytes_read + additional + 63) & !63).min(total_bytes)
    }
}
```

---

## STEP 7: Wire into cascade query

Update `Cascade::query()` to use corrections and adaptive healing:

```rust
// Inside the cascade query loop, REPLACE the fixed band check:

// BEFORE (fixed threshold):
// if projected > self.bands[2] { continue; }

// AFTER (corrected + adaptive):
let projected = d1 * (total_bytes / s1_bytes) as u32;
let corrected = self.cushion_correct(self.gamma_correct(projected));
let sigma_est = estimation_sigma(s1_bytes, projected, total_bytes);

// Quick rejection on corrected distance
if self.expose_raw(corrected) == Band::Reject { continue; }

// Boundary proximity check: do we need healing?
let heal_to = self.healing_target(corrected, sigma_est, s1_bytes, total_bytes);

if heal_to > s1_bytes {
    // Read additional bytes (Tetris piece fills the gap)
    let d_extra = hamming_fn(&query[s1_bytes..heal_to], &cand[s1_bytes..heal_to]) as u32;
    let cumulative = d1 + d_extra;
    let projected2 = cumulative * (total_bytes / heal_to) as u32;
    let corrected2 = self.cushion_correct(self.gamma_correct(projected2));
    
    if self.expose_raw(corrected2) == Band::Reject { continue; }
    
    // Continue to full if needed...
}
```

Add helper:

```rust
impl Cascade {
    /// Raw band classification without the expose() public API overhead.
    #[inline]
    fn expose_raw(&self, corrected_distance: u32) -> Band {
        let bands = if self.use_empirical { &self.empirical_bands } else { &self.bands };
        if corrected_distance < bands[0] { Band::Foveal }
        else if corrected_distance < bands[1] { Band::Near }
        else if corrected_distance < bands[2] { Band::Good }
        else if corrected_distance < bands[3] { Band::Weak }
        else { Band::Reject }
    }
}
```

---

## STEP 8: Periodic maintenance (integrate gamma + kappa + fold)

Wire the corrections into the observe() path:

```rust
impl Cascade {
    pub fn observe(&mut self, distance: u32) -> Option<ShiftAlert> {
        // ...existing Welford + reservoir update from Session C...
        
        // Periodic maintenance every 5000 observations
        if self.running_count % 5000 == 0 && self.reservoir.len() >= 500 {
            self.calibrate_gamma();      // update γ from skewness
            self.calibrate_kappa();      // update κ from kurtosis
            self.fold_boundaries();      // move bands to density troughs
        }
        
        // ...existing shift detection...
    }
    
    pub fn recalibrate(&mut self, alert: &ShiftAlert) {
        // ...existing reset from Session C...
        
        // Reset corrections to neutral
        self.gamma = 100;
        self.kappa = 100;
        self.pressure = [0; 4];
        self.rebuild_gamma_lut();
    }
}
```

---

## STEP 9: Port to lance-graph

Copy the correction logic into `lance-graph/crates/lance-graph/src/graph/blasgraph/hdr.rs`.

The lance-graph version is simpler because it operates on `&[u64]` word arrays
instead of `&[u8]` byte slices. Same gamma LUT. Same cushion cubic. Same fold.

```bash
cd ../lance-graph  # or wherever it is
# Read rustynum's hdr.rs, copy gamma_correct, cushion_correct, 
# fold_boundaries, healing_target, calibrate_gamma, calibrate_kappa
# into the Cascade struct in lance-graph's hdr.rs.
```

Adapt field types if needed (lance-graph may use different integer widths).

---

## STEP 10: Tests

```rust
#[test]
fn gamma_identity_when_symmetric() {
    let hdr = Cascade::calibrate(&[8192; 200]);
    assert_eq!(hdr.gamma, 100);  // symmetric data → no correction
    assert_eq!(hdr.gamma_correct(8000), 8000);  // identity
    assert_eq!(hdr.gamma_correct(8192), 8192);  // identity
}

#[test]
fn gamma_corrects_right_skew() {
    let mut hdr = Cascade::calibrate(&[8192; 100]);
    
    // Feed right-skewed data (many low distances, few high)
    for i in 0..2000 {
        let d = 7000 + (i % 200);  // clustered below μ
        hdr.observe(d);
    }
    for i in 0..200 {
        let d = 9000 + (i * 10);   // sparse tail above μ
        hdr.observe(d);
    }
    
    // Force recalibration
    hdr.calibrate_gamma();
    
    // γ should be > 100 (right-skewed → stretch left side)
    assert!(hdr.gamma > 100, "Right-skewed data should give γ > 1.0, got {}", hdr.gamma);
    
    // A distance below μ should be STRETCHED (corrected > raw)
    let raw = 7500;
    let corrected = hdr.gamma_correct(raw);
    // Gamma > 1 on [0,1] range: x^γ < x for x < 1, so corrected < raw
    // (stretches the compressed left side toward center)
    assert_ne!(corrected, raw, "Gamma should change the distance");
}

#[test]
fn cushion_identity_when_normal_kurtosis() {
    let mut hdr = Cascade::calibrate(&[8192; 200]);
    hdr.kappa = 100;  // normal
    
    assert_eq!(hdr.cushion_correct(8000), 8000);  // center: no correction
    assert_eq!(hdr.cushion_correct(8192), 8192);  // at μ: no correction
}

#[test]
fn cushion_widens_heavy_tails() {
    let mut hdr = Cascade::calibrate(&[8192; 200]);
    hdr.mu = 8192;
    hdr.kappa = 200;  // heavy tails (2× normal kurtosis)
    
    // Distance far from μ should be pushed further out
    let raw = 7000;  // 1192 below μ
    let corrected = hdr.cushion_correct(raw);
    assert!(corrected < raw, 
        "Heavy-tail cushion should push low distances further from μ: raw={} corrected={}",
        raw, corrected);
}

#[test]
fn fold_moves_boundary_to_trough() {
    let mut hdr = Cascade::calibrate(&[8192; 100]);
    
    // Create bimodal reservoir: peak at 7000 and 8500, trough at 7800
    let mut bimodal: Vec<u32> = Vec::new();
    for _ in 0..500 { bimodal.push(7000 + (fastrand() % 200)); }  // peak 1
    for _ in 0..500 { bimodal.push(8500 + (fastrand() % 200)); }  // peak 2
    // Note: gap around 7800 — no samples there
    
    for &d in &bimodal {
        hdr.reservoir.observe(d);
    }
    
    // Before fold: bands are at σ-based positions
    let band_before = hdr.bands[1];  // Near/Good boundary
    
    hdr.fold_boundaries();
    
    let band_after = hdr.bands[1];
    
    // The boundary should have moved toward the trough (~7800)
    // It should be closer to 7800 than it was before
    let dist_before = (band_before as i64 - 7800).unsigned_abs();
    let dist_after = (band_after as i64 - 7800).unsigned_abs();
    
    assert!(dist_after <= dist_before,
        "Fold should move boundary toward trough: before={} after={} trough=7800",
        band_before, band_after);
}

#[test]
fn fold_reduces_pressure() {
    let mut hdr = Cascade::calibrate(&[8192; 100]);
    
    // Fill reservoir with data that clusters around default boundaries
    for _ in 0..1000 {
        let d = hdr.bands[1] + (fastrand() % 20) - 10;  // right on the boundary
        hdr.reservoir.observe(d);
    }
    
    hdr.fold_boundaries();
    
    // After fold, boundary should have moved away from the cluster
    // Pressure should be lower (fewer candidates on the boundary)
    assert!(hdr.pressure[1] < 500,
        "Fold should reduce boundary pressure, got {}", hdr.pressure[1]);
}

#[test]
fn healing_reads_more_at_high_pressure_boundary() {
    let mut hdr = Cascade::calibrate(&[8192; 200]);
    hdr.sigma = 64;
    hdr.pressure = [10, 500, 10, 10];  // b₁ has high pressure
    
    let sigma_est = 256;  // 1/16 sample estimation σ
    
    // Distance near high-pressure boundary b₁: should heal
    let near_b1 = hdr.bands[1] + 30;  // close to b₁
    let target = hdr.healing_target(near_b1, sigma_est, 128, 2048);
    assert!(target > 128, "Should read more bytes near high-pressure boundary");
    
    // Distance far from any boundary: should not heal
    let far_away = hdr.bands[0] - 500;  // deep in Foveal
    let target_far = hdr.healing_target(far_away, sigma_est, 128, 2048);
    assert_eq!(target_far, 128, "Should not heal when far from boundaries");
}

#[test]
fn full_pipeline_correction_converges() {
    let mut hdr = Cascade::calibrate(&[8192; 100]);
    
    // Feed skewed + heavy-tailed data over multiple maintenance cycles
    for round in 0..5 {
        for i in 0..5000 {
            // Right-skewed, heavy-tailed distribution
            let base = 7000 + (i % 300);
            let tail = if i % 50 == 0 { 2000 } else { 0 };  // occasional outlier
            hdr.observe(base + tail);
        }
        
        // After 5000 observations: maintenance triggers gamma + kappa + fold
        println!("Round {}: γ={} κ={} bands={:?} pressure={:?}",
            round, hdr.gamma, hdr.kappa, hdr.bands, hdr.pressure);
    }
    
    // After convergence: gamma should be non-trivial (skewed data)
    assert_ne!(hdr.gamma, 100, "Gamma should adapt to skewed data");
    // Kappa should reflect heavy tails
    assert_ne!(hdr.kappa, 100, "Kappa should adapt to heavy-tailed data");
    // Boundaries should have folded to density troughs
    // (hard to assert exact values, but monotonicity must hold)
    for i in 1..4 {
        assert!(hdr.bands[i] > hdr.bands[i-1], "Bands must be monotonic");
    }
}

/// Simple fast random for tests (not for production reservoir)
fn fastrand() -> u32 {
    use std::cell::Cell;
    thread_local! { static STATE: Cell<u64> = Cell::new(12345); }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        x as u32
    })
}
```

---

## STEP 11: Verify both repos

```bash
# Rustynum:
RUSTFLAGS="-C target-cpu=native" cargo test --workspace
cargo clippy --workspace -- -D warnings
cargo test -p rustynum-core -- hdr  # all hdr tests

# Lance-graph:
cd ../lance-graph
cargo test --workspace
cargo clippy --workspace -- -D warnings
```

---

## NOT IN SCOPE

```
× Don't add PreciseMode to lance-graph (separate session)
× Don't touch SIMD dispatch (comes with BitVec rebuild)
× Don't add GPU backends (roadmap)
× Don't modify Plane/Node/Mask (separate prompt)
× Don't touch simd_avx512.rs or simd.rs (Session A already did this)
× Don't add chromatic aberration correction (multimodal per-mode gamma)
  → This is a future extension if bimodal data needs per-peak correction.
  → For now, the fold_boundaries handles bimodal by finding the trough.
```
