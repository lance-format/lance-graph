# UNIFIED_HDR_RENAME_AND_CROSSPOLINATE.md

## Part 0: Rename Files and Functions (Both Repos)
## Part 1: Cross-Pollinate Algorithms (Rustynum ↔ Lance-Graph)

**Repos:** rustynum (WRITE), lance-graph (WRITE)
**Read first:** rustynum-core/src/simd_compat.rs, rustynum-core/src/simd.rs,
  lance-graph/crates/lance-graph/src/graph/blasgraph/light_meter.rs

---

## PART 0A: SIMD FILE RENAME (rustynum)

### The Problem

The filenames are backwards from what every session assumes:

```
simd_compat.rs  = AVX-512 PRIMARY (F32x16, F64x8, VPOPCNTDQ)
simd.rs         = AVX2 FALLBACK + dispatcher
```

Every session reads "compat" as "legacy fallback" and routes things wrong.

### The Rename

```
BEFORE:                    AFTER:
simd_compat.rs          →  simd_avx512.rs       (rename, primary AVX-512)
simd.rs                 →  simd.rs              (stays, dispatcher + imports avx2)
NEW:                        simd_avx2.rs         (extract AVX2 impls FROM simd.rs)
NEW:                        simd_arm.rs          (placeholder for NEON/SVE)
NEW:                        simd_avx512_nightly.rs (placeholder for FP16/AMX)
```

### Step 1: Rename simd_compat.rs → simd_avx512.rs

```bash
cd rustynum-core/src
git mv simd_compat.rs simd_avx512.rs
```

### Step 2: Add re-export shim (zero breakage)

Create new `simd_compat.rs` with ONLY:

```rust
//! Backward-compatibility shim. All types moved to simd_avx512.rs.
//! 
//! Migrate imports: `use crate::simd_compat::X` → `use crate::simd_avx512::X`
#[allow(deprecated)]
#[deprecated(since = "0.4.0", note = "renamed to simd_avx512")]
pub use crate::simd_avx512::*;
```

### Step 3: Update lib.rs

```rust
pub mod simd_avx512;      // AVX-512 primary (was simd_compat)
pub mod simd;             // dispatcher
pub mod simd_avx2;        // AVX2 fallback (extracted from simd.rs)

// Backward compat — remove in next major version
#[allow(deprecated)]
pub mod simd_compat;
```

### Step 4: Extract simd_avx2.rs from simd.rs

Move THESE functions from simd.rs into simd_avx2.rs:

```
hamming_avx2()              ← the Harley-Seal implementation
hamming_scalar_popcnt()     ← scalar fallback
popcount_avx2()             ← AVX2 popcount
popcount_scalar()           ← scalar fallback
dot_f32_avx2()              ← AVX2 dot product (if exists)
dot_i8_scalar()             ← scalar INT8 dot
f32_to_fp16_scalar()        ← bit-shift conversion
```

simd.rs KEEPS only:
- The `OnceLock` dispatchers
- The `select_*_fn()` functions
- The `pub fn` entry points that delegate

### Step 5: Update simd.rs imports

```rust
// simd.rs — dispatcher only

mod simd_avx512;  // or use crate::simd_avx512 if pub
mod simd_avx2;

use std::sync::OnceLock;

static HAMMING: OnceLock<fn(&[u8], &[u8]) -> u64> = OnceLock::new();

#[inline]
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u64 {
    HAMMING.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512vpopcntdq") {
                return simd_avx512::hamming_distance
            }
            if is_x86_feature_detected!("avx2") {
                return simd_avx2::hamming_distance
            }
        }
        #[cfg(target_arch = "aarch64")]
        { return simd_avx2::hamming_scalar }  // LLVM emits NEON CNT
        
        simd_avx2::hamming_scalar
    })(a, b)
}
```

### Step 6: Migrate direct simd_compat imports (20+ sites)

Find-replace across workspace. The shim makes this non-urgent but do it now:

```bash
# These are ALL the files that import from simd_compat:
sed -i 's/simd_compat/simd_avx512/g' \
  rustyblas/src/level3.rs \
  rustyblas/src/bf16_gemm.rs \
  rustyblas/src/int8_gemm.rs \
  rustyblas/examples/gemm_benchmark.rs \
  rustymkl/src/fft.rs \
  rustymkl/src/vml.rs \
  rustynum-core/src/prefilter.rs \
  rustynum-core/src/simd_avx2.rs \
  rustynum-core/src/bf16_hamming.rs \
  rustynum-core/src/simd.rs \
  rustynum-rs/src/simd_ops/mod.rs \
  rustynum-rs/src/num_array/hdc.rs \
  rustynum-rs/src/num_array/array_struct.rs \
  rustynum-rs/src/num_array/bitwise.rs
```

Also update comments in lib.rs files:

```bash
# Fix comments mentioning simd_compat
grep -rn "simd_compat" --include="*.rs" | grep -v "target/" | grep "//"
# Update each comment to say simd_avx512
```

### Step 7: Verify

```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings
# The deprecated shim should produce zero warnings because we migrated all imports
```

### Downstream Debt: ZERO

The re-export shim means nothing breaks. The sed command migrates all internal
imports. External consumers (ladybug-rs) import via `rustynum_core::simd::`
(the dispatcher) which didn't change. The TYPES are imported from
`rustynum_core::simd_avx512::` but with the shim, the old path still works.

---

## PART 0B: HDR RENAME (Both Repos)

### Rustynum Renames

```
FILE:
  NEW: rustynum-core/src/hdr.rs    (new file, struct + methods)

MOVE INTO hdr.rs FROM simd.rs:
  hdr_cascade_search()          →  hdr::Cascade::query()
  HdrResult                     →  hdr::RankedHit { index, distance, band }
  PreciseMode                   →  hdr::PreciseMode (stays)

MOVE INTO hdr.rs NEW:
  Band enum                     ←  from lance-graph design
  ShiftAlert                    ←  from lance-graph design
  ReservoirSample               ←  from lance-graph design

RENAME IN hdc.rs (wrappers stay, delegate to hdr::Cascade):
  hamming_distance_adaptive()   →  calls hdr::Cascade::test()
  hamming_search_adaptive()     →  calls hdr::Cascade::query()
  cosine_search_adaptive()      →  calls hdr::Cascade::query(precise: PreciseMode::Vnni)

BACKWARD COMPAT:
  Keep hdr_cascade_search() as deprecated wrapper in simd.rs:
  #[deprecated(note = "use hdr::Cascade::query()")]
  pub fn hdr_cascade_search(...) { Cascade::from_params(...).query(...) }
```

### Lance-Graph Renames

```
FILE:
  light_meter.rs                →  hdr.rs
  
TYPE:
  LightMeter                    →  Cascade
  
METHODS:
  cascade_query()               →  query()
  NEW: expose()                 ←  single distance → band classification
  NEW: test()                   ←  single pair pass/fail

MODULE:
  mod.rs: pub mod light_meter   →  pub mod hdr
```

### The Unified API (identical in both repos)

```rust
// Usage reads the same everywhere:
let hdr = hdr::Cascade::calibrate(&sample_distances);

// Batch query:
let hits = hdr.query(&query, &candidates, 10);

// Single classification:
let band = hdr.expose(distance);

// Single pair test:
let is_close = hdr.test(&a, &b);

// Feed running stats:
hdr.observe(distance);

// Shift detection:
if let Some(shift) = hdr.drift() {
    hdr.recalibrate(&shift);
}

// Types:
hdr::Band::{Foveal, Near, Good, Weak, Reject}
hdr::RankedHit { index: usize, distance: u32, band: Band }
hdr::ShiftAlert { old_mu, new_mu, old_sigma, new_sigma, observations }
hdr::PreciseMode::{Off, Vnni, F32{..}, BF16{..}, DeltaXor{..}, BF16Hamming{..}}
```

---

## PART 1: CROSS-POLLINATION (after renames are done)

### 1. Lance-graph → Rustynum: ReservoirSample + Auto-Switch

**What:** Port `ReservoirSample`, `skewness()`, `kurtosis()`, and the
auto-switch from σ-bands to empirical quantiles.

**Why:** Rustynum's cascade blindly assumes normal distribution. Real
corpora are bimodal (clusters), heavy-tailed (power law entities),
or skewed (popularity distribution). The auto-switch fixes this.

**Where:** `rustynum-core/src/hdr.rs` — add `ReservoirSample` as a field
on `Cascade`, check distribution shape every 1000 observations.

```rust
// Inside Cascade:
reservoir: ReservoirSample,
empirical_bands: [u32; 4],
use_empirical: bool,

// In observe():
if count % 1000 == 0 {
    let skew = self.reservoir.skewness(self.mu, self.sigma);
    let kurt = self.reservoir.kurtosis(self.mu, self.sigma);
    self.use_empirical = skew.abs() >= 2 || kurt < 200 || kurt > 500;
    if self.use_empirical {
        self.empirical_bands = [
            self.reservoir.quantile(0.001),
            self.reservoir.quantile(0.023),
            self.reservoir.quantile(0.159),
            self.reservoir.quantile(0.500),
        ];
    }
}
```

### 2. Lance-graph → Rustynum: Integer Hot Path

**What:** Replace f64 sigma estimation in `Cascade::query()` with integer
arithmetic. Port `isqrt()` from lance-graph.

**Why:** The current rustynum cascade computes `sigma_est` and `s1_reject`
as f64 on every query. That's float on the hot path. Lance-graph proved
it works with u32 bands and integer isqrt.

**Where:** `rustynum-core/src/hdr.rs` — replace:

```rust
// BEFORE (float):
let sigma_est = (vec_bytes as f64) * (8.0 * p_thresh * (1.0 - p_thresh) / s1_bytes as f64).sqrt();
let s1_reject = threshold as f64 + 3.0 * sigma;

// AFTER (integer):
// Pre-calibrated bands. One u32 comparison per candidate. No float.
if projected > self.bands[2] { continue; }  // that's it.
```

### 3. Lance-graph → Rustynum: Persistent Calibration

**What:** Replace per-query 128-sample warmup with persistent `Cascade`
state + Welford shift detection.

**Why:** On a million-query workload, rustynum wastes 128 × 1M = 128 million
SIMD prefix compares just on warmup. With persistent calibration: zero.

**Where:** `rustynum-core/src/hdr.rs` — `Cascade` struct stores mu, sigma,
bands. `calibrate()` called once. `observe()` feeds Welford. `drift()`
returns `ShiftAlert` when distribution changes.

```rust
// BEFORE (per-query warmup):
pub fn hdr_cascade_search(query, db, ...) {
    let warmup_n = 128;
    for i in 0..warmup_n { /* sample to estimate sigma */ }  // EVERY QUERY
    ...
}

// AFTER (persistent):
let hdr = Cascade::calibrate(&initial_sample);  // ONCE
for query in queries {
    let hits = hdr.query(&query, &db, 10);       // no warmup
    for hit in &hits {
        hdr.observe(hit.distance);                // feed Welford
    }
    if let Some(shift) = hdr.drift() {
        hdr.recalibrate(&shift);                  // only when needed
    }
}
```

### 4. Rustynum → Lance-graph: Incremental Stroke 2

**What:** Stage 2 in lance-graph recomputes full distance from scratch.
Rustynum's Stroke 2 computes ONLY the remaining bytes and adds to the
prefix distance.

**Why:** Saves 1/16 of the full compare per survivor. With 5% survivors
on 16K vectors, that's 128 bytes × 5% = 6.4 bytes saved per original
candidate. Small per-candidate, meaningful at scale.

**Where:** `lance-graph/src/graph/blasgraph/hdr.rs`

```rust
// BEFORE (lance-graph, full recompute):
Stage 1: sample 1/16 → projected distance → reject
Stage 2: sample 1/4 → projected distance → reject  
Stage 3: FULL distance from scratch

// AFTER (incremental, from rustynum):
Stage 1: d_prefix = hamming(query[..s1], cand[..s1]) → projected → reject
Stage 2: d_rest = hamming(query[s1..], cand[s1..])   → d_full = d_prefix + d_rest
// Stage 2 IS the full distance, computed incrementally. No Stage 3 needed.
// One fewer pass over the data for every survivor.
```

### 5. Rustynum → Lance-graph: PreciseMode (Stroke 3)

**What:** Lance-graph stops at Hamming. Add optional precision tier
for survivors.

**Why:** After cascade reduces 100K candidates to 200, compute exact
INT8 cosine or BF16 structured Hamming on the 200 survivors. The extra
cost is negligible (200 × 32 cycles = 6400 cycles) but the ranking
quality jumps from "approximate Hamming" to "near-exact cosine."

**Where:** `lance-graph/src/graph/blasgraph/hdr.rs`

```rust
// Add to Cascade:
pub fn query_precise(
    &self,
    query_words: &[u64],
    candidates: &[(usize, &[u64])],
    top_k: usize,
    precise: PreciseMode,  // ← from rustynum
) -> Vec<RankedHit> {
    let mut hits = self.query(query_words, candidates, top_k * 2);  // wider net
    if precise != PreciseMode::Off {
        self.apply_precision(&query_bytes, &candidate_bytes, &mut hits, precise);
    }
    hits.truncate(top_k);
    hits
}
```

Initially support only `PreciseMode::Off` and `PreciseMode::BF16Hamming`.
Others require SIMD dispatch which comes with the BitVec rebuild.

### 6. Rustynum → Lance-graph: SIMD Dispatch

**What:** Lance-graph's `words_hamming()` is a scalar `count_ones()` loop.
Replace with dispatched SIMD.

**Why:** On 16K vectors, VPOPCNTDQ is 8x faster than scalar popcount.
The cascade itself becomes faster, widening the rejection gap.

**Where:** This comes FREE with the 16K BitVec + SIMD rebuild
(FIX_BLASGRAPH_SPO.md). When BitVec gets tiered SIMD, LightMeter/Cascade
calls it automatically. No separate work needed.

**DEFER THIS** until the BitVec rebuild. Don't hand-roll SIMD dispatch
in lance-graph's hdr.rs when it's about to come from BitVec.

---

## EXECUTION ORDER

```
STEP   REPO         WHAT                              RISK    TIME
─────────────────────────────────────────────────────────────────────
 0A    rustynum     simd file rename + shim            low     30 min
 0B-r  rustynum     hdr.rs + rename cascade fns        low     45 min
 0B-l  lance-graph  hdr.rs + rename LightMeter         low     20 min
       BOTH         cargo test --workspace             —       5 min
       BOTH         verify deprecated shims work       —       5 min
─── CHECKPOINT: names unified, all tests pass ──────────────────────
 1     rustynum     Port ReservoirSample + auto-switch med     60 min
 2     rustynum     Integer hot path (replace f64)     med     45 min
 3     rustynum     Persistent calibration             med     60 min
       rustynum     cargo test + cargo bench           —       10 min
─── CHECKPOINT: rustynum has lance-graph innovations ───────────────
 4     lance-graph  Incremental Stroke 2               low     30 min
 5     lance-graph  PreciseMode (Off + BF16Hamming)    med     45 min
       lance-graph  cargo test                         —       5 min
─── CHECKPOINT: lance-graph has rustynum innovations ───────────────
 6     DEFERRED     SIMD dispatch in lance-graph       —       comes with BitVec rebuild
```

### VERIFY AFTER EACH STEP

```bash
# Rustynum:
RUSTFLAGS="-C target-cpu=native" cargo test --workspace
cargo clippy --workspace -- -D warnings

# Lance-graph:
cargo test --workspace
cargo clippy --workspace -- -D warnings

# Cross-check: identical API surface
diff <(grep "pub fn\|pub struct\|pub enum" rustynum-core/src/hdr.rs | sort) \
     <(grep "pub fn\|pub struct\|pub enum" lance-graph/.../hdr.rs | sort)
# Should show only lance-graph-specific extras (words_hamming internal, etc.)
```

---

## DOWNSTREAM DEBT SUMMARY

```
CHANGE                          DEBT                            FIX
────────────────────────────────────────────────────────────────────
simd_compat → simd_avx512      Re-export shim, zero breakage   sed across workspace
                                Deprecation warnings only       Remove shim in v0.5
                                
hdr_cascade_search → Cascade   Deprecated wrapper in simd.rs   Remove in v0.5
                                Old signature still works

LightMeter → Cascade           Internal to lance-graph          No external consumers
                                Zero debt

hamming_search_adaptive         Stays in hdc.rs as wrapper      Delegates to Cascade
                                API unchanged for Python/users

ladybug-rs                     Imports via crate::simd::        Dispatcher unchanged
                                Zero debt                        No action needed
```

**Total downstream debt: two deprecated shims, both removable in next major version.
All existing tests pass unchanged. All existing APIs continue to work.**
