# INTEGRATION_SESSIONS.md

## Session Prompts: From Inventory to Wired System

Each session is self-contained with exact file paths, function signatures,
test criteria, and dependency tracking.

---

## SESSION G: SIMD Clean Refactor

**Prereqs:** None. First to execute.
**Repo:** rustynum
**Branch:** claude/simd-clean

### Task
Replace `rustynum-core/src/simd.rs` (2435 lines, 107 detections) with
`simd_clean.rs` (234 lines, LazyLock, dispatch! macro).

### Exact Steps
1. Copy `.claude/simd_clean.rs` to `rustynum-core/src/simd.rs`
2. Verify all `pub fn` signatures match existing exports
3. Ensure `simd_avx512.rs` has all functions the dispatch calls
4. Ensure `simd_avx2.rs` has all functions the dispatch calls
5. Ensure `scalar_fns.rs` has all functions the dispatch calls
6. Fill gaps: element-wise ops missing from scalar_fns.rs
7. Wire `scalar_fns` into `lib.rs` (`pub mod scalar_fns;`)
8. Add `#[target_feature(enable = "avx512f")]` to every fn in simd_avx512.rs
9. Add `#[target_feature(enable = "avx2,fma")]` to every fn in simd_avx2.rs
10. Remove unnecessary `unsafe` blocks around safe intrinsics (Rust 1.94)
11. Run `cargo test --workspace` — must pass 1543+ tests
12. Run `cargo clippy --workspace -- -D warnings`
13. Run benchmarks: sdot, hamming, plane_distance — compare to PR #102 baseline

### Success Criteria
- simd.rs < 250 lines
- `cargo test` passes (all platforms)
- sdot benchmark ≤ PR #100 baseline (regression from PR #102 fixed)
- No `is_x86_feature_detected!` outside of simd.rs

### Key File
`.claude/simd_clean.rs` — the replacement file (on main)

---

## SESSION H: Plane Evolution (encounter_toward, encounter_away)

**Prereqs:** None (independent of Session G)
**Repo:** rustynum
**Branch:** claude/plane-evolution

### Task
Add directional encounter methods to Plane and Node.
These are the INTEGER equivalents of DreamerV3's STE gradient.

### Exact Changes

In `rustynum-core/src/plane.rs`, add:

```rust
/// Encounter toward another plane's bit pattern.
/// For each bit in other.bits(): if set, push acc[k] toward +1; if clear, toward -1.
/// This IS the DreamerV3 STE gradient expressed as integer accumulation.
pub fn encounter_toward(&mut self, other: &mut Plane) {
    let other_bits = other.bits_bytes_ref();
    for k in 0..Self::BITS {
        let byte_idx = k / 8;
        let bit_idx = k % 8;
        if other_bits[byte_idx] & (1 << bit_idx) != 0 {
            self.acc.values[k] = self.acc.values[k].saturating_add(1);
        } else {
            self.acc.values[k] = self.acc.values[k].saturating_sub(1);
        }
    }
    self.dirty = true;
    self.encounters += 1;
}

/// Encounter AWAY from another plane's bit pattern (repulsive).
/// Opposite direction: if other bit set, push toward -1; if clear, toward +1.
pub fn encounter_away(&mut self, other: &mut Plane) {
    let other_bits = other.bits_bytes_ref();
    for k in 0..Self::BITS {
        let byte_idx = k / 8;
        let bit_idx = k % 8;
        if other_bits[byte_idx] & (1 << bit_idx) != 0 {
            self.acc.values[k] = self.acc.values[k].saturating_sub(1);
        } else {
            self.acc.values[k] = self.acc.values[k].saturating_add(1);
        }
    }
    self.dirty = true;
    self.encounters += 1;
}

/// RL-weighted encounter: reward_sign = +1 for reward, -1 for punishment.
/// Positive reward: encounter_toward. Negative reward: encounter_away.
pub fn reward_encounter(&mut self, evidence: &mut Plane, reward_sign: i8) {
    if reward_sign >= 0 {
        self.encounter_toward(evidence);
    } else {
        self.encounter_away(evidence);
    }
}
```

In `rustynum-core/src/node.rs`, add:

```rust
/// Compute all 7 non-null SPO projections at once.
/// Returns distances for [S__, _P_, __O, SP_, S_O, _PO, SPO].
pub fn project_all(&mut self, other: &mut Node) -> [Distance; 7] {
    let d_s = self.s.distance(&mut other.s);
    let d_p = self.p.distance(&mut other.p);
    let d_o = self.o.distance(&mut other.o);
    
    // Compound projections combine individual distances
    // (details depend on Distance enum implementation)
    [
        self.distance(other, S__),
        self.distance(other, _P_),
        self.distance(other, __O),
        self.distance(other, SP_),
        self.distance(other, S_O),
        self.distance(other, _PO),
        self.distance(other, SPO),
    ]
}

/// RL credit assignment: given BF16 exponent bits,
/// encounter toward matching projections, away from failing ones.
pub fn credit_assignment(&mut self, other: &mut Node, exponent: u8) {
    // Bit 1 = S__ matched
    if exponent & 0b00000010 != 0 {
        self.s.encounter_toward(&mut other.s);
    } else {
        self.s.encounter_away(&mut other.s);
    }
    // Bit 2 = _P_ matched
    if exponent & 0b00000100 != 0 {
        self.p.encounter_toward(&mut other.p);
    } else {
        self.p.encounter_away(&mut other.p);
    }
    // Bit 3 = __O matched
    if exponent & 0b00001000 != 0 {
        self.o.encounter_toward(&mut other.o);
    } else {
        self.o.encounter_away(&mut other.o);
    }
}
```

### Tests
```rust
#[test]
fn encounter_toward_converges() {
    let mut a = Plane::new();
    let mut b = Plane::random(42);
    // After 10 encounters toward b, a.bits should approach b.bits
    for _ in 0..10 { a.encounter_toward(&mut b); }
    let d = a.distance(&mut b);
    assert!(d.raw().unwrap() < PLANE_BITS as u32 / 4); // < 25% disagreement
}

#[test]
fn encounter_away_diverges() {
    let mut a = Plane::random(42);
    let mut b = a.clone();
    // After encountering AWAY, distance should increase
    let d_before = a.distance(&mut b).raw().unwrap();
    for _ in 0..10 { a.encounter_away(&mut b); }
    let d_after = a.distance(&mut b).raw().unwrap();
    assert!(d_after > d_before);
}

#[test]
fn credit_assignment_selective() {
    let mut a = Node::random(1);
    let mut b = Node::random(2);
    let d_s_before = a.s.distance(&mut b.s).raw().unwrap();
    let d_p_before = a.p.distance(&mut b.p).raw().unwrap();
    
    // Exponent says P matched (bit 2), S didn't (bit 1 = 0)
    a.credit_assignment(&mut b, 0b00000100);
    
    let d_s_after = a.s.distance(&mut b.s).raw().unwrap();
    let d_p_after = a.p.distance(&mut b.p).raw().unwrap();
    
    // S should diverge (punished), P should converge (rewarded)
    assert!(d_s_after >= d_s_before); // may equal if already at boundary
    assert!(d_p_after <= d_p_before);
}
```

---

## SESSION I: BF16 Truth Assembly

**Prereqs:** Session H (project_all, credit_assignment)
**Repo:** rustynum
**Branch:** claude/bf16-truth

### Task
Build the BF16 value from 2³ projections. Integer only. No float arithmetic.

### Exact Changes

In `rustynum-core/src/bf16_hamming.rs`, add:

```rust
/// Assemble BF16 truth value from 7 SPO projections.
/// sign = causality direction (0 = causing, 1 = caused)
/// exponent = which projections are in Foveal or Near band
/// mantissa = finest distance of best matching projection, normalized to 7 bits
pub fn bf16_from_projections(
    projections: &[Band; 7],
    finest_distance: u32,
    band_foveal_max: u32,
    causality: CausalityDirection,
) -> u16 {
    let sign: u16 = match causality {
        CausalityDirection::Causing => 0,
        CausalityDirection::Experiencing => 1,
    };
    
    let mut exponent: u16 = 0;
    for (i, band) in projections.iter().enumerate() {
        match band {
            Band::Foveal | Band::Near => exponent |= 1 << (i + 1),
            _ => {}
        }
    }
    
    // Mantissa: finest_distance normalized to 7 bits
    let mantissa: u16 = if band_foveal_max > 0 {
        ((finest_distance as u64 * 127) / band_foveal_max as u64).min(127) as u16
    } else {
        0
    };
    
    (sign << 15) | (exponent << 7) | mantissa
}

/// Extract 8-bit exponent from BF16 value. Integer shift only.
#[inline(always)]
pub fn bf16_extract_exponent(bf16: u16) -> u8 {
    ((bf16 >> 7) & 0xFF) as u8
}

/// Extract sign bit (causality direction).
#[inline(always)]
pub fn bf16_extract_sign(bf16: u16) -> CausalityDirection {
    if bf16 & 0x8000 != 0 {
        CausalityDirection::Experiencing
    } else {
        CausalityDirection::Causing
    }
}

/// Extract 7-bit mantissa (finest distance).
#[inline(always)]
pub fn bf16_extract_mantissa(bf16: u16) -> u8 {
    (bf16 & 0x7F) as u8
}
```

### NarsTruth as BF16 Pair (32 bits)

In `rustynum-core/src/causality.rs`, add:

```rust
/// NARS truth value packed as two BF16 values in 32 bits.
/// Upper 16 bits: BF16 frequency. Lower 16 bits: BF16 confidence.
/// Fits in one f32 slot. Compatible with VDPBF16PS for revision.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PackedNarsTruth(pub u32);

impl PackedNarsTruth {
    pub fn new(frequency: f32, confidence: f32) -> Self {
        let f_bits = ((frequency.to_bits() >> 16) & 0xFFFF) as u16;  // truncate to BF16
        let c_bits = ((confidence.to_bits() >> 16) & 0xFFFF) as u16;
        Self(((f_bits as u32) << 16) | (c_bits as u32))
    }
    
    pub fn frequency_f32(&self) -> f32 {
        f32::from_bits(((self.0 >> 16) as u32) << 16)
    }
    
    pub fn confidence_f32(&self) -> f32 {
        f32::from_bits((self.0 & 0xFFFF) << 16)
    }
    
    /// Convert from existing NarsTruthValue
    pub fn from_truth(t: &NarsTruthValue) -> Self {
        Self::new(t.frequency, t.confidence)
    }
}
```

---

## SESSION J: PackedDatabase (GEMM Panel Packing for Cascades)

**Prereqs:** None (independent)
**Repo:** rustynum
**Branch:** claude/packed-database

### Task
Implement stroke-aligned database layout, analogous to GEMM panel packing
(siboehm article). Candidate data stored contiguously per stroke region
for sequential streaming instead of scattered per-candidate access.

### Key Reference
`.claude/CASCADE_TETRIS.md` — full spec with pseudocode
`.claude/L1_CACHE_BOUNDARY.md` — L1 constraints

### Implementation in rustynum-core/src/hdr.rs (or new file packed_db.rs)

```rust
pub struct PackedDatabase {
    stroke1: Vec<u8>,  // all candidates' [0..128] contiguous
    stroke2: Vec<u8>,  // all candidates' [128..512] contiguous  
    stroke3: Vec<u8>,  // all candidates' [512..2048] contiguous
    n_candidates: usize,
    bytes_per_candidate: usize,
}

impl PackedDatabase {
    pub fn pack(candidates: &[&[u8]]) -> Self { ... }
    
    pub fn cascade_scan(
        &self,
        query: &[u8],
        bands: &[u32; 4],
        k: usize,
    ) -> Vec<(usize, u32)> { ... }
}
```

### Benchmark
Compare `Cascade::query()` vs `PackedDatabase::cascade_scan()` on 100K, 1M candidates.
Target: 2-3x improvement from sequential vs scattered access.

---

## SESSION K: Message Passing (Binary GNN)

**Prereqs:** Session H (encounter_toward/away)
**Repo:** rustynum (new file: rustynum-core/src/message_pass.rs)
**Branch:** claude/message-passing

### Task
Implement K-round binary GNN message passing using encounter() as aggregation.
This is the BitGNN equivalent but deterministic and CPU-only.

### Key Reference
`.claude/RESEARCH_THREADS.md` Thread 4 — BitGNN connection

### Implementation

```rust
pub struct SpoGraph {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,  // compact adjacency
}

pub struct Edge {
    pub source: u32,
    pub target: u32,
    pub truth: PackedNarsTruth,
    pub projection: u8,  // which SPO projections match (BF16 exponent)
}

impl SpoGraph {
    pub fn message_passing(&mut self, rounds: usize) {
        for _ in 0..rounds {
            // Collect messages, apply encounters
            // Each node's planes evolve based on neighbor influence
            // Weighted by edge truth confidence
        }
    }
}
```

---

## SESSION L: Wiring the Full Pipeline

**Prereqs:** Sessions H, I, J, K
**Repo:** rustynum
**Branch:** claude/full-pipeline

### Task
Wire: cascade → projections → BF16 → encounter → NARS

### Implementation (new file: rustynum-core/src/rl_step.rs)

```rust
/// One complete RL step. Deterministic. Integer only (except f32 hydration).
pub fn rl_step(
    query_node: &mut Node,
    candidate_node: &mut Node,
    cascade: &Cascade,
) -> (u16, PackedNarsTruth) {
    // 1. Compute 7 projections
    let projections = query_node.project_all(candidate_node);
    
    // 2. Band classify each projection
    let bands = projections.map(|d| cascade.expose(d.raw().unwrap_or(u32::MAX)));
    
    // 3. Assemble BF16 truth
    let bf16 = bf16_from_projections(&bands, finest, foveal_max, direction);
    
    // 4. Credit assignment (RL gradient)
    let exponent = bf16_extract_exponent(bf16);
    query_node.credit_assignment(candidate_node, exponent);
    
    // 5. NARS truth from plane truths
    let truth = PackedNarsTruth::from_truth(&query_node.truth(SPO).into());
    
    (bf16, truth)
}
```

### End-to-End Test
```rust
#[test]
fn rl_converges_on_known_pattern() {
    let mut query = Node::random(1);
    let mut candidate = query.clone(); // identical = should converge to Foveal
    let cascade = Cascade::calibrate(&[0, 100, 200, 500, 1000], 2048);
    
    for _ in 0..100 {
        let (bf16, truth) = rl_step(&mut query, &mut candidate, &cascade);
        // Should converge: exponent all 1s, mantissa → 0, truth → high
    }
    
    let exp = bf16_extract_exponent(bf16);
    assert_eq!(exp & 0b11111110, 0b11111110); // all projections match
}
```

---

## TOTAL EFFORT ESTIMATE

```
SESSION    LINES    DAYS    DEPENDENCIES
G (simd)    ~100     1      none
H (plane)   ~200     1      none
I (bf16)    ~200     1      H
J (packed)  ~300     1      none
K (msgpass) ~300     2      H
L (wire)    ~200     1      H, I

Total:     ~1300    ~7 days
```

Each session can run as a focused CC session with its own branch.
Sessions G, H, J are independent — can run in parallel.
Sessions I, K depend on H.
Session L depends on H + I.

The result: the first fully binary RL + GNN + semiring graph system.
Deterministic. CPU-only. ~1300 lines of new code.
