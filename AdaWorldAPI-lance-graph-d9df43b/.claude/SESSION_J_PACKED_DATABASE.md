# SESSION_J: PackedDatabase — Panel Packing for Cascade Search

## Mission: Repack candidate database so cascade strokes are contiguous in memory

**The benchmark proved the problem:**
```
Batch hamming 1K candidates:  221M cand/s (fits cache, scattered OK)
Batch hamming 1M candidates:   57M cand/s (L3 bandwidth bound, scattered kills us)
                               3.9x degradation from cache pressure
```

**The fix is siboehm's panel packing applied to binary search instead of GEMM.**

siboehm's GEMM optimization chain (https://siboehm.com/articles/22/Fast-MMM-on-CPU):
```
Naive:           4481ms
Loop reorder:      89ms  (50x — cache-aware inner loop)
L1 tiling:         70ms  (1.3x — tile to L1 size)  
Panel packing:     35ms  (2x — sequential memory access)
MKL:                8ms  (4x — register blocking + prefetch + threading)
```

Panel packing alone = 2x. That's our low-hanging fruit.
Applied to cascade: 57M → ~110M candidates/s. Maybe more because
the prefetcher is PERFECT for sequential access patterns.

---

## THE PARALLEL: GEMM Panel Packing = Cascade Stroke Packing

```
GEMM (siboehm):                       CASCADE (us):
─────────────────────────────────────────────────────────────
Matrix B (N×K)                         Database (N candidates × 2KB each)
Column panel (K×NR block)              Stroke region (128B × N block)
Panel packing: reorder B               Stroke packing: reorder database
  so panel columns are contiguous        so stroke-1 bytes are contiguous
Inner loop: sequential FMA              Inner loop: sequential XOR+popcount
  across contiguous panel                  across contiguous stroke
Prefetcher handles sequential           Prefetcher handles sequential
Register accumulators (6×NR)            ZMM accumulators (8 × u64 popcount)
```

---

## STEP 1: Define the Stroke Layout

A fingerprint is 2048 bytes (16384 bits). The cascade processes it in strokes:

```
STROKE 1:  bytes [0..128)      = 1024 bits    coarse rejection (~90% eliminated)
STROKE 2:  bytes [128..512)    = 3072 bits    medium filter (~90% of survivors)
STROKE 3:  bytes [512..2048)   = 12288 bits   precise distance (final ranking)

Total: 128 + 384 + 1536 = 2048 bytes = full fingerprint
Non-overlapping. Incremental. Each stroke refines the previous.
```

### Current Layout (scattered)
```
database: &[u8]  // N candidates × 2048 bytes, interleaved

candidate[0]: [stroke1₁₂₈ | stroke2₃₈₄ | stroke3₁₅₃₆]  byte 0..2048
candidate[1]: [stroke1₁₂₈ | stroke2₃₈₄ | stroke3₁₅₃₆]  byte 2048..4096
candidate[2]: [stroke1₁₂₈ | stroke2₃₈₄ | stroke3₁₅₃₆]  byte 4096..6144
...

Stroke 1 scan: read bytes [0..128], skip 1920, read [2048..2176], skip 1920, ...
  Stride: 2048 bytes. Prefetcher: confused. Cache: thrashing.
```

### Packed Layout (contiguous strokes)
```
packed.stroke1: &[u8]  // N × 128 bytes, contiguous
packed.stroke2: &[u8]  // N × 384 bytes, contiguous  
packed.stroke3: &[u8]  // N × 1536 bytes, contiguous
packed.index:   &[u32] // original candidate IDs (for result mapping)

packed.stroke1: [cand[0]₀₋₁₂₈ | cand[1]₀₋₁₂₈ | cand[2]₀₋₁₂₈ | ...]
packed.stroke2: [cand[0]₁₂₈₋₅₁₂ | cand[1]₁₂₈₋₅₁₂ | cand[2]₁₂₈₋₅₁₂ | ...]
packed.stroke3: [cand[0]₅₁₂₋₂₀₄₈ | cand[1]₅₁₂₋₂₀₄₈ | cand[2]₅₁₂₋₂₀₄₈ | ...]

Stroke 1 scan: sequential read through packed.stroke1
  Stride: 0. Prefetcher: perfect. Cache: streaming.
```

---

## STEP 2: Implement PackedDatabase

```rust
/// Stroke-aligned database layout for streaming cascade search.
/// Panel packing (siboehm GEMM analogy): reorder data so each stroke
/// is contiguous across all candidates. Prefetcher handles sequential.
pub struct PackedDatabase {
    /// Stroke 1: first 128 bytes of each candidate, contiguous
    stroke1: Vec<u8>,       // N × STROKE1_BYTES
    /// Stroke 2: bytes 128..512 of each candidate, contiguous
    stroke2: Vec<u8>,       // N × STROKE2_BYTES
    /// Stroke 3: bytes 512..2048 of each candidate, contiguous
    stroke3: Vec<u8>,       // N × STROKE3_BYTES
    /// Original candidate indices (for result mapping back to source)
    index: Vec<u32>,        // N entries
    /// Number of candidates
    num_candidates: usize,
}

pub const STROKE1_BYTES: usize = 128;   // 1024 bits
pub const STROKE2_BYTES: usize = 384;   // 3072 bits  
pub const STROKE3_BYTES: usize = 1536;  // 12288 bits
pub const FINGERPRINT_BYTES: usize = 2048; // 16384 bits total

impl PackedDatabase {
    /// Pack a flat database into stroke-aligned layout.
    /// This is the "panel packing" step — done ONCE at database construction.
    /// O(N × 2048) memory copy. Amortized over all subsequent queries.
    pub fn pack(database: &[u8], row_bytes: usize) -> Self {
        assert_eq!(row_bytes, FINGERPRINT_BYTES);
        let n = database.len() / row_bytes;
        
        let mut stroke1 = Vec::with_capacity(n * STROKE1_BYTES);
        let mut stroke2 = Vec::with_capacity(n * STROKE2_BYTES);
        let mut stroke3 = Vec::with_capacity(n * STROKE3_BYTES);
        let mut index = Vec::with_capacity(n);
        
        for i in 0..n {
            let base = i * row_bytes;
            stroke1.extend_from_slice(&database[base..base + STROKE1_BYTES]);
            stroke2.extend_from_slice(&database[base + STROKE1_BYTES..base + STROKE1_BYTES + STROKE2_BYTES]);
            stroke3.extend_from_slice(&database[base + STROKE1_BYTES + STROKE2_BYTES..base + row_bytes]);
            index.push(i as u32);
        }
        
        Self { stroke1, stroke2, stroke3, index, num_candidates: n }
    }
    
    /// Get stroke 1 slice for candidate i (128 bytes)
    #[inline(always)]
    pub fn get_stroke1(&self, i: usize) -> &[u8] {
        &self.stroke1[i * STROKE1_BYTES..(i + 1) * STROKE1_BYTES]
    }
    
    /// Get stroke 2 slice for candidate i (384 bytes)
    #[inline(always)]
    pub fn get_stroke2(&self, i: usize) -> &[u8] {
        &self.stroke2[i * STROKE2_BYTES..(i + 1) * STROKE2_BYTES]
    }
    
    /// Get stroke 3 slice for candidate i (1536 bytes)
    #[inline(always)]
    pub fn get_stroke3(&self, i: usize) -> &[u8] {
        &self.stroke3[i * STROKE3_BYTES..(i + 1) * STROKE3_BYTES]
    }
    
    /// Original candidate ID for result mapping
    #[inline(always)]
    pub fn original_id(&self, i: usize) -> u32 {
        self.index[i]
    }
    
    pub fn num_candidates(&self) -> usize {
        self.num_candidates
    }
}
```

---

## STEP 3: Implement Packed Cascade Query

The cascade query on PackedDatabase is fundamentally different from the
scattered version. Each stroke is a SEQUENTIAL SCAN with early exit.

```rust
impl PackedDatabase {
    /// Cascade query on packed layout. Three-stroke early rejection.
    /// Stroke 1: sequential scan of packed.stroke1 → reject ~90%
    /// Stroke 2: sequential scan of packed.stroke2 for survivors → reject ~90%  
    /// Stroke 3: sequential scan of packed.stroke3 for survivors → exact distance
    pub fn cascade_query(
        &self,
        query: &[u8],          // full 2048-byte fingerprint
        cascade: &Cascade,     // for band classification
        k: usize,              // top-k results
    ) -> Vec<RankedHit> {
        let query_s1 = &query[..STROKE1_BYTES];
        let query_s2 = &query[STROKE1_BYTES..STROKE1_BYTES + STROKE2_BYTES];
        let query_s3 = &query[STROKE1_BYTES + STROKE2_BYTES..];
        
        // ── STROKE 1: coarse rejection ─────────────────────────
        // Sequential scan through packed stroke1 data.
        // Prefetcher is PERFECT here — pure sequential read.
        // VPOPCNTDQ processes 64 bytes per instruction = 2 instructions per candidate.
        let reject_threshold_s1 = cascade.reject_threshold_for_bytes(STROKE1_BYTES);
        
        let mut survivors: Vec<(usize, u64)> = Vec::with_capacity(self.num_candidates / 10);
        
        // Batch hamming on contiguous stroke1 data
        for i in 0..self.num_candidates {
            let cand_s1 = self.get_stroke1(i);
            let d1 = simd::hamming_distance(query_s1, cand_s1);
            if d1 < reject_threshold_s1 {
                survivors.push((i, d1));
            }
            // Prefetch next candidate's stroke1 (sequential, prefetcher should handle it,
            // but explicit prefetch doesn't hurt)
            if i + 4 < self.num_candidates {
                unsafe {
                    core::arch::x86_64::_mm_prefetch(
                        self.stroke1[(i + 4) * STROKE1_BYTES..].as_ptr() as *const i8,
                        core::arch::x86_64::_MM_HINT_T0,
                    );
                }
            }
        }
        
        // ── STROKE 2: medium filter ────────────────────────────
        // Only scan survivors. Still sequential within packed.stroke2.
        let reject_threshold_s2 = cascade.reject_threshold_for_bytes(
            STROKE1_BYTES + STROKE2_BYTES
        );
        
        let mut survivors2: Vec<(usize, u64)> = Vec::with_capacity(survivors.len() / 10);
        
        for &(idx, d1) in &survivors {
            let cand_s2 = self.get_stroke2(idx);
            let d2 = simd::hamming_distance(query_s2, cand_s2);
            let d_cumulative = d1 + d2;  // cumulative distance through stroke 2
            if d_cumulative < reject_threshold_s2 {
                survivors2.push((idx, d_cumulative));
            }
        }
        
        // ── STROKE 3: precise distance ─────────────────────────
        // Final ranking on remaining survivors.
        let mut results: Vec<RankedHit> = Vec::with_capacity(survivors2.len());
        
        for &(idx, d12) in &survivors2 {
            let cand_s3 = self.get_stroke3(idx);
            let d3 = simd::hamming_distance(query_s3, cand_s3);
            let d_total = d12 + d3;  // full hamming distance
            let band = cascade.expose(d_total);
            results.push(RankedHit {
                index: self.original_id(idx) as usize,
                distance: d_total,
                band,
            });
        }
        
        // Sort and take top-k
        results.sort_unstable_by_key(|h| h.distance);
        results.truncate(k);
        results
    }
}
```

---

## STEP 4: Implement with array_windows (Rust 1.94)

The stroke scan can use `array_chunks` for typed, monomorphized inner loops:

```rust
/// Process stroke 1 using array_chunks for compile-time-known sizes.
/// LLVM sees [u8; 128] → full unrolling, no bounds checks.
fn stroke1_scan_typed(
    query_s1: &[u8; STROKE1_BYTES],
    packed_stroke1: &[u8],
    num_candidates: usize,
    threshold: u64,
) -> Vec<(usize, u64)> {
    let mut survivors = Vec::with_capacity(num_candidates / 10);
    
    // array_chunks gives us &[u8; 128] per candidate — compile-time known size
    for (i, chunk) in packed_stroke1.array_chunks::<STROKE1_BYTES>().enumerate() {
        let d = simd::hamming_distance(query_s1, chunk);
        if d < threshold {
            survivors.push((i, d));
        }
    }
    
    survivors
}
```

The compiler knows `chunk` is exactly 128 bytes. It generates the optimal
VPOPCNTDQ loop (2 × 64-byte loads) with zero bounds checking.
This is what JIT would produce. The compiler does it at build time.

---

## STEP 5: Benchmark PackedDatabase vs Scattered

### Test harness

```rust
// Setup: 1M candidates, 2KB fingerprints
let database: Vec<u8> = generate_random_database(1_000_000, FINGERPRINT_BYTES);
let query: Vec<u8> = generate_random_query(FINGERPRINT_BYTES);
let cascade = Cascade::calibrate_from_sample(&database, FINGERPRINT_BYTES);

// Pack once (amortized)
let packed = PackedDatabase::pack(&database, FINGERPRINT_BYTES);

// Benchmark scattered (current)
let t0 = Instant::now();
for _ in 0..100 {
    let _hits = cascade.query(&query, &database, 10, threshold);
}
let scattered_time = t0.elapsed() / 100;

// Benchmark packed (new)
let t0 = Instant::now();
for _ in 0..100 {
    let _hits = packed.cascade_query(&query, &cascade, 10);
}
let packed_time = t0.elapsed() / 100;

println!("Scattered: {:?}", scattered_time);
println!("Packed:    {:?}", packed_time);
println!("Speedup:   {:.2}x", scattered_time.as_nanos() as f64 / packed_time.as_nanos() as f64);
```

### Expected results (based on siboehm's 2x from panel packing)

```
CANDIDATES    SCATTERED       PACKED          SPEEDUP
1K            4μs             3μs             1.3x (already fits cache)
10K           49μs            25μs            2.0x (prefetcher kicks in)
100K          714μs           280μs           2.5x (streaming dominates)
1M            8327μs          3200μs          2.6x (pure streaming)
```

The speedup increases with database size because the prefetcher advantage
grows when the working set exceeds L2/L3. At 1K candidates the data
already fits in cache so layout doesn't matter much.

### Also benchmark the packing cost

```rust
let t0 = Instant::now();
let packed = PackedDatabase::pack(&database, FINGERPRINT_BYTES);
let pack_time = t0.elapsed();
println!("Pack time for 1M: {:?}", pack_time);
// Expected: ~2ms (memcpy of 1M × 2KB = 2GB, sequential)
// Amortized over 1000s of queries: negligible
```

---

## STEP 6: Memory Layout Analysis

```
SCATTERED (current):
  Total memory: N × 2048 bytes
  Stroke 1 access pattern: stride = 2048 bytes
  Cache lines loaded per stroke-1 candidate: 2 (128 bytes / 64 byte line)
  Cache lines WASTED per candidate: 30 (remaining 1920 bytes in same lines)
  Effective bandwidth utilization: 128/2048 = 6.25% during stroke 1
  
PACKED:
  Total memory: N × 2048 bytes (same total, different layout)
  Stroke 1 access pattern: stride = 0 (contiguous)
  Cache lines loaded per stroke-1 candidate: 2 (128 bytes / 64 byte line)
  Cache lines WASTED per candidate: 0 (next candidate is adjacent)
  Effective bandwidth utilization: 100% during stroke 1

  Stroke 1 only (90% rejection):
    Scattered: reads N × 2048 bytes of cache lines (wastes 94%)
    Packed: reads N × 128 bytes of cache lines (wastes 0%)
    Memory bandwidth reduction: 16x for stroke 1 alone
    
  With 90% rejection per stroke:
    Scattered total reads: N×2048 (stroke 1 pollutes everything)
    Packed total reads: N×128 + 0.1N×384 + 0.01N×1536 = N×(128+38.4+15.36) = N×181.8
    Memory bandwidth reduction: 2048/181.8 = 11.3x
```

The 11.3x memory bandwidth reduction is why panel packing gives 2-3x speedup
even though the CPU is already fast. The bottleneck at 1M candidates is
DRAM bandwidth, not compute. Packing eliminates 89% of DRAM reads.

---

## STEP 7: SPO Node Packing (extend to 3-plane Nodes)

For Node queries (S + P + O), the packing extends to SPO projections:

```rust
/// Packed database for Node-level cascade (3 planes per candidate)
pub struct PackedNodeDatabase {
    /// Each plane has its own 3-stroke packing
    s_stroke1: Vec<u8>,   // N × 128 bytes
    s_stroke2: Vec<u8>,   // N × 384 bytes
    s_stroke3: Vec<u8>,   // N × 1536 bytes
    p_stroke1: Vec<u8>,   // N × 128 bytes
    p_stroke2: Vec<u8>,   // N × 384 bytes
    p_stroke3: Vec<u8>,   // N × 1536 bytes
    o_stroke1: Vec<u8>,   // N × 128 bytes
    o_stroke2: Vec<u8>,   // N × 384 bytes
    o_stroke3: Vec<u8>,   // N × 1536 bytes
    index: Vec<u32>,
    num_candidates: usize,
}
```

This enables Mask-aware cascade: when searching with mask=S__ (subject only),
only s_stroke1/2/3 are accessed. P and O data never enters cache.

```
CURRENT NODE SEARCH (mask=S__):
  Read 6KB per candidate (full S+P+O), only use 2KB (S plane)
  Bandwidth waste: 67%
  
PACKED NODE SEARCH (mask=S__):
  Read s_stroke1 only: 128 bytes per candidate
  Bandwidth waste: 0%
  Speedup vs scattered: ~16x for stroke 1 (128 vs 2048+4096 bytes accessed)
```

---

## STEP 8: Lance Integration

PackedDatabase is a natural fit for Lance columnar storage:

```
LANCE DATASET:
  Column "stroke1": FixedSizeBinary(128) × N  ← one column per stroke
  Column "stroke2": FixedSizeBinary(384) × N
  Column "stroke3": FixedSizeBinary(1536) × N
  Column "node_id": UInt32 × N

Each column IS a packed stroke. Lance stores columns contiguously.
Reading stroke1 column = reading packed.stroke1. Zero transformation needed.
The Lance columnar format IS panel packing. We just use it correctly.
```

---

## WHERE THIS GOES

```
rustynum-core/src/packed.rs          — PackedDatabase + PackedNodeDatabase
rustynum-core/src/hdr.rs             — cascade_query_packed() method
ndarray/src/hpc/packed.rs            — ndarray port (after rustynum is verified)
lance-graph/                          — Lance columnar integration
```

## DEPENDENCIES

```
NEEDS:
  simd::hamming_distance (dispatch!, verified, merged)
  Cascade::expose() (band classification, exists)
  Cascade reject thresholds per stroke size (NEW: derive from band boundaries)

DOES NOT NEED:
  BF16 (that's a separate system)
  encounter() (PackedDatabase is read-only search)
  Node/Plane types (PackedDatabase operates on raw &[u8])
```

## COMPLETION CRITERIA

- [ ] PackedDatabase::pack() transposes scattered → contiguous strokes
- [ ] cascade_query_packed() with 3-stroke early rejection
- [ ] array_chunks::<128>() for typed stroke 1 scan (Rust 1.94)
- [ ] Benchmark: packed vs scattered at 1K, 10K, 100K, 1M candidates
- [ ] Memory bandwidth analysis matches prediction (11.3x reduction)
- [ ] Pack time benchmarked and documented (must be < 10ms for 1M)
- [ ] SPO node packing variant for Mask-aware queries
- [ ] All existing cascade tests pass (backward compatible)
- [ ] New tests: pack → query → results match unpacked query exactly
