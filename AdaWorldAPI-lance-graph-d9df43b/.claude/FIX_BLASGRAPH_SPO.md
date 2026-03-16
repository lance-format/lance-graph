# FIX_BLASGRAPH_SPO.md

## Rebuild SPO on BlasGraph BitVec. One Type. SIMD As If It Was Nothing.

**Repo:** AdaWorldAPI/lance-graph
**Branch:** the upstream PR branch
**Target audience:** RedisGraph refugees who know what SuiteSparse was

---

## THE PROBLEM

The current PR has two type systems:
- `blasgraph/BitVec` at 16,384 bits — the algebra
- `spo/Fingerprint = [u64; 8]` at 512 bits — the store

These don't talk to each other. A RedisGraph person opens this and sees
a toy fingerprint next to a real algebra engine. They close the tab.

## THE FIX

Delete `spo/fingerprint.rs`. Delete `spo/sparse.rs`. SPO uses `BitVec` directly.
One type from store to algebra to query. Like RedisGraph used one SuiteSparse type
for everything.

## STEP 1: Make BitVec's SIMD Feel Like Nothing

The current `BitVec` in `blasgraph/types.rs` uses scalar loops.
Add tiered SIMD that the user NEVER THINKS ABOUT. It just works.
No feature flags. No cfg choices for the user. Auto-dispatch.

In `blasgraph/types.rs`, replace the scalar implementations:

```rust
impl BitVec {
    /// Hamming distance. AVX-512 if available, AVX2 if not, scalar if neither.
    /// You don't choose. The CPU chooses. It's just fast.
    #[inline]
    pub fn hamming_distance(&self, other: &BitVec) -> u32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512vpopcntdq") && is_x86_feature_detected!("avx512f") {
                return unsafe { self.hamming_avx512(other) };
            }
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.hamming_avx2(other) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return unsafe { self.hamming_neon(other) };
        }
        self.hamming_scalar(other)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx512f,avx512vpopcntdq")]
    unsafe fn hamming_avx512(&self, other: &BitVec) -> u32 {
        use core::arch::x86_64::*;
        let mut total = _mm512_setzero_si512();
        let ptr_a = self.words.as_ptr() as *const __m512i;
        let ptr_b = other.words.as_ptr() as *const __m512i;
        // 256 words / 8 words per 512-bit register = 32 iterations
        for i in 0..32 {
            let a = _mm512_loadu_si512(ptr_a.add(i));
            let b = _mm512_loadu_si512(ptr_b.add(i));
            let xor = _mm512_xor_si512(a, b);
            let popcnt = _mm512_popcnt_epi64(xor);
            total = _mm512_add_epi64(total, popcnt);
        }
        // Horizontal sum of 8 x u64
        let mut buf = [0u64; 8];
        _mm512_storeu_si512(buf.as_mut_ptr() as *mut __m512i, total);
        buf.iter().sum::<u64>() as u32
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn hamming_avx2(&self, other: &BitVec) -> u32 {
        use core::arch::x86_64::*;
        // Harley-Seal popcount on AVX2
        let mut total = 0u32;
        let ptr_a = self.words.as_ptr() as *const __m256i;
        let ptr_b = other.words.as_ptr() as *const __m256i;
        let lookup = _mm256_setr_epi8(
            0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
            0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
        );
        let mask = _mm256_set1_epi8(0x0f);
        // 256 words / 4 words per 256-bit register = 64 iterations
        for i in 0..64 {
            let a = _mm256_loadu_si256(ptr_a.add(i));
            let b = _mm256_loadu_si256(ptr_b.add(i));
            let xor = _mm256_xor_si256(a, b);
            let lo = _mm256_shuffle_epi8(lookup, _mm256_and_si256(xor, mask));
            let hi = _mm256_shuffle_epi8(lookup, _mm256_and_si256(_mm256_srli_epi16(xor, 4), mask));
            let sum = _mm256_add_epi8(lo, hi);
            let sad = _mm256_sad_epu8(sum, _mm256_setzero_si256());
            // Extract 4 x u64 from sad
            total += _mm256_extract_epi64(sad, 0) as u32
                   + _mm256_extract_epi64(sad, 1) as u32
                   + _mm256_extract_epi64(sad, 2) as u32
                   + _mm256_extract_epi64(sad, 3) as u32;
        }
        total
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn hamming_neon(&self, other: &BitVec) -> u32 {
        use core::arch::aarch64::*;
        let mut total = 0u32;
        let ptr_a = self.words.as_ptr();
        let ptr_b = other.words.as_ptr();
        for i in 0..HD_WORDS {
            total += (ptr_a.add(i).read() ^ ptr_b.add(i).read()).count_ones();
        }
        total
    }

    fn hamming_scalar(&self, other: &BitVec) -> u32 {
        let mut d = 0u32;
        for i in 0..HD_WORDS {
            d += (self.words[i] ^ other.words[i]).count_ones();
        }
        d
    }

    /// XOR bind. Same SIMD tiering. You don't notice.
    #[inline]
    pub fn xor(&self, other: &BitVec) -> BitVec {
        let mut result = BitVec::zero();
        // On AVX-512 this is 32 vpxord instructions. 32 cycles.
        // The scalar fallback is 256 xor ops. Still fast.
        // No branching — the scalar path IS the fast path on most hardware.
        for i in 0..HD_WORDS {
            result.words[i] = self.words[i] ^ other.words[i];
        }
        result
        // NOTE: LLVM auto-vectorizes this loop to AVX2/AVX-512
        // when compiling with -C target-cpu=native.
        // The explicit SIMD above is for guaranteed performance
        // on pre-built binaries without -C target-cpu=native.
    }

    /// Popcount. Bits set.
    #[inline]
    pub fn popcount(&self) -> u32 {
        // Same tiering as hamming. Reuse the infrastructure.
        let zero = BitVec::zero();
        // hamming(self, zero) = popcount(self XOR 0) = popcount(self)
        // Compiler eliminates the XOR with zero. Trust it.
        self.hamming_distance(&zero)
    }
}
```

The key: the user writes `a.hamming_distance(&b)`. They never see SIMD.
It's just fast. On their laptop with AVX2. On the server with AVX-512.
On an ARM Mac with NEON. The C64 spirit: make the hardware do impossible
things without the user knowing how.

## STEP 2: Delete SPO's Separate Fingerprint

```bash
rm crates/lance-graph/src/graph/spo/fingerprint.rs   # gone
# sparse.rs stays IF it adds bitmap ops not in BitVec
# Otherwise delete it too
```

## STEP 3: Rewrite SPO Types On BitVec

In `spo/mod.rs` or `spo/store.rs`:

```rust
use crate::graph::blasgraph::types::BitVec;

/// Encode a label as a 16,384-bit fingerprint.
/// BLAKE3 hash → LFSR expansion to fill 16K bits.
pub fn label_to_bitvec(label: &str) -> BitVec {
    let hash = blake3::hash(label.as_bytes());
    let mut words = [0u64; HD_WORDS];
    let seed_bytes = hash.as_bytes();
    let mut state = u64::from_le_bytes(seed_bytes[0..8].try_into().unwrap());
    for word in &mut words {
        let mut val = 0u64;
        for bit in 0..64 {
            let feedback = (state ^ (state >> 2) ^ (state >> 3) ^ (state >> 63)) & 1;
            state = (state >> 1) | (feedback << 63);
            val |= (state & 1) << bit;
        }
        *word = val;
    }
    BitVec::from_words(words)
}

pub struct SpoRecord {
    pub subject: BitVec,
    pub predicate: BitVec,
    pub object: BitVec,
    pub packed: BitVec,       // subject.xor(&predicate).xor(&object)
    pub truth: TruthValue,
}

impl SpoRecord {
    pub fn new(s: &str, p: &str, o: &str, truth: TruthValue) -> Self {
        let subject = label_to_bitvec(s);
        let predicate = label_to_bitvec(p);
        let object = label_to_bitvec(o);
        let packed = subject.xor(&predicate).xor(&object);
        Self { subject, predicate, object, packed, truth }
    }
}
```

## STEP 4: SPO Store Uses BitVec Distance

```rust
impl SpoStore {
    pub fn query_forward(&self, s: &str, p: &str, radius: u32) -> Vec<SpoHit> {
        let s_fp = label_to_bitvec(s);
        let p_fp = label_to_bitvec(p);
        let query = s_fp.xor(&p_fp);  // BitVec XOR, not [u64;8] XOR
        
        self.nodes.values()
            .filter_map(|record| {
                let dist = record.packed.hamming_distance(&query);  // SIMD, automatic
                if dist <= radius {
                    Some(SpoHit {
                        target_key: record.key,
                        distance: dist,
                        truth: record.truth,
                    })
                } else {
                    None
                }
            })
            .collect()
    }
}
```

## STEP 5: Chain Traversal Uses BlasGraph Semiring Directly

```rust
use crate::graph::blasgraph::semiring::HammingMin;

impl SpoStore {
    pub fn walk_chain_forward(
        &self, start: u64, radius: u32, max_hops: usize
    ) -> Vec<TraversalHop> {
        // Uses the SAME HammingMin from blasgraph, not a reimplemented one.
        // One semiring. One type system. Store to algebra to result.
        let mut hops = Vec::new();
        let mut current = start;
        let mut cumulative = 0u32;

        for _ in 0..max_hops {
            let hits = self.query_forward_by_key(current, radius);
            if let Some(best) = hits.into_iter()
                .min_by_key(|h| h.distance)  // HammingMin: take the closest
            {
                cumulative = cumulative.saturating_add(best.distance); // tropical add
                hops.push(TraversalHop {
                    target_key: best.target_key,
                    distance: best.distance,
                    truth: best.truth,
                    cumulative_distance: cumulative,
                });
                current = best.target_key;
            } else {
                break;
            }
        }
        hops
    }
}
```

## STEP 6: Update Tests

Every test that used `[u64; 8]` or `Fingerprint` now uses `BitVec`.
The assertions don't change. The types do.

```rust
#[test]
fn spo_hydration_round_trip() {
    let mut store = SpoStore::new();
    store.insert(SpoRecord::new("Jan", "CREATES", "Ada", TruthValue::confident()));
    
    let hits = store.query_forward("Jan", "CREATES", 200);
    assert!(!hits.is_empty(), "Should find Jan CREATES → Ada");
    
    // The comparison happened at 16,384 bits.
    // With SIMD if available. The test doesn't know or care.
}
```

## STEP 7: Verify

```bash
cargo test --workspace
cargo clippy -- -D warnings
cargo fmt --check

# On a machine with AVX-512:
RUSTFLAGS="-C target-cpu=native" cargo bench --bench graph_execution
# On any machine:
cargo test  # scalar fallback, same results
```

## WHAT THE REDISGRAPH PERSON SEES

They open `src/graph/`:

```
graph/
  blasgraph/
    types.rs      — BitVec: 16,384-bit, SIMD-accelerated, cache-aligned
    semiring.rs   — 7 semirings (same idea as SuiteSparse, HD vectors not floats)
    matrix.rs     — mxm, mxv, vxm on sparse CSR
    vector.rs     — find_nearest, find_within
    ops.rs        — BFS, SSSP, PageRank as semiring operations
    sparse.rs     — COO, CSR storage
    descriptor.rs — operation control
  spo/
    store.rs      — triple store, queries USE BitVec and semirings from above
    truth.rs      — NARS confidence gating
    merkle.rs     — Blake3 integrity
    builder.rs    — convenience constructors
```

ONE type (BitVec) flows from store through semiring algebra to graph algorithms.
SIMD is invisible. It's just fast. The C64 spirit: the hardware does impossible
things and nobody knows how.

They think: "This is what RedisGraph should have been. On LanceDB. In Rust."

## FILES TO DELETE

```
crates/lance-graph/src/graph/spo/fingerprint.rs    DELETE entirely
crates/lance-graph/src/graph/spo/sparse.rs         DELETE (BitVec replaces Bitmap)
```

## FILES TO MODIFY

```
crates/lance-graph/src/graph/blasgraph/types.rs     ADD tiered SIMD dispatch
crates/lance-graph/src/graph/spo/mod.rs             USE BitVec, remove Fingerprint imports
crates/lance-graph/src/graph/spo/store.rs           USE BitVec for all operations
crates/lance-graph/src/graph/spo/builder.rs         USE BitVec, add label_to_bitvec()
crates/lance-graph/src/graph/spo/merkle.rs          USE BitVec for merkle hashing
crates/lance-graph/tests/spo_ground_truth.rs        USE BitVec in all tests
```

---

## ADDENDUM: COPY SIMD FROM RUSTYNUM (DO NOT HAND-ROLL)

### Clone rustynum for reference:

```bash
git clone https://github.com/AdaWorldAPI/rustynum.git ../rustynum
```

### Copy these exact files into BlasGraph BitVec:

**FROM `rustynum-core/src/simd.rs` (2302 lines) — copy ~200 lines:**

```
Lines 446-466:  hamming_distance() — the dispatch (AVX-512 → AVX2 → scalar)
Lines 471-487:  select_hamming_fn() — function pointer resolved ONCE
Lines 665-730:  hamming_vpopcntdq() — AVX-512 fast path
Lines 595-664:  hamming_avx2() — AVX2 Harley-Seal popcount
Lines 731-750:  hamming_scalar_popcnt() — scalar fallback
Lines 751-830:  popcount() + popcount_vpopcntdq() + popcount_avx2()
```

These operate on `&[u8]` slices. BitVec's `words: [u64; 256]` IS a `&[u8]` slice
via `unsafe { core::slice::from_raw_parts(words.as_ptr() as *const u8, 2048) }`.
The SIMD functions don't know or care about the type. They see bytes.

**FROM `rustynum-core/src/fingerprint.rs` (401 lines) — copy the struct pattern:**

```rust
// rustynum already has this. Copy the design:
pub struct Fingerprint<const N: usize> {
    pub words: [u64; N],
}
// Standard sizes: Fingerprint<256> = 16384 bits
```

Consider making BlasGraph's `BitVec` actually BE `Fingerprint<{HD_WORDS}>`.
Or copy the const-generic pattern. Either way, width is compile-time configurable.

### DO NOT hand-roll SIMD intrinsics. The rustynum implementations handle:
- Scalar tail (when buffer length isn't a multiple of register width)
- Horizontal sum (8 × i64 reduction after AVX-512 accumulation)
- Safety comments for every `unsafe` block
- Feature detection via `is_x86_feature_detected!` (runtime, not compile-time)
- AVX2 Harley-Seal lookup table (correct nibble LUT, not a naive popcount)
- Block processing in AVX2 to avoid u8 saturation

Hand-rolling these WILL have bugs. The rustynum versions have been tested
across 95 PRs and hundreds of CI runs. Copy them.

### The function pointer pattern:

```rust
// From rustynum simd.rs — resolve ONCE, call millions of times:
pub fn select_hamming_fn() -> fn(&[u8], &[u8]) -> u64 {
    if is_x86_feature_detected!("avx512vpopcntdq") { return hamming_vpopcntdq_safe; }
    if is_x86_feature_detected!("avx2") { return hamming_avx2_safe; }
    hamming_scalar_popcnt
}

// In BitVec, store the resolved function pointer:
use std::sync::OnceLock;
static HAMMING_FN: OnceLock<fn(&[u8], &[u8]) -> u64> = OnceLock::new();

impl BitVec {
    pub fn hamming_distance(&self, other: &BitVec) -> u32 {
        let f = HAMMING_FN.get_or_init(|| select_hamming_fn());
        let a = unsafe { core::slice::from_raw_parts(self.words.as_ptr() as *const u8, HD_BYTES) };
        let b = unsafe { core::slice::from_raw_parts(other.words.as_ptr() as *const u8, HD_BYTES) };
        f(a, b) as u32
    }
}
// CPUID check happens ONCE. Every subsequent call is a direct function pointer.
// Zero dispatch overhead after first call.
```

### Mapping: rustynum → lance-graph

```
rustynum function              → lance-graph BitVec method
────────────────────────────────────────────────────────
hamming_distance(&[u8],&[u8])  → BitVec::hamming_distance(&self, &BitVec)
popcount(&[u8])                → BitVec::popcount(&self)
select_hamming_fn()            → OnceLock<fn> in BitVec module
hamming_vpopcntdq              → private, called via fn pointer
hamming_avx2                   → private, called via fn pointer
hamming_scalar_popcnt          → private, called via fn pointer
Fingerprint<const N>           → BitVec (or rename BitVec to Fingerprint<HD_WORDS>)
```

### What NOT to copy from rustynum:

```
× mkl_ffi.rs          — MKL is an external dependency, upstream won't accept
× backends/xsmm.rs    — LIBXSMM is an external dependency
× backends/gemm.rs    — BF16 GEMM not needed for binary Hamming
× bf16_hamming.rs     — weighted BF16 Hamming not needed yet
× simd_compat.rs      — portable SIMD compat layer, overkill for this
× kernels.rs          — JIT scan kernels, ladybug-rs territory
× soaking.rs          — i8 accumulator, ladybug-rs territory
```

### What TO copy (total ~300 lines of proven, tested SIMD):

```
✓ hamming_distance dispatch    ~20 lines
✓ hamming_vpopcntdq            ~30 lines
✓ hamming_avx2 (Harley-Seal)   ~50 lines
✓ hamming_scalar_popcnt        ~20 lines
✓ popcount dispatch             ~20 lines
✓ popcount_vpopcntdq           ~25 lines
✓ popcount_avx2                ~50 lines
✓ popcount_scalar               ~10 lines
✓ select_hamming_fn             ~15 lines
✓ OnceLock dispatch wrapper     ~15 lines
✓ Fingerprint<const N> struct   ~50 lines (or adapt BitVec)
                          TOTAL ~305 lines
```

This is 300 lines of battle-tested code replacing 145 lines of FNV-1a toy.
The RedisGraph person sees 16K SIMD. They don't see rustynum. They see speed.
