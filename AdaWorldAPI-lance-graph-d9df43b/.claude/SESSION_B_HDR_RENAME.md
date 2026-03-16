# SESSION_B_HDR_RENAME.md

## Rename LightMeter → hdr::Cascade (lance-graph only)

**Repo:** lance-graph (WRITE)
**Scope:** one file rename, one type rename, one module rename. Nothing else.
**Stop when:** `cargo test --workspace` passes.

---

## STEP 1: Rename file

```bash
cd crates/lance-graph/src/graph/blasgraph
git mv light_meter.rs hdr.rs
```

## STEP 2: Update mod.rs

In `crates/lance-graph/src/graph/blasgraph/mod.rs`:

```rust
// BEFORE:
pub mod light_meter;

// AFTER:
pub mod hdr;
```

## STEP 3: Rename type inside hdr.rs

In the renamed `hdr.rs`, find-replace:

```
LightMeter  →  Cascade
```

Keep ALL internals unchanged. Just the struct name.

## STEP 4: Rename methods inside hdr.rs

```
cascade_query()  →  query()
```

Add these thin wrappers if they don't exist:

```rust
impl Cascade {
    /// Classify a single distance into a sigma band.
    #[inline]
    pub fn expose(&self, distance: u32) -> Band {
        self.band(distance)  // band() already exists, expose is the public name
    }

    /// Single pair test: is this distance in a useful band?
    #[inline]
    pub fn test_distance(&self, distance: u32) -> bool {
        self.band(distance) <= Band::Good
    }
}
```

## STEP 5: Update internal references

Search for any file in lance-graph that imports `light_meter` or `LightMeter`:

```bash
grep -rn "light_meter\|LightMeter" crates/ --include="*.rs"
```

Update each to use `hdr` and `Cascade`.

## STEP 6: Update tests

In `crates/lance-graph/tests/hdr_proof.rs` (and any other test files):

```
use lance_graph::graph::blasgraph::light_meter::LightMeter
→
use lance_graph::graph::blasgraph::hdr::Cascade
```

Replace `LightMeter::` with `Cascade::` in test bodies.

## STEP 7: Verify

```bash
cargo test --workspace
cargo clippy --workspace -- -D warnings
```

---

## NOT IN SCOPE

```
× Don't add ReservoirSample improvements (already merged in PR #9)
× Don't add PreciseMode (Session C cross-pollinates this)
× Don't add incremental Stroke 2 (Session C)
× Don't touch SIMD dispatch (comes with BitVec rebuild)
× Don't touch rustynum (Session A does that)
× Don't rename Band, RankedHit, ShiftAlert (names already match target)
```
