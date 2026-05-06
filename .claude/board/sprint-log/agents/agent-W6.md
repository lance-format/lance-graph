# Agent W6 — medcare-realtime/src/lib.rs

**Round:** 2 (Stage 2 — medcare-realtime skeleton)
**Owner:** crates/medcare-realtime/src/lib.rs (initial; gate exports deferred to W10)
**Commit:** medcare-rs `609e8a4`
**Status:** ✅ committed

## Action

Module map mirroring smb-realtime/src/lib.rs:
- `pub mod stack` — exposed as v1 surface
- `gate` module declaration deferred to W10 (Round 3) per per-round file ownership

Crate-level lints `#![warn(missing_docs)]` + `#![forbid(unsafe_code)]`
match smb-realtime convention.

## Self-review

- ✅ "What this crate is NOT" anti-pattern list (not a query engine,
  not a new contract layer, not a separate process)
- ✅ Round 3 gate module declaration commented in (W10 flips it on)
- ✅ Layer-2 placement note per SINGLE_BINARY_TOPOLOGY.md
- ⚠️ Gate re-export shape will need adjustment in W10: smb-realtime
  re-exports `AccessDecision, AllowAllGate, MembraneGate, Policy,
  SmbMembraneGate`. medcare-realtime should re-export the same set
  with `MedCareMembraneGate` substituting for `SmbMembraneGate`.
