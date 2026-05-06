# Agent W9 — medcare-realtime/src/gate.rs (MedCareMembraneGate)

**Round:** 3 (Stage 3 — MedCareMembraneGate impl)
**Owner:** crates/medcare-realtime/src/gate.rs (~360 LOC, 13 tests)
**Commit:** medcare-rs `702e863`
**Status:** ✅ committed

## Action

Mirror of smb-office-rs#29 SmbMembraneGate adapted to medcare. Newtype
wrapping `Arc<medcare_rbac::Policy>` + (role × entity) binding to bridge
the orphan rule with upstream `MembraneGate` trait.

## Output verification

- ✅ Builders: new, with_write_predicate, with_read_depth, from_medcare_policy
- ✅ `evaluate(commit) -> AccessDecision` exposes Escalate distinctly
- ✅ `impl MembraneGate for MedCareMembraneGate` collapses to bool
- ✅ Three TD caveats from PR #29 documented in module head
- ✅ I-2 invariant: Send + Sync without async runtime (compile-time test)

## 13 tests

Doctor happy path (2) + Unknown role (2) + Auditor split (1) +
Receptionist depth gating (2 — CRITICAL #2 carry-forward) +
Anamnese append-only (3 — CRITICAL #1 carry-forward) +
Pinned predicate (1) + Arc share + AllowAllGate (2) +
Send+Sync compile (1).

## Self-review

- ✅ Mirrors PR #29 file-for-file with medcare type substitution
- ⚠️ BtM Escalate path not implemented in v1 (gate doesn't see row data)
- ⚠️ Action operations not reachable via gate — flagged for Meta-3 review
