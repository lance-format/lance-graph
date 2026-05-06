# Agent W2 — medcare-rbac src/lib.rs + src/access.rs

**Round:** 1 (Stage 1 — medcare-rbac scaffolding)
**Owner:** crates/medcare-rbac/src/lib.rs + src/access.rs
**Commits:** medcare-rs `7b91459` (lib.rs) + `2fdace7` (access.rs)
**Status:** ✅ committed

## Action

### File 1 — src/lib.rs (1641 bytes)

Module map mirroring lance-graph-rbac with medcare layering doc:
- `pub mod access` / `permission` / `policy` / `role` (4 declarations)
- Layer-2 placement note per SINGLE_BINARY_TOPOLOGY.md
- §73 SGB V row-level visibility note: cross-doctor referral visibility
  is gate/RLS concern, not RBAC role decision
- Consumer pointers: medcare-realtime::gate (Round 3) + medcare-server
  RLS (after upstream DM-7)

### File 2 — src/access.rs (2457 bytes)

`AccessDecision` enum (Allow / Deny / Escalate) + 3 predicates
(is_allowed / is_denied / is_escalation). Identical to upstream PLUS
one medcare-specific test:

```rust
fn btm_escalation_is_distinct_from_deny() {
    let btm = AccessDecision::Escalate {
        reason: "BtM second signature required",
    };
    assert!(btm.is_escalation());
    assert!(!btm.is_denied());
}
```

Captures the BtM (controlled-substance) dual-control pattern: prescriptions
needing second signature return `Escalate`, distinct from outright `Deny`.

## Output verification

- ✅ AccessDecision enum: 3 variants, &'static str reasons (matches upstream)
- ✅ const fn predicates (matches upstream — usable in const eval)
- ✅ 3 stock tests + 1 medcare-specific BtM test = 4 tests total
- ✅ Doc comment documents the §73 SGB V boundary

## Blockers / open questions

- **TD-MEMBRANE-ESCALATE-LOSSY cross-ref**: the gate's `should_emit()`
  collapses Escalate to false; this is acknowledged in access.rs doc
  but not yet wired through to medcare-realtime/gate.rs (Round 3).
  W9 needs to address.

## Self-review

- ✅ access.rs: PR #29 caveat (Escalate-lossy) referenced in doc
- ⚠️ lib.rs: no `pub use` re-exports at top level — matches upstream
  shape but consumer code will use deeper paths. Defer to Meta-1.
- ⚠️ BtM test is a smoke test — no actual gate integration yet.
  *Real BtM-flagged-prescription test belongs in Round 3 (W12).*
