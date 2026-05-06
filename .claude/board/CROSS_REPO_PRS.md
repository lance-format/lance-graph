# Cross-Repo PR Cross-References

> **Append-only log of merged PRs in OTHER AdaWorldAPI repos** that
> touch the lance-graph topology — i.e. consume canonical contract
> types, validate Single-Binary-Topology invariants, or close
> entropy-ledger rows on the consumer side.
>
> `PR_ARC_INVENTORY.md` is scoped to lance-graph PRs (Added / Locked
> / Deferred semantics rooted in this repo). Cross-repo PRs need
> their own trail because they validate or extend lance-graph design
> from outside; they don't add types to lance-graph itself.
>
> ## APPEND-ONLY rule
>
> 1. New entries PREPEND at the top (most recent first).
> 2. Each entry is IMMUTABLE except the **Confidence** line.
> 3. Entries reference lance-graph entropy-ledger rows and topology
>    layers explicitly.
> 4. PRs in repos OUTSIDE the Claude Code MCP allowlist get a
>    **(MCP scope: out-of-scope; diff not fetched)** marker; entry
>    captures what's known from prior context. Update when access
>    or paste arrives.
>
> **READ BY:** sessions auditing topology validation, sessions
> proposing new consumer-side work, sessions tracking the MEDCARE-
> PARITY-1 / MEDCARE-* / SMB-* / Q2-* entropy-ledger rows that close
> on the consumer side.

---

## MedCareV2 #7 — merged 2026-05-06

**Repo:** `AdaWorldAPI/MedCareV2` (C# .NET Framework 4.8 desktop)
**MCP scope:** out-of-scope; diff not fetched.
**Topology layer:** L3 caller-side (separate process from medcare-rs Rust binary)

**Topology placement.** MedCareV2 LanceProbe is a Windows .NET 4.8
desktop application running OUTSIDE the medcare-rs binary. It calls
`/api/__parity/csharp` over HTTP, which is an L3 serving endpoint
in the Rust binary at `medcare-rs/routes/parity.rs:46`. From the
Rust topology's perspective:

- The C# probe is on the OUTBOUND side of the tokio boundary —
  same side as MySQL sink-in and other network egress.
- The Rust-side parity ingest at `routes/parity.rs:46` is an L3
  serving endpoint (M5-class outbound POST).
- Nothing inside the Rust binary's L1/L2 is async on the C# probe's
  behalf — the probe is a client, the Rust binary serves it.

**Entropy-ledger row anchor.** MEDCARE-PARITY-1 (the parity ring
between C# probe and Rust binary). Currently a hypothetical row in
the ledger; this PR is the consumer-side advancement that will
eventually let the row carry a SHIPPED/PR cite once both sides
are wired.

**What's known from prior context** (no diff access):
- PR sequence is #4 → #5 → R2-R6 follow-ups → #7. PR #7 is one
  follow-up on the LanceProbe arc.
- The probe drives the parity-clean window discussion in
  `foundry-consumer-parity-v1` — when C# probe diffs match the
  Rust binary's Lance reads ≡ MySQL oracle reads over a §10 BMV-Ä
  audit period, the parity-clean window opens.
- Per the parity-ring narrative: probe currently produces real
  diffs (or 401s/404s on auth/route gaps) rather than 200-OK
  bypasses.

**Confidence (2026-05-06):** Cannot verify — out of MCP scope.
Citation here is for traceability; promote to FINDING when the
diff is paste-shared or the allowlist is extended.

**Cross-refs:**
- `SINGLE_BINARY_TOPOLOGY.md` Layer 3 § "External probes" entry
- `foundry-consumer-parity-v1.md` (parity-clean window discussion)
- `medcare-rs/routes/parity.rs:46` (Rust-side ingest endpoint)

---

## q2 #35 — merged 2026-05-06

**Repo:** `AdaWorldAPI/q2`
**MCP scope:** in-scope; diff fetched.
**Topology layer:** L1 driver migration + L3 SSE serving update

**Title:** Phase 2B: canonical R1 surface + MockShaderDriver +
planner NARS deduction

**Topology placement.** Q2 cockpit-server is the Palantir-Gotham-
equivalent consumer per `q2-foundry-integration-v1.md`. This PR
migrated cockpit-server to the canonical L1 contract surface:

- **L1 driver abstraction adopted.** `MockShaderDriver` now
  implements the canonical `CognitiveShaderDriver` trait. SSE
  handler calls `driver.dispatch_with_sink(&dispatch, &mut SseSink)`.
- **L1 canonical DTOs adopted.** Dropped `thinking-engine` and
  `cognitive-shader-driver` deps; consumes
  `lance_graph_contract::cognitive_shader::{ShaderDispatch,
  ShaderResonance, ShaderBus, ShaderCrystal}` directly.
- **L1 canonical NARS algebra adopted.** Hand-rolled `f=f1*f2,
  c=f1*f2*c1*c2` (q2 was the 4th copy) replaced with bridge to
  `lance_graph_planner::nars::truth::TruthValue::deduction`.
- **L3 SSE wire-shape compaction.** `cycle_fingerprint: [u64;256]`
  (2KB inner) → `cycle_fingerprint_hash: u64` (8B XOR-fold).
  `color_acc: [f32;32]` (128B) → `color_acc_active_dims: u8`
  (1B). Concrete I-3 BBB enforcement evidence at the L1→L3
  projection.
- **L2 per-row gate exposed in L3 wire.** New SSE field:
  `gate: { gate: u8, merge: 'Xor'|'Bundle'|'Superposition'|
  'AlphaFrontToBack' }`. The L2 `collapse_gate::GateDecision`
  propagates through to the L3 SSE stream.

**Entropy-ledger rows resolved:**
- **THINK-1** — q2 migrated from `thinking_engine::dto::*` to
  canonical `cognitive_shader::*`. Consumer-side closure for q2.
- **TRUTH-1** — reduced from 4 copies to 3 (q2 dropped its
  hand-rolled copy). Closes the q2-specific instance.
- **MOCK-DRIVER-IS-CONTRACT-CITIZEN** — stub driver implements the
  canonical trait; not a parallel API.

**Where CycleAccumulator becomes load-bearing.**
Phase 2B uses MockShaderDriver at low rate; SSE cadence
`cycle_ms=300` is tractable. Phase 3 replaces with `BgzShaderDriver`
at real cognitive-cycle speed (20-200 ns/op). At 100 ns/cycle, a
300 ms window produces ~3M cycles — SSE/HTTP/browser cannot
absorb that. CycleAccumulator (per topology I-4) sits between the
real driver and SseSink at exactly this boundary.

**Confidence (2026-05-06):** Working. 12/12 unit tests pass
(7 dto_bridge + 5 mock_driver). cargo check clean,
tsc --noEmit clean, npm build clean.

**Cross-refs:**
- `SINGLE_BINARY_TOPOLOGY.md` (validates I-1 / I-3 / I-4)
- `q2-foundry-integration-v1.md` (Gotham-parity scope)
- `unified-integration-v1.md` (DU-4 `rationale_phase` is the
  next q2-side adoption candidate)

---

## smb-office-rs #29 — merged 2026-05-06

**Repo:** `AdaWorldAPI/smb-office-rs`
**MCP scope:** in-scope; diff fetched.
**Topology layer:** L2 membrane (RBAC gate at the inner→outer
projection boundary)

**Title:** feat(smb-realtime): SmbMembraneGate + domain_profile —
close Foundry-seal POLICY-1 seam

**Topology placement.** SmbMembraneGate is in-process, sync, gating
zero-copy CognitiveShader→callcenter handshakes for the SMB
consumer crate. Per topology I-1, the consumer is compiled into
the same binary as lance-graph; the gate decision (20-200 ns) runs
at L1 inner speed.

**Architectural call resolved.** Orphan rule blocked
`impl MembraneGate for rbac::Policy` (both upstream-owned).
Newtype `SmbMembraneGate` wraps `Arc<lance_graph_rbac::Policy>` +
`(role × entity_type)` binding; impls `MembraneGate::should_emit`
by routing `gate_commit` to `Operation::Read` / `Operation::Write`.

**Entropy-ledger rows resolved:**
- **POLICY-1 / MEMBRANE-GATE-1** — SMB-side closure. The medcare
  side is still PENDING (~30-LOC mirror as `MedCareMembraneGate`
  over `Arc<medcare_rbac::Policy>`).

**Three open caveats** (carry into TD as needed):
- TD-MEMBRANE-FACULTY-BLIND — `should_emit` ignores
  `external_role / faculty_role / expert_id`. Faculty-aware policy
  is a future concern; trait shape will need rework.
- TD-MEMBRANE-ESCALATE-LOSSY — `Escalate` collapses to `false` in
  `should_emit`. Lossy. Consider `emit_decision()` extension.
- TD-MEMBRANE-FIRST-VS-ANY — default-commit picks
  `writable_predicates.first()`; if predicate-specific RLS exists,
  this denies when "any" should allow. Verify or document.

**Confidence (2026-05-06):** Working. 13 new unit tests passing
(11 gate + 2 domain_profile). clippy --all-targets --no-deps
-D warnings: clean. smb-realtime --features full: 46 passing
(was 33).

**Cross-refs:**
- `SINGLE_BINARY_TOPOLOGY.md` Layer 2 § "Membrane (transcode +
  RBAC)" entry
- `ARCHITECTURE_ENTROPY_LEDGER.md` POLICY-1 + Section B Foundry-
  seal cluster
- `soa-dto-fma-map.md` open-seam #3 ("R4 ↔ R6 MembraneGate bridge
  missing")
- `external_membrane.rs:7-13` (BBB invariant the gate enforces)

---

## How to use this file

1. **When reviewing a cross-repo PR** that touches lance-graph
   types: check this file first. If listed, read the entry to
   understand topology placement before diving into the diff.
2. **When a new cross-repo PR merges:** prepend an entry. Use the
   format above (Repo / MCP scope / Layer / placement / ledger
   rows / confidence / cross-refs).
3. **When MCP scope changes:** if a previously out-of-scope repo
   becomes accessible, update the relevant entry's MCP-scope line
   and promote Confidence after fetching the diff.
4. **When closing an entropy-ledger row** via a cross-repo PR:
   update the ledger row's status AND cite the entry here. Both
   sides of the cross-reference matter.
