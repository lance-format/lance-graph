# MedCare Policy Scaffolding Gap — finding 2026-05-06

> **Append-only finding.** Surfaces a workspace-shape mismatch
> discovered while attempting to mirror smb-office-rs PR #29
> (`SmbMembraneGate`) onto the medcare consumer side. The "30-LOC
> mirror" framing is wrong; medcare-rs lacks the prerequisite
> scaffolding crates that smb-office-rs already had.
>
> **READ BY:** sessions proposing `MedCareMembraneGate`, sessions
> closing POLICY-1 / MEMBRANE-GATE-1 on the medcare consumer side,
> sessions reviewing `foundry-consumer-parity-v1` consumer status.

---

## Finding

**smb-office-rs PR #29** added `SmbMembraneGate` in ~30 LOC because
two prerequisites already existed:

1. **`smb-realtime` crate** — host crate for the gate impl, with its
   own feature flags, tests, and existing membrane integration.
2. **`lance-graph-rbac` crate** — `Policy / Role / Operation /
   AccessDecision / smb_policy()` types, all upstream-owned. PR #29
   newtyped `Arc<lance_graph_rbac::Policy>` to bridge the orphan rule.

**medcare-rs has neither prerequisite.** Workspace inventory at
`AdaWorldAPI/medcare-rs` (commit `2816c2e0`):

| Crate | Status | Equivalent to |
|---|---|---|
| `medcare-core` | exists | smb-core |
| `medcare-db` | exists | smb-db |
| `medcare-analytics` | exists | smb-analytics |
| `medcare-pdf` | exists | (no smb analog) |
| `medcare-server` | exists | smb-server |
| **`medcare-realtime`** | **MISSING** | smb-realtime |
| **`medcare-rbac`** | **MISSING** | lance-graph-rbac (medcare side) |

The workspace `Cargo.toml` even acknowledges this in a comment:

> "the runtime membrane (Phoenix / PostgREST / RLS) lights up in F2
> once the upstream blockers (DM-7 + DM-8) land."

So the scaffolding gap is intentional and tracked. What's missing is
an explicit plan for closing it.

---

## What `MedCareMembraneGate` actually requires

To mirror PR #29's pattern on the medcare side, three pieces must
land in sequence:

### Stage 1 — `medcare-rbac` crate (~300 LOC + tests)

New workspace crate. Mirrors `lance-graph-rbac`'s shape with
medcare-domain entities (Patient / Diagnosis / LabResult /
Prescription / Anamnese / Ueberweisung):

- `Policy` (role × entity × operation matrix)
- `Role` (Doctor / Auditor / Receptionist / etc. — confirm with
  §73 SGB V regulatory scope)
- `Operation::{Read { depth }, Write { predicate }}`
- `AccessDecision::{Allow, Deny, Escalate}`
- `medcare_policy()` factory returning the canonical default
- 15-20 unit tests covering each role × entity decision

**Open question:** does medcare share `lance-graph-rbac`'s `Policy`
shape exactly, or does the §73 SGB V context (Überweisung-an-
Facharzt referral visibility) require a different structure? PR
smb-office-rs#97 mentioned by the prior session adds the regulatory
shape; that's the input to this scaffolding decision.

### Stage 2 — `medcare-realtime` crate skeleton (~200 LOC + tests)

New workspace crate. Mirrors `smb-realtime` shape:

- `Cargo.toml` with feature gates `auth-rls / realtime / full`
- `src/lib.rs` with module declarations
- `src/stack.rs` with `MedCareStack` (mirror of `SmbStack`) + a
  `domain_profile()` accessor for `StepDomain::MedCare.profile()`
- Boilerplate tests pinning the canonical `DomainProfile` defaults
  for medcare (audit_retention_days, auto_action_confidence,
  escalation, requires_fail_closed)

### Stage 3 — `MedCareMembraneGate` impl (~300 LOC + tests)

The gate itself. Mirror of PR #29's `SmbMembraneGate`:

- `src/gate.rs` in `medcare-realtime`
- `MedCareMembraneGate` newtype wrapping `Arc<medcare_rbac::Policy>` +
  `(role × entity_type)` binding
- `impl MembraneGate for MedCareMembraneGate` routing `gate_commit`
  to `Operation::Read` / `Operation::Write`
- Builders: `new`, `from_medcare_policy`, `with_write_predicate`,
  `with_read_depth`, `evaluate`
- 11-13 unit tests covering each role × entity × commit/read path
  (including the regulatory-relevant Überweisung-an-Facharzt
  referral visibility role-gating)

---

## Total scope

**~800 LOC across three new files in two new crates.** Far from a
30-LOC mirror. Each stage is independently shippable:

- Stage 1 (medcare-rbac): can land alone, no upstream deps beyond
  what's in the workspace already.
- Stage 2 (medcare-realtime skeleton): depends on Stage 1.
- Stage 3 (`MedCareMembraneGate`): depends on Stage 1 and Stage 2.

Realistic effort: ~2 person-days for Stages 1+2, ~1 person-day for
Stage 3. Compared to ½ day for the SMB mirror that didn't need
scaffolding.

---

## What this changes about the topology doc

`SINGLE_BINARY_TOPOLOGY.md` Layer 2 § Membrane currently says:

> POLICY-1 / MEMBRANE-GATE-1:
>   • SMB side: SHIPPED PR #29
>   • medcare side: PENDING (mirror as MedCareMembraneGate over
>     Arc<medcare_rbac::Policy>; ~30 LOC)

The "~30 LOC" estimate is incorrect. Updated estimate per this
finding: **~800 LOC across 3-stage scaffolding sequence**. The
gate itself is still ~300 LOC; the rest is workspace plumbing.

The topology doc itself stays correct — POLICY-1 medcare side is
PENDING, just with a bigger lift than first billed.

---

## Recommended path forward

**Option A — schedule the scaffolding work as a dedicated session.**
Allocate ~3 person-days. Land Stages 1+2+3 in three sequenced PRs.
Cleanest outcome; produces a real `MedCareMembraneGate` that closes
POLICY-1 medcare side.

**Option B — defer until DM-7 / DM-8 land in lance-graph-callcenter.**
The workspace comment already gestures at this:
"runtime membrane lights up in F2 once the upstream blockers
(DM-7 + DM-8) land". Once DM-7 (RlsRewriter) and DM-8 (PostgREST
handler) are real, the medcare-side scaffolding has clearer
consumers and the policy shape is anchored in regulatory reality.

**Option C — parallel-track Stage 1 only.** Land `medcare-rbac` now
(it's independent) so the regulatory policy shape gets pinned. Stage
2 + 3 wait for DM-7 / DM-8. Lowest-risk path.

The default in this finding doc is no-decision — surface the gap,
let the user pick when to schedule. Marker for sessions stumbling
on the same conflation: this file exists, read it before
re-discovering the gap.

---

## Cross-references

- **`SINGLE_BINARY_TOPOLOGY.md`** Layer 2 § Membrane — POLICY-1 line
  (estimate to be corrected when this finding is acted on)
- **`CROSS_REPO_PRS.md`** smb-office-rs#29 entry — the reference
  implementation
- **`foundry-consumer-parity-v1.md`** — consumer-side parity scope;
  medcare consumer is half-built per this finding
- **`callcenter-membrane-v1.md`** DM-7 / DM-8 — upstream blockers
  the workspace comment references
- **`q2-foundry-integration-v1.md`** — sister consumer that's
  further along (q2 PR #35 closed THINK-1 for q2 already)
