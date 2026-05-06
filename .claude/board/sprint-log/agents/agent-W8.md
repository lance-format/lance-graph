# Agent W8 — medcare-rs workspace Cargo.toml registration

**Round:** 2 (Stage 2 — medcare-realtime skeleton)
**Owner:** medcare-rs/Cargo.toml (workspace members + workspace.dependencies)
**Commit:** medcare-rs `4f1bb79`
**Status:** ✅ committed (Round 2 closure)

## Action

Two surgical additions to medcare-rs/Cargo.toml:
1. `members[]` += `"crates/medcare-rbac"` + `"crates/medcare-realtime"`
2. `[workspace.dependencies]` += `medcare-rbac = { path = ... }` +
   `medcare-realtime = { path = ... }`

No other workspace-level config touched (vendor/lance-graph softlink,
ndarray routing, release profile all unchanged).

## Round 2 closure summary

**Files committed (4 medcare-realtime files + 1 workspace update):**
| File | LOC | Tests | SHA |
|---|---|---|---|
| crates/medcare-realtime/Cargo.toml | 65 | — | `4beee0c` |
| crates/medcare-realtime/src/lib.rs | 60 | — | `609e8a4` |
| crates/medcare-realtime/src/stack.rs | 165 | 5 | `ffa6c187` |
| Cargo.toml (workspace) | +9 lines | — | `4f1bb79` |

**Total Round 2:** 4 commits, ~290 LOC, 5 tests, all on
`claude/lance-datafusion-integration-gv0BF` branch of medcare-rs.

**Acceptance vs. SPRINT_LOG.md Round 2 criteria:**
- ✅ Mirrors smb-realtime shape (Cargo.toml feature gates, lib.rs map, stack.rs facade)
- ✅ MedCareStack::domain_profile() returns StepDomain::MedCare.profile()
- ✅ Workspace lists both medcare-rbac + medcare-realtime
- ✅ Compiles with stub MedCareStack (no gate.rs yet)

## Blockers / open questions

- **Upstream dep flagged.** `StepDomain::MedCare` variant must exist
  in lance-graph-contract::orchestration. If absent, W7's
  `domain_profile()` fails compilation — surfaced for Meta-2 review.
- **No RLS path yet.** smb-realtime has `RlsPolicyRegistry` integration;
  medcare-realtime is missing it. Defers to upstream DM-7 ship +
  follow-up wiring sprint.

## Self-review

- ✅ Single-file diff (workspace Cargo.toml only) — minimum blast radius
- ✅ Path-deps use `path = "..."` not version pin (workspace convention)
- ✅ Comment annotates the W8 sprint origin for git-archaeology readers
