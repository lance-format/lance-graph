# Agent W5 — medcare-realtime/Cargo.toml

**Round:** 2 (Stage 2 — medcare-realtime skeleton)
**Owner:** crates/medcare-realtime/Cargo.toml
**Commit:** medcare-rs `4beee0c`
**Status:** ✅ committed

## Action

Wrote `crates/medcare-realtime/Cargo.toml` mirroring smb-realtime feature
gate structure. Minimum viable v1: 3 features (auth-rls, postgrest,
full) + 5 deps (lance-graph-contract, lance-graph-callcenter, medcare-rbac,
thiserror, tracing).

## Deferred from v1 (vs. smb-realtime)

| Dep | Why deferred |
|---|---|
| `arrow` | Only needed when typed-Arrow ingest paths land. v1 gate doesn't need it. |
| `tokio` | Only needed when DM-5 PhoenixServer lands (L3 outbound work). |
| `futures` / `async-trait` | No async surface in v1; gate is sync. |
| `smb-bridge` equivalents | No medcare-bridge exists; not in scope for this sprint. |

## Self-review

- ✅ Feature flags forward to upstream (auth-rls / postgrest / full)
- ✅ medcare-rbac dep wired (W8 must register the workspace dep)
- ✅ Per topology I-2: no tokio in this dep set
- ⚠️ medcare_ontology() not yet referenced; will need it when DM-8 lands
