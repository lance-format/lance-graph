Project Structure Proposal (Issue #92)
======================================

This document captures a pragmatic "proper" structure for lance-graph based on
issue #92. It focuses on clearer separation of concerns, a Cargo workspace, and
first-class Python bindings.

Goals
-----
- Keep lightweight crates usable without pulling in heavy dependencies.
- Allow custom graph-native operators without forcing everything through SQL.
- Make Python bindings and examples first-class citizens.

Current Layout (after refactor)
-------------------------------
- `crates/lance-graph/` hosts the Rust engine crate.
- `python/` contains PyO3 bindings plus the `lance_graph` and
  `knowledge_graph` Python packages.
- `examples/` holds simple Cypher usage examples.

Proposed Layout (Workspace)
---------------------------
The issue recommends moving to a Cargo workspace to split functionality and
avoid pulling DataFusion/Tokio into every consumer:

```
lance-graph/
├── Cargo.toml              # Workspace definition
├── README.md               # Unified docs
├── docs/                   # ADRs + design notes
├── crates/
│   ├── lance-graph/         # Facade crate (re-exports)
│   ├── lance-graph-core/    # AST + schema types (no heavy deps)
│   └── lance-graph-planner/ # Cypher -> LogicalPlan
├── python/                  # PyO3 + Python packages
└── examples/                # End-to-end examples
```

Why this helps
--------------
- `lance-graph-core` can be reused by tooling and schema validators without
  DataFusion or Tokio.
- `lance-graph-planner` keeps translation logic isolated for easier evolution.
- A future `lance-graph-physical` crate can be introduced once we have
  substantial graph-native kernels (traversal, vector search).
- The facade crate offers a stable public API while allowing internal churn.

Suggested Migration Steps
-------------------------
1. Add a top-level `Cargo.toml` workspace and move the existing Rust crate into
   `crates/lance-graph/` without changing its internal module layout.
2. Extract AST + schema types into `crates/lance-graph-core/` and update
   references.
3. Split the planner (`Cypher -> LogicalPlan`) into `lance-graph-planner`.
4. Refactor the Simple executor so it lives in the workspace cleanly (either
   inside the facade crate or as a `lance-graph-simple` crate).
5. Update `python/Cargo.toml` to depend on the workspace crates.
6. Add ADRs for each step under `docs/` to document tradeoffs.
7. Update GitHub workflows to match the new workspace paths (build, test,
   release/publish), otherwise releases will fail.
