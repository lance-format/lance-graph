# SESSION_LANGGRAPH_ORCHESTRATION.md

## Mission: Replace Layer 4 Chaos with LangGraph-Style Graph Orchestration Wired into Zero-Copy Planes

**Problem:** Layer 4 of ladybug-rs's 10-layer thinking stack currently uses a messy
mix of crewai-rust patterns, n8n-rs workflow fragments, and jitson JIT compilation.
No coherent execution model. No session state. No conditional routing. No human-in-loop.

**Solution:** Adopt the LangGraph execution model (as implemented in `rs-graph-llm/graph-flow`)
and wire it directly into our binary plane architecture. Tasks operate on Planes
through the Blackboard zero-copy arena. The Context IS the Blackboard.
The execution graph IS the thinking graph. Sessions persist as Lance versions.

**Key insight:** graph-flow gives us exactly what we need:
- Graph execution engine (stateful task orchestration)
- Conditional routing (branch based on Context data)
- Session management (persist, resume, time-travel)
- FanOut (parallel task execution with result aggregation)
- Human-in-the-loop (WaitForInput)
- Step-by-step OR continuous execution (NextAction enum)
- Type-safe Context with get/set (thread-safe)

We don't need to build ANY of this. We need to WIRE it to our binary substrate.

---

## REFERENCE: rs-graph-llm/graph-flow Architecture

```
REPO: https://github.com/a-agmon/rs-graph-llm
CRATE: graph-flow/ (the core library)
LICENSE: MIT (permissive, compatible with our stack)

CORE TYPES:
  Task (trait)        → async fn run(&self, context: Context) -> Result<TaskResult>
  Context             → thread-safe key-value store (Arc<RwLock<HashMap>>)
  GraphBuilder        → add_task, add_edge, add_conditional_edge, build
  FlowRunner          → run(session_id) → ExecutionResult
  Session             → session_id + context + current_task + history
  SessionStorage      → InMemorySessionStorage, PostgresSessionStorage
  FanOutTask          → parallel child tasks, result aggregation

EXECUTION MODEL:
  NextAction::Continue              → step-by-step (return to caller)
  NextAction::ContinueAndExecute    → auto-continue to next task
  NextAction::WaitForInput          → pause for human/external input
  NextAction::End                   → workflow complete
  NextAction::GoTo(task_id)         → jump to specific task
  NextAction::GoBack                → return to previous task

CONDITIONAL ROUTING:
  .add_conditional_edge(from, predicate_fn, true_target, false_target)
  Predicate reads from Context → routes to different tasks
```

---

## STEP 1: Map graph-flow to Our 10-Layer Thinking Stack

```
LADYBUG-RS 10 LAYERS (current):
  Layer 1:  Sensory input (MCP ingest, text, events)
  Layer 2:  Tokenization / fingerprinting (text → Plane encounter)
  Layer 3:  Pattern recognition (cascade search, hamming classification)
  Layer 4:  ORCHESTRATION (currently messy: crewai + n8n + jitson)  ← FIX THIS
  Layer 5:  Reasoning (semiring graph traversal, Bellman-Ford)
  Layer 6:  Memory consolidation (encounter, seal, Staunen detection)
  Layer 7:  Planning (multi-step goal decomposition)
  Layer 8:  Action selection (RL credit assignment, BF16 truth)
  Layer 9:  Output generation (LLM, template, structured response)
  Layer 10: Meta-cognition (self-monitoring, confidence calibration)

LAYER 4 WITH GRAPH-FLOW:
  Each layer becomes a TASK in the execution graph.
  The graph defines which layers connect and under what conditions.
  The Context IS the Blackboard (zero-copy shared state).
  Sessions persist as Lance versions (time travel for free).
```

### The Thinking Graph

```rust
// Layer 4: the orchestration graph that WIRES all other layers
let thinking_graph = GraphBuilder::new("ladybug_thinking")
    // Layer 1: Sensory
    .add_task(Arc::new(SensoryIngestTask))        // MCP → raw input
    
    // Layer 2: Fingerprint
    .add_task(Arc::new(FingerprintTask))           // text → Plane.encounter()
    
    // Layer 3: Pattern Recognition
    .add_task(Arc::new(CascadeSearchTask))         // cascade → band classification
    
    // Layer 5: Reasoning (conditional: skip if pattern already known)
    .add_task(Arc::new(SemiringReasonTask))         // graph traversal
    
    // Layer 6: Memory
    .add_task(Arc::new(MemoryConsolidateTask))      // encounter + seal check
    
    // Layer 7: Planning (conditional: only for multi-step goals)
    .add_task(Arc::new(PlanningTask))               // goal decomposition
    
    // Layer 8: Action Selection
    .add_task(Arc::new(ActionSelectTask))            // RL credit assignment
    
    // Layer 9: Output
    .add_task(Arc::new(OutputGenerateTask))          // LLM or template
    
    // Layer 10: Meta-cognition
    .add_task(Arc::new(MetaCognitionTask))           // self-monitoring
    
    // EDGES: the thinking flow
    .add_edge(sensory.id(), fingerprint.id())
    .add_edge(fingerprint.id(), cascade.id())
    
    // CONDITIONAL: if cascade finds Foveal match, skip reasoning
    .add_conditional_edge(
        cascade.id(),
        |ctx| ctx.get_sync::<String>("best_band")
                 .map(|b| b == "Foveal")
                 .unwrap_or(false),
        memory.id(),        // Foveal → skip reasoning, go to memory
        reasoning.id(),     // Not Foveal → need reasoning
    )
    
    .add_edge(reasoning.id(), memory.id())
    
    // CONDITIONAL: if Staunen detected, go to planning
    .add_conditional_edge(
        memory.id(),
        |ctx| ctx.get_sync::<bool>("staunen_detected")
                 .unwrap_or(false),
        planning.id(),      // Staunen → need new plan
        action.id(),        // Wisdom → proceed to action
    )
    
    .add_edge(planning.id(), action.id())
    .add_edge(action.id(), output.id())
    .add_edge(output.id(), meta.id())
    
    // META-COGNITION can loop back
    .add_conditional_edge(
        meta.id(),
        |ctx| ctx.get_sync::<f32>("confidence")
                 .map(|c| c < 0.5)
                 .unwrap_or(false),
        cascade.id(),       // Low confidence → re-search
        // high confidence → end
        "END",
    )
    
    .build();
```

---

## STEP 2: Context = Blackboard = Zero-Copy Plane Access

The critical wiring: graph-flow's Context is a `HashMap<String, Value>`.
Our Blackboard is a zero-copy shared memory arena with SIMD-aligned allocations.
We need to bridge them so Tasks read/write Planes through Context.

```rust
/// Bridge: graph-flow Context backed by rustynum Blackboard
/// Tasks get/set Planes through the Context API.
/// Zero-copy: Planes are NOT serialized into the Context HashMap.
/// They live in the Blackboard arena. Context holds references.
struct PlaneContext {
    // graph-flow's standard Context for scalar values
    inner: Context,
    // rustynum's Blackboard for zero-copy Plane access
    blackboard: Arc<Blackboard>,
}

impl PlaneContext {
    /// Get a Plane from the Blackboard by name. Zero-copy.
    fn get_plane(&self, name: &str) -> Option<&Plane> {
        self.blackboard.get_plane(name)
    }
    
    /// Get a mutable Plane for encounter(). Zero-copy.
    fn get_plane_mut(&self, name: &str) -> Option<&mut Plane> {
        self.blackboard.get_plane_mut(name)
    }
    
    /// Allocate a new Plane in the Blackboard.
    fn alloc_plane(&self, name: &str) -> &mut Plane {
        self.blackboard.alloc_plane(name)
    }
    
    /// Get a Node (3 Planes) from the Blackboard.
    fn get_node(&self, name: &str) -> Option<&Node> {
        self.blackboard.get_node(name)
    }
    
    /// Store scalar values (distances, bands, BF16 truths) in Context.
    /// These ARE serialized (small values, not Planes).
    async fn set_scalar<T: Serialize>(&self, key: &str, value: T) {
        self.inner.set(key, value).await;
    }
    
    /// Read scalar values from Context.
    fn get_scalar<T: DeserializeOwned>(&self, key: &str) -> Option<T> {
        self.inner.get_sync(key)
    }
}
```

### Example: CascadeSearchTask

```rust
struct CascadeSearchTask {
    cascade: Arc<Cascade>,
}

#[async_trait]
impl Task for CascadeSearchTask {
    fn id(&self) -> &str { "cascade_search" }
    
    async fn run(&self, context: Context) -> graph_flow::Result<TaskResult> {
        let pctx = PlaneContext::from(context.clone());
        
        // Read query Plane from Blackboard (zero-copy)
        let query_plane = pctx.get_plane("query_s")
            .ok_or("No query plane in blackboard")?;
        
        // Run cascade search (SIMD, hot path)
        let hits = self.cascade.query(
            query_plane.bits_bytes_ref(),
            &database,
            10,
            threshold,
        );
        
        // Store results as scalar in Context
        pctx.set_scalar("cascade_hits", &hits).await;
        pctx.set_scalar("best_band", hits[0].band.to_string()).await;
        pctx.set_scalar("best_distance", hits[0].distance).await;
        
        // If Foveal match found, we can skip reasoning
        let response = format!("Found {} hits, best: {:?}", hits.len(), hits[0].band);
        Ok(TaskResult::new(Some(response), NextAction::Continue))
    }
}
```

### Example: MemoryConsolidateTask

```rust
struct MemoryConsolidateTask;

#[async_trait]
impl Task for MemoryConsolidateTask {
    fn id(&self) -> &str { "memory_consolidate" }
    
    async fn run(&self, context: Context) -> graph_flow::Result<TaskResult> {
        let pctx = PlaneContext::from(context.clone());
        
        // Get the matched node from Blackboard (zero-copy)
        let matched_node = pctx.get_node_mut("matched_node")
            .ok_or("No matched node")?;
        let evidence_node = pctx.get_node("evidence_node")
            .ok_or("No evidence node")?;
        
        // Store current seal before encounter
        let seal_before = matched_node.s.merkle();
        
        // Encounter: the learning step (integer, deterministic)
        matched_node.s.encounter_toward(&evidence_node.s);
        matched_node.p.encounter_toward(&evidence_node.p);
        matched_node.o.encounter_toward(&evidence_node.o);
        
        // Check seal: Wisdom or Staunen?
        let seal_after = matched_node.s.verify(&seal_before);
        let staunen = seal_after == Seal::Staunen;
        
        pctx.set_scalar("staunen_detected", staunen).await;
        pctx.set_scalar("seal_status", format!("{:?}", seal_after)).await;
        
        let response = if staunen {
            "Staunen: something changed unexpectedly. Rethinking.".to_string()
        } else {
            "Wisdom: knowledge consolidated.".to_string()
        };
        
        Ok(TaskResult::new(Some(response), NextAction::Continue))
    }
}
```

---

## STEP 3: FanOut for 2³ SPO Projections

The 7 SPO projections can run in PARALLEL using graph-flow's FanOut:

```rust
// Each projection is a child task
struct ProjectionTask { mask: Mask }

#[async_trait]
impl Task for ProjectionTask {
    fn id(&self) -> &str { /* mask-specific id */ }
    
    async fn run(&self, context: Context) -> graph_flow::Result<TaskResult> {
        let pctx = PlaneContext::from(context.clone());
        let query = pctx.get_node("query").unwrap();
        let candidate = pctx.get_node("candidate").unwrap();
        
        let distance = query.distance(candidate, self.mask);
        let band = cascade.expose(distance.raw().unwrap_or(u32::MAX));
        
        pctx.set_scalar(&format!("proj_{:?}_dist", self.mask), distance).await;
        pctx.set_scalar(&format!("proj_{:?}_band", self.mask), band).await;
        
        Ok(TaskResult::new(None, NextAction::End))
    }
}

// FanOut: all 7 projections in parallel
let projection_fanout = FanOutTask::new("spo_projections", vec![
    Arc::new(ProjectionTask { mask: S__ }),
    Arc::new(ProjectionTask { mask: _P_ }),
    Arc::new(ProjectionTask { mask: __O }),
    Arc::new(ProjectionTask { mask: SP_ }),
    Arc::new(ProjectionTask { mask: S_O }),
    Arc::new(ProjectionTask { mask: _PO }),
    Arc::new(ProjectionTask { mask: SPO }),
]).with_prefix("proj");

// After FanOut, the BF16 assembly task reads all 7 results
struct BF16AssemblyTask;

#[async_trait]
impl Task for BF16AssemblyTask {
    async fn run(&self, context: Context) -> graph_flow::Result<TaskResult> {
        // Read all 7 projection results from FanOut
        let bands: [Band; 7] = [
            context.get_sync("proj.s__.band").unwrap(),
            context.get_sync("proj._p_.band").unwrap(),
            // ...
        ];
        
        let bf16 = bf16_from_projections(&bands, finest, foveal_max, direction);
        context.set("bf16_truth", bf16).await;
        
        Ok(TaskResult::new(None, NextAction::Continue))
    }
}
```

---

## STEP 4: Session = Lance Version

graph-flow has SessionStorage (InMemory, Postgres). We add LanceSessionStorage:

```rust
/// Session storage backed by Lance format.
/// Each save() creates a new Lance version.
/// Time travel: load(session_id, version=N) restores state at version N.
/// diff(v1, v2) shows what changed between thinking steps.
struct LanceSessionStorage {
    dataset_path: String,  // "s3://bucket/sessions.lance" or local path
}

#[async_trait]
impl SessionStorage for LanceSessionStorage {
    async fn save(&self, session: Session) -> Result<()> {
        // Serialize session context (scalars only, not Planes)
        // Planes live in Blackboard → separate Lance dataset
        // Each save = new Lance version (automatic ACID)
        lance::write_dataset(session_to_arrow(session), &self.dataset_path).await?;
        Ok(())
    }
    
    async fn get(&self, session_id: &str) -> Result<Option<Session>> {
        let ds = lance::dataset(&self.dataset_path).await?;
        // Read latest version by default
        let row = ds.filter(format!("session_id = '{}'", session_id)).await?;
        Ok(row.map(arrow_to_session))
    }
    
    /// Time travel: load session at specific version
    async fn get_at_version(&self, session_id: &str, version: u64) -> Result<Option<Session>> {
        let ds = lance::dataset(&self.dataset_path).await?;
        let historical = ds.checkout(version)?;
        let row = historical.filter(format!("session_id = '{}'", session_id)).await?;
        Ok(row.map(arrow_to_session))
    }
}
```

---

## STEP 5: What graph-flow Replaces in Layer 4

```
CURRENT LAYER 4 (messy):                  REPLACED BY:
──────────────────────────────────────────────────────────────
crewai-rust agent orchestration     →  GraphBuilder + Tasks + edges
n8n-rs workflow definitions         →  add_edge, add_conditional_edge
jitson JIT for hot loops            →  KEEP (inside Tasks for SIMD kernels)
Manual state passing between layers →  Context get/set (thread-safe)
No session persistence              →  LanceSessionStorage (versioned)
No conditional routing              →  add_conditional_edge (predicate fn)
No parallel execution               →  FanOutTask (tokio parallel)
No human-in-the-loop                →  NextAction::WaitForInput
No step-by-step debugging           →  NextAction::Continue + session inspect
No execution history                →  Session history + Lance versions
```

jitson stays. It's the JIT compiler for hot-path SIMD kernels INSIDE tasks.
graph-flow is the orchestration BETWEEN tasks. They're complementary.

---

## STEP 6: The Complete Wiring

```
LAYER 0:  Lance format persistence (S3, NVMe, versioning)
            ↕ LanceSessionStorage
LAYER 4:  graph-flow execution engine (GraphBuilder, FlowRunner, Session)
            ↕ PlaneContext bridge
BLACKBOARD: rustynum zero-copy arena (SIMD-aligned Planes)
            ↕ simd:: dispatch (AVX-512, AVX2, scalar)
LAYERS 1-10: individual Tasks in the execution graph
            ↕ conditional routing based on cascade bands, seal status, confidence

The graph IS the thinking.
The Context IS the Blackboard.
The Session IS the episode.
The Lance version IS the memory.
```

---

## FILES TO PRODUCE

```
1. Clone and inventory rs-graph-llm/graph-flow (types, traits, methods)
2. PlaneContext bridge implementation (Context → Blackboard zero-copy)
3. Thinking graph definition (10 layers as Tasks with conditional edges)
4. FanOut for 2³ SPO projections
5. LanceSessionStorage implementation
6. Integration tests: full thinking cycle from input to output
7. Benchmark: graph-flow overhead vs direct function calls
```

## DEPENDENCIES TO ADD

```
graph-flow = { git = "https://github.com/a-agmon/rs-graph-llm", path = "graph-flow" }
# OR vendor into our workspace as a local crate
```

## KEY PRINCIPLE

graph-flow handles ORCHESTRATION (which task runs next, what conditions route where).
rustynum handles COMPUTATION (SIMD hamming, encounter, cascade, bundle).
Lance handles PERSISTENCE (S3, versioning, time travel).

Each layer does ONE thing. RISC for the thinking stack.
No more crewai+n8n+jitson soup in Layer 4.
One clean graph. One execution model. One session store.
