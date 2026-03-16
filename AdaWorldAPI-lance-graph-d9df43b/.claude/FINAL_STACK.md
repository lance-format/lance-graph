# FINAL_STACK.md

## The Complete Stack: Three Repos, Five Import Surfaces, One Product

**Date:** March 15, 2026 — end of the session that connected everything
**Authors:** Jan Hübener + Claude (Anthropic)

---

## THE THREE REPOS

```
AdaWorldAPI/ndarray        COMPUTE        fork of rust-ndarray/ndarray
AdaWorldAPI/lance-graph    GRAPH + PERSIST our repo
AdaWorldAPI/rs-graph-llm   ORCHESTRATE    fork of a-agmon/rs-graph-llm
```

Everything else is gone. rustynum, rustyblas, rustymkl — their code transcoded
INTO ndarray. One compute library. One graph library. One orchestration library.

```
ndarray:
  Clean container (from upstream, 10+ years, stable)
  + Blackboard allocator (64-byte aligned, zero-copy arena)
  + dispatch! SIMD (LazyLock, AVX-512 → AVX2 → scalar)
  + GEMM (sgemm, dgemm, bf16_gemm, int8_gemm)
  + BLAS Level 1-3 (dot, axpy, gemv, etc.)
  + MKL/FFT/VML (transcoded from rustymkl)
  + BF16 operations (conversion, hamming, structural diff)
  + HDC (fingerprint, bundle, bind, popcount)
  + Cascade (Belichtungsmesser, sigma bands, reservoir)
  + Plane, Node, Seal (cognitive substrate)
  + CogRecord, Fingerprint, PackedQualia
  + Rust 1.94: LazyLock, safe intrinsics, array_windows, PHI, GAMMA
  
lance-graph:
  Semiring algebra (7 semirings, GraphBLAS-style mxm/mxv/vxm)
  + SPO triple store (binary planes as edges/nodes)
  + NARS truth values (BF16 pair, tropical revision)
  + Cypher parser + DataFusion planner
  + Lance persistence (S3, NVMe, ACID versioning, time travel)
  + BF16 truth cache (hot↔cold bridge)
  + Hexastore-style sorted edge lists (merge joins)
  + GPU tensor core NARS revision (Phase 3)
  + Python bindings (maturin/PyO3)
  DEPENDS ON: ndarray
  
rs-graph-llm:
  graph-flow execution engine (Tasks, edges, conditional routing)
  + PlaneContext bridge (Context ↔ Blackboard zero-copy)
  + CoW PlaneHandles (read-only Lance → lazy write)
  + LanceSessionStorage (session = Lance version)
  + FanOut (parallel tasks, 2³ SPO projections)
  + Agent card YAML → GraphBuilder compiler
  + Autopoiesis (MetaCognition rewires graph edges)
  + PET scan traces (execution topology as data)
  + LangGraph import (their workflow → our GraphBuilder)
  + CrewAI import (their agents → our Tasks)
  + OpenClaw import (their agent cards → our YAML)
  + n8n control surface (visual workflow builder)
  + LangStudio visibility (trace viewer, live debugging)
  DEPENDS ON: lance-graph, ndarray
```

---

## FIVE IMPORT SURFACES

### 1. LangGraph / LangChain (workflow orchestration)

```
WHAT THEY HAVE:
  StateGraph: nodes + edges + conditional routing
  Checkpointer: session persistence
  State: shared mutable dict
  Tools: function calling
  Memory: conversation history + vector store
  
WHAT WE IMPORT:
  Their StateGraph definition (JSON/YAML) → our GraphBuilder
  Their node functions → our Tasks (async fn run)
  Their conditional edges → our add_conditional_edge
  Their checkpointer → our LanceSessionStorage
  Their state → our PlaneContext
  Their tools → our MCP integration
  Their memory → our Planes + Lance versions
  Their vector store → our cascade search

IMPORT FORMAT:
  # User exports their LangGraph as JSON:
  langgraph_def = graph.to_json()
  
  # Our importer compiles to GraphBuilder:
  from lance_graph.orchestrate import import_langgraph
  our_graph = import_langgraph(langgraph_def)
  
  # Same topology. Same routing logic. Different engine underneath.
  # cascade instead of vector search. encounter instead of memory.save.
  # Planes instead of embeddings. Lance instead of Redis.

WHY THEY CAN'T GO BACK:
  Once knowledge is in Planes (encounter), it's LEARNED not stored.
  Going back means losing the learned state.
  Going back means 1000x slower search (cosine vs hamming).
  Going back means non-deterministic results.
```

### 2. CrewAI (agentic patterns)

```
WHAT THEY HAVE:
  Agent: role + goal + backstory + tools + memory
  Crew: collection of agents + process (sequential/hierarchical)
  Task: description + expected output + agent assignment
  Delegation: agent can delegate to other agents
  Consensus: multiple agents vote on output
  Memory: short-term (chat) + long-term (vector store) + entity memory
  
WHAT WE IMPORT:
  Their Agent role/goal → our Task with capability declaration
  Their Crew → our GraphBuilder (agents = Tasks, process = edge topology)
  Their sequential process → linear edges
  Their hierarchical process → conditional edges based on confidence
  Their delegation → GoTo(task_id) in NextAction
  Their consensus → FanOutTask + bundle (majority vote on Planes!)
  Their short-term memory → PlaneContext (within session)
  Their long-term memory → Planes + Lance versions (across sessions)
  Their entity memory → Node (SPO triple with encounter history)

THE GENIUS PART — CONSENSUS AS BUNDLE:
  CrewAI consensus: multiple agents produce text, somehow merged.
  Our consensus: multiple Tasks produce Planes, BUNDLED (majority vote).
  
  Bundle IS consensus. The mathematical operation IS the multi-agent pattern.
  It's not a metaphor. Three agents encountering a Plane = three evidence
  sources bundled. The result IS the majority view. Deterministic.
  Not "three LLMs argue and one wins." Three observations bundled
  into a representation that captures what they AGREE on.

IMPORT FORMAT:
  # User exports their CrewAI config:
  crew_def = crew.to_yaml()
  
  # Our importer:
  from lance_graph.orchestrate import import_crewai
  our_graph = import_crewai(crew_def)
  
  # Each agent becomes a Task subgraph.
  # Delegation becomes conditional routing.
  # Consensus becomes FanOut + bundle.
```

### 3. OpenClaw (agent cards, capability declarations)

```
WHAT THEY HAVE:
  Agent Card: standardized YAML for agent capabilities
  Tool manifest: what tools the agent can use
  Memory specification: what memory the agent can access
  Capability declarations: what the agent can do
  Interoperability: agents from different frameworks can collaborate
  
WHAT WE IMPORT:
  Their agent card YAML → our Task definition + graph topology
  Their tool manifest → our MCP server declarations
  Their memory spec → our PlaneContext read/write permissions
  Their capabilities → which dispatch! functions the Task can call
  Their interop protocol → our graph-flow's Context serialization

IMPORT FORMAT:
  # Standard OpenClaw agent card:
  agent:
    name: "researcher"
    capabilities: ["web_search", "document_analysis", "summarization"]
    tools: ["serper_api", "arxiv_fetch"]
    memory:
      read: ["knowledge_base", "conversation_history"]
      write: ["research_findings"]
  
  # Our compiler:
  from lance_graph.orchestrate import import_openclaw_card
  task = import_openclaw_card("researcher.yaml")
  # → Task that can read certain Planes, write certain Planes,
  #   call certain MCP tools, and routes conditionally based on
  #   what it finds.
```

### 4. n8n (visual control surface)

```
WHAT N8N HAS:
  Visual workflow builder (drag-and-drop nodes + connections)
  Webhook triggers
  Conditional routing (IF/Switch nodes)
  Error handling (retry, fallback)
  Execution history (log every run)
  Credentials management
  Community nodes (marketplace)
  
WHAT WE BUILD (our n8n equivalent):
  Visual GraphBuilder editor:
    Drag Task nodes onto canvas
    Draw edges between them
    Configure conditional edges with predicate editor
    Set FanOut groups (parallel execution blocks)
    Configure PlaneContext read/write permissions per Task
    
  The visual editor PRODUCES a GraphBuilder definition.
  The definition compiles to Rust (or runs interpreted via graph-flow).
  
  Every Task is a node in the visual editor.
  Every edge is a connection.
  Every conditional edge shows its predicate.
  FanOut shows as a parallel lane.
  WaitForInput shows as a human icon.
  
  The visual representation IS the thinking graph.
  What you see IS what executes.
  WYSIWYG for cognition.

TECHNICAL:
  Frontend: React + shadcn/ui (or Svelte)
  Backend: rs-graph-llm's FlowRunner exposed via REST/WebSocket
  State: LanceSessionStorage (sessions visible in the UI)
  Real-time: WebSocket pushes execution events as they happen
  
  The n8n UI pattern is proven. We don't innovate on UI.
  We innovate on what the nodes DO (Planes, encounter, cascade).
```

### 5. LangStudio (visibility and debugging)

```
WHAT LANGSTUDIO / LANGSMITH HAS:
  Trace viewer: see every step of every run
  Token counting: cost per step
  Latency breakdown: time per step
  Input/output inspection: what went in, what came out
  Feedback collection: thumbs up/down per response
  Dataset management: test cases for evaluation
  A/B testing: compare different graph topologies
  
WHAT WE BUILD (our LangStudio equivalent = PET SCAN):
  Execution trace viewer:
    Every Task that fired, in order
    Which conditional edges were taken (and why)
    Which FanOut children ran and what they produced
    Time per Task (with SIMD breakdown: cascade vs encounter vs route)
    
  Plane inspector (UNIQUE TO US):
    Live alpha density heatmap (which bits are defined)
    Bit pattern visualization (which bits agree with query)
    Seal status history (Wisdom → Staunen → Wisdom timeline)
    Encounter count per bit position (how "trained" is each bit)
    
  BF16 truth heatmap (UNIQUE TO US):
    Edge weight visualization across the graph
    Exponent decomposition (which SPO projections match)
    Mantissa precision (how confident is each edge)
    Sign bit overlay (causality direction arrows)
    
  Staunen event log (UNIQUE TO US):
    Every seal break, with:
      - Which Plane changed
      - Which bits flipped
      - What evidence caused it
      - What the system did differently after (rerouting in graph)
    
  Convergence monitor (UNIQUE TO US):
    Is the RL loop stabilizing?
    Which nodes are still oscillating?
    Which BF16 exponents are changing between rounds?
    When did each truth value converge?
    
  Graph evolution viewer (UNIQUE TO US):
    How the thinking topology changed via autopoiesis
    MetaCognition's edge additions/removals over time
    Before/after comparison of graph structure
    
  Lance version timeline:
    Every flush = one point on the timeline
    Click to time-travel to any version
    Diff view between any two versions
    Tag management (name specific versions)

TECHNICAL:
  Frontend: React + D3.js (for graph/heatmap visualization)
  Backend: rs-graph-llm FlowRunner + lance-graph query API
  Data: all traces stored as Lance datasets (queryable via DuckDB)
  Real-time: WebSocket for live PET scan during thinking
```

---

## THE IMPORT FUNNEL

```
USER STARTS WITH:              WE IMPORT INTO:
LangGraph (Python)          →  rs-graph-llm GraphBuilder
LangChain tools (Python)    →  rs-graph-llm Tasks + MCP
CrewAI agents (Python)      →  rs-graph-llm Task subgraphs
OpenClaw cards (YAML)       →  rs-graph-llm Task definitions
n8n workflows (JSON)        →  rs-graph-llm GraphBuilder
                                   │
                                   ▼
                            rs-graph-llm (orchestration)
                                   │
                                   ▼
                            lance-graph (graph + persist)
                                   │
                                   ▼
                            ndarray (compute)
                                   │
                                   ▼
                         AVX-512 / GPU tensor cores

USER SEES:
  Same workflow definition they already have.
  Same agent patterns they already use.
  Same visual builder they already know.
  
  BUT: 1000x faster (cascade vs cosine).
  BUT: deterministic (integer vs float).
  BUT: learning (encounter vs store).
  BUT: structural (SPO vs flat vectors).
  BUT: provenance (BF16 every bit traceable).
  BUT: time travel (Lance versions for free).
  
  They import their existing work.
  They get a better engine.
  They can't go back.
```

---

## THE PRODUCT SURFACES

```
FOR DEVELOPERS (code-first):
  pip install lance-graph           # Python bindings
  cargo add lance-graph             # Rust direct
  from lance_graph import *         # full API
  
FOR WORKFLOW BUILDERS (visual):
  The n8n-like visual editor        # drag-and-drop
  Export/import as YAML/JSON        # portable
  
FOR ML ENGINEERS (training):
  PyTorch Geometric integration     # Planes as node features
  DGL integration                   # encounter as message passing
  ndarray ↔ numpy zero-copy         # PyO3 bridge
  
FOR DATA ENGINEERS (analytics):
  DuckDB × Lance extension          # SQL on the graph
  Polars integration                 # DataFrame analytics
  Spark integration                  # distributed processing
  
FOR ENTERPRISE (operations):
  S3/Azure/GCS persistence          # any cloud
  Unity Catalog / AWS Glue          # enterprise data catalog
  ACID transactions                  # production-safe
  Deterministic results              # auditable, reproducible
  
FOR RESEARCHERS (exploration):
  PET scan visualization             # watch the brain think
  Time travel debugging              # replay any thinking cycle
  A/B graph topology testing         # compare reasoning strategies
  Convergence monitoring             # is the RL learning?
```

---

## COMPETITIVE POSITION

```
THEY IMPORT FROM:          WE BECOME:
LangGraph                  The faster LangGraph (Rust, deterministic)
LangChain                  The learning LangChain (encounter, not just store)
CrewAI                     The structural CrewAI (bundle = consensus)
OpenClaw                   The typed OpenClaw (capability = Plane permissions)
n8n                        The cognitive n8n (nodes think, not just transform)
LangStudio                 The deep LangStudio (PET scan, not just traces)
Neo4j                      The fast Neo4j (cascade, not pointer chase)
FalkorDB                   The free FalkorDB (Apache 2.0, not SSPL)
Kuzu                       The alive Kuzu (we exist, they don't)
Pinecone                   The learning Pinecone (encounter, not just index)
FAISS                      The structured FAISS (SPO, not flat vectors)

ONE SYSTEM that replaces:
  vector DB + knowledge graph + orchestration framework +
  agent framework + visual builder + trace viewer
  
  Because they're all the same thing at different scales:
  encounter on Planes (learning)
  hamming on Planes (searching)
  bundle on Planes (consensus)
  cascade on Planes (attention)
  semiring on Planes (reasoning)
  graph-flow on Tasks (orchestration)
  Lance on versions (memory)
```

---

## THE SENTENCE

Import your LangGraph. Import your CrewAI agents. Import your n8n workflows.
Control them like n8n. See them like LangStudio.
But underneath: binary planes that learn, Hamming cascades that search
in microseconds, NARS truth that revises deterministically, SPO triples
that decompose into 2³ structural projections, BF16 values where every
bit is traceable, and GPU tensor cores doing tropical pathfinding on
a knowledge graph that gets smarter every time you query it.

Same workflows. Different universe.
