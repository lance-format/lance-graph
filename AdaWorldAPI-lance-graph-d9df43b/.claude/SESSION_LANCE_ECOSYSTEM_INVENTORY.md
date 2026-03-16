# SESSION_LANCE_ECOSYSTEM_INVENTORY.md

## Mission: What Does the Original Lance Ecosystem Do That We Probably Missed?

**Context:** We forked lance-graph and built our binary semiring algebra on top.
But the ORIGINAL lance ecosystem (lance-format/lance, LanceDB, lance-graph,
DuckDB extension, lance-namespace) has capabilities we may not be using.
They connect externally to S3, Azure, GCS, DuckDB, Polars, Spark, PyTorch.
We could connect those same pipes INTERNALLY to our binary planes.

**The question:** What infrastructure did lance-format build that we get FOR FREE
by being a lance-native system — and what are we NOT using?

---

## STEP 1: Inventory the Original Lance Format (lance-format/lance)

```
REPO: https://github.com/lance-format/lance
```

### Storage Backends (What They Connect To)
```
INVESTIGATE:
  1. S3 (AWS) — lance.dataset("s3://bucket/path")
  2. S3 Express One Zone — ultra-low latency single-zone
  3. Azure Blob Store — lance.dataset("az://bucket/path")
  4. Google Cloud Storage — lance.dataset("gs://bucket/path")
  5. S3-compatible (MinIO, R2, Tigris) — custom endpoint
  6. Local NVMe / POSIX filesystem
  7. EBS (AWS block storage)
  8. EFS (AWS network filesystem)
  
  HOW WE USE THIS:
    Our Planes (2KB binary vectors) stored as Lance columns.
    Lance handles S3/Azure/GCS persistence transparently.
    We get cloud storage for free. No custom persistence layer.
    
  WHAT TO CHECK:
    - Can we store Plane.acc (i8[16384] = 16KB) as a Lance column efficiently?
    - Can we store Fingerprint<256> (u64[256] = 2KB) as a fixed-size binary column?
    - Can we store BF16 edge weights as a Lance column?
    - What's the random access latency for reading one Plane from S3?
    - Can we use S3 Express for hot path data (< 10ms latency)?
```

### Versioning and Time Travel
```
INVESTIGATE:
  Lance creates a new VERSION on every write. Old versions are readable.
  dataset.checkout(version=N) reads state at version N.
  ACID transactions. Zero-copy.
  
  HOW WE USE THIS:
    Every round of encounter() changes Plane state.
    If Planes are Lance columns, each encounter round = new version.
    Time travel = "what did this node know at time T?"
    diff(v1, v2) = which bits changed = learning delta = Staunen detection
    
  WHAT TO CHECK:
    - What's the storage overhead per version? (immutable fragments = COW)
    - Can we version at the Plane level or only at the dataset level?
    - How fast is diff between two versions?
    - Can we tag versions? (e.g., tag = "after_training_epoch_50")
    - Can we branch? (experiment with different encounter sequences)
```

### Data Evolution (Add Columns Without Rewrite)
```
INVESTIGATE:
  lance supports add_columns() with backfilled values.
  No full table rewrite. Just new column fragments.
  
  HOW WE USE THIS:
    Add new Planes to existing Nodes without rewriting S, P, O.
    Add BF16 edge weight cache column without rewriting the graph.
    Add alpha density statistics column for query optimizer.
    
  WHAT TO CHECK:
    - Can add_columns work with computed columns (UDF)?
    - Can we add a "bundle" column that's computed from existing planes?
    - Performance of add_columns on 1M row dataset?
```

### Concurrent Writers
```
INVESTIGATE:
  S3 doesn't support atomic commits. Lance uses DynamoDB for locking.
  lance.dataset("s3+ddb://bucket/path?ddbTableName=mytable")
  Also supports custom commit_lock context manager.
  
  HOW WE USE THIS:
    Multiple encounter() streams writing to same graph concurrently.
    Hot path (cascade discovery) writes new edges.
    Cold path (query evaluation) reads edges.
    Both can run simultaneously with proper locking.
    
  WHAT TO CHECK:
    - What's the overhead of DynamoDB locking per write?
    - Can we use optimistic concurrency instead? (version check)
    - Is there a local-only locking mechanism for embedded use?
```

---

## STEP 2: Inventory LanceDB (the Vector DB built on Lance)

```
REPO: https://github.com/lancedb/lancedb
```

### Vector Search
```
INVESTIGATE:
  LanceDB provides vector similarity search on Lance datasets.
  IVF-PQ index for approximate nearest neighbor.
  Full-text search (BM25).
  Hybrid search (vector + full-text + SQL).
  
  HOW WE USE THIS:
    We DON'T use float vector search. We use Hamming on binary planes.
    BUT: LanceDB's indexing infrastructure might support binary indices.
    IVF-PQ on binary vectors → multi-index hashing equivalent?
    
  WHAT TO CHECK:
    - Does LanceDB support binary vector types?
    - Can we register a custom distance function (hamming)?
    - Can we use their index infrastructure with our cascade as the distance?
    - Their ANN index vs our cascade: which is faster for binary search?
```

### Table API
```
INVESTIGATE:
  db = lancedb.connect("s3://bucket/path")
  table = db.create_table("nodes", data)
  table.add(new_data)
  table.search(query_vector).limit(10)
  table.where("column > value")
  table.to_pandas() / table.to_arrow() / table.to_polars()
  
  HOW WE USE THIS:
    Our Nodes could be a LanceDB table.
    table.search(query_plane).limit(10) → cascade under the hood.
    table.where("alpha_density > 0.8") → SQL filter on plane metadata.
    table.to_arrow() → zero-copy to DuckDB/Polars for analytics.
    
  WHAT TO CHECK:
    - Can we use custom search functions with LanceDB?
    - Can we store 2KB binary columns efficiently?
    - What's the overhead of LanceDB's table API vs raw Lance?
```

### Ecosystem Integrations
```
INVESTIGATE:
  LanceDB integrates with:
    - LangChain (RAG pipeline)
    - LlamaIndex (RAG pipeline)
    - PyTorch / PyTorch Geometric (GNN training dataloader)
    - DGL (graph neural networks)
    - Pandas, Polars, DuckDB (analytics)
    - Hugging Face (model hub)
    
  HOW WE USE THIS:
    LangChain/LlamaIndex: our graph as knowledge source for RAG
    PyTorch Geometric: our Planes as node features for GNN
    DGL: same
    Pandas/Polars: analytics on graph metadata
    DuckDB: SQL on graph (the DuckDB × Lance extension!)
    
  WHAT TO CHECK:
    - PyG integration: can it load our Planes as node features directly?
    - DGL integration: can it use our encounter() as message passing?
    - DuckDB extension: can we query our graph with SQL?
    - LangChain: can we replace their vector store with our cascade?
```

---

## STEP 3: Inventory Original lance-graph (lance-format/lance-graph)

```
REPO: https://github.com/lance-format/lance-graph
```

### What It Actually Does
```
INVESTIGATE:
  CypherQuery → DataFusion plan → Arrow table scan + filter + join
  GraphConfig with node labels and ID columns
  Knowledge graph service (CLI + FastAPI)
  LLM-powered text extraction for knowledge graph bootstrap
  
  WHAT TO CHECK:
    - How does their CypherQuery map to DataFusion?
    - What Cypher features do they support?
    - How does their GraphConfig compare to our BlasGraph?
    - What is their knowledge_graph service architecture?
    - Do they have any graph algorithms (BFS, shortest path, PageRank)?
    - How do they handle relationships? Arrow tables with foreign keys?
```

### Their Python Bindings
```
INVESTIGATE:
  pip install lance-graph
  from lance_graph import CypherQuery, GraphConfig
  
  WHAT TO CHECK:
    - How do they expose Rust to Python? (PyO3? maturin?)
    - What's their Python API surface?
    - Can we extend their API with our binary operations?
    - Is their CypherQuery reusable as-is?
```

### Their Knowledge Graph Service
```
INVESTIGATE:
  knowledge_graph package:
    - CLI for init, query, bootstrap
    - FastAPI web service
    - LLM text extraction (OpenAI)
    - Lance-backed storage
    
  HOW WE USE THIS:
    Their web service pattern for our graph API.
    Their LLM extraction → our encounter() pipeline.
    Text → LLM → triples → encounter() into Planes.
    
  WHAT TO CHECK:
    - Can we plug our binary graph into their FastAPI service?
    - Can we replace their LLM extraction with encounter()?
    - What endpoints do they expose?
```

---

## STEP 4: Inventory DuckDB × Lance Extension

```
SOURCE: https://lancedb.com/blog/lance-x-duckdb-sql-retrieval-on-the-multimodal-lakehouse-format/
```

```
INVESTIGATE:
  INSTALL lance FROM community;
  LOAD lance;
  SELECT * FROM 's3://bucket/path/dataset.lance' LIMIT 5;
  
  DuckDB can:
    - Scan Lance datasets directly (local or S3)
    - Push down column selection and filters
    - Hybrid retrieval with vector search table functions
    - Join Lance datasets with other DuckDB tables
    
  HOW WE USE THIS:
    Our binary graph stored as Lance → queryable from DuckDB SQL.
    
    SELECT node_id, alpha_density 
    FROM 's3://bucket/nodes.lance' 
    WHERE alpha_density > 0.8;
    
    SELECT a.node_id, b.node_id, hamming_distance(a.plane_s, b.plane_s)
    FROM 's3://bucket/nodes.lance' a, 's3://bucket/nodes.lance' b
    WHERE hamming_distance(a.plane_s, b.plane_s) < 100;
    
  WHAT TO CHECK:
    - Can we register custom DuckDB functions (hamming_distance)?
    - Can DuckDB push down binary predicates to Lance scan?
    - What's the performance of DuckDB scanning our 2KB binary columns?
    - Can we use DuckDB's parallel execution for graph queries?
```

---

## STEP 5: Inventory lance-namespace

```
REPO: https://github.com/lance-format/lance-namespace
```

```
INVESTIGATE:
  Standardized access to collections of Lance tables.
  Implementations for: Hive, Polaris, Gravitino, Unity Catalog, AWS Glue.
  
  HOW WE USE THIS:
    Our graph as a namespace in enterprise catalog.
    Unity Catalog → our SPO graph accessible from Databricks.
    AWS Glue → our graph accessible from any AWS analytics tool.
    
  WHAT TO CHECK:
    - Can we register our graph as a lance-namespace?
    - What metadata does namespace expose?
    - Can we expose our graph through Unity Catalog?
```

---

## STEP 6: What We Get FOR FREE by Being Lance-Native

After completing the inventory, produce this table:

```
CAPABILITY                           LANCE PROVIDES     WE USE IT?
──────────────────────────────────────────────────────────────────────
S3 persistence                       ✓                   ?
Azure Blob persistence               ✓                   ?
GCS persistence                      ✓                   ?
NVMe-optimized local storage         ✓                   ?
ACID versioning (time travel)        ✓                   ?
Zero-copy Arrow interop              ✓                   ?
Concurrent writers (DynamoDB lock)   ✓                   ?
Column evolution (add without rewrite) ✓                 ?
DuckDB SQL queries                   ✓                   ?
Pandas/Polars export                 ✓                   ?
PyTorch dataloader                   ✓                   ?
LangChain/LlamaIndex RAG            ✓                   ?
Spark integration                    ✓                   ?
Ray integration                      ✓                   ?
Enterprise catalog (Unity, Glue)     ✓                   ?
Full-text search (BM25)              ✓                   ?
Vector similarity search (IVF-PQ)    ✓                   ?
```

For every "?" answer: what would it take to wire it up?
For every "no": why not, and should we?

---

## STEP 7: The Internal Connection Map

The key insight: they connect EXTERNALLY (S3, DuckDB, Spark).
We connect INTERNALLY (Plane ↔ Node ↔ Cascade ↔ Semiring).

Can we expose our internal connections through their external interfaces?

```
THEIR EXTERNAL:                    OUR INTERNAL EQUIVALENT:
S3 → Lance dataset                 Lance dataset → Plane columns
DuckDB SQL → Lance scan            DuckDB SQL → hamming_distance UDF
LangChain retriever → LanceDB      LangChain retriever → cascade search
PyG node features → Lance column   PyG node features → Plane.bits()
Spark → Lance table                Spark → graph analytics on Planes
```

The BRIDGE: make our internal types (Plane, Node, Edge, NarsTruth)
look like Lance columns to the external ecosystem. Then everything
they built works with our data. No custom integration needed.

```rust
// Make a Plane serializable as a Lance column:
impl From<Plane> for lance::array::BinaryArray {
    fn from(plane: Plane) -> Self {
        // acc: i8[16384] → 16KB binary blob
        // OR bits: u64[256] → 2KB binary blob + alpha: u64[256] → 2KB binary blob
    }
}

// Make a Node serializable as a Lance row:
impl From<Node> for lance::RecordBatch {
    fn from(node: Node) -> Self {
        // columns: [id, plane_s, plane_p, plane_o, seal, encounters]
    }
}

// Make an Edge serializable as a Lance row:
impl From<Edge> for lance::RecordBatch {
    fn from(edge: Edge) -> Self {
        // columns: [source_id, target_id, bf16_truth, projection_byte]
    }
}
```

Once this bridge exists:
- `lance.dataset("s3://bucket/my_graph.lance")` reads our graph from S3
- DuckDB scans our graph with SQL
- PyTorch Geometric loads our Planes as GNN node features
- LangChain uses our cascade as a retriever
- Databricks accesses our graph through Unity Catalog
- We get the ENTIRE lance ecosystem for free

---

## FILES TO PRODUCE

```
1. LANCE_ECOSYSTEM_INVENTORY.md      — complete capability map
2. LANCE_FREE_CAPABILITIES.md        — what we get for free, what's not wired
3. LANCE_BRIDGE_SPEC.md              — Plane/Node/Edge → Lance column serialization
4. LANCE_INTEGRATION_PRIORITY.md     — which integrations to wire first
```

Push all to `.claude/` in both repos.

---

## KEY INSIGHT

We built the cognitive layer (encounter, cascade, bundle, semiring).
Lance built the infrastructure layer (S3, versioning, DuckDB, PyTorch).
We are ON TOP of Lance but not USING Lance.
Wiring the two together gives us:

```
A database that LEARNS (our layer)
  + persists to any cloud (Lance S3/Azure/GCS)
  + queries from SQL (DuckDB × Lance)
  + trains from PyTorch (Lance dataloader)
  + integrates with RAG (LangChain × Lance)
  + catalogs in enterprise (Unity Catalog × lance-namespace)
  + versions automatically (Lance ACID)
  + time-travels for free (Lance checkout)

No other system has this stack.
Because no other system is both a learning engine AND a Lance-native store.
```
