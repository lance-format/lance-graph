"""Reusable FastAPI component for the Lance knowledge graph service."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import pyarrow as pa
import yaml
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .config import KnowledgeGraphConfig
from .service import LanceKnowledgeGraph
from .store import LanceGraphStore


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    rows: List[Dict[str, Any]]
    row_count: int


class VectorQueryRequest(BaseModel):
    """Request body for vector-reranked Cypher queries.

    Supply either ``vector`` (raw floats) or ``query_text`` (auto-embedded via OpenAI).
    """

    query: str = Field(..., description="Cypher statement to execute.")
    column: str = Field(..., description="Name of the vector column to search.")

    # Choose one: pass the vector directly or pass the text
    # and let the server automatically embed it
    vector: Optional[List[float]] = Field(
        None,
        description="Query vector (float list). Mutually exclusive with query_text.",
    )
    query_text: Optional[str] = Field(
        None, description="Text to embed as query vector. Requires OpenAI API key."
    )

    metric: Literal["cosine", "l2", "dot"] = Field(
        "cosine", description="Distance metric: cosine | l2 | dot."
    )
    top_k: int = Field(10, ge=1, le=10000, description="Number of nearest neighbours.")
    include_distance: bool = Field(
        True, description="Include _distance column in results."
    )
    embedding_model: str = Field(
        "text-embedding-3-small",
        description="OpenAI embedding model (only used when query_text is provided).",
    )


class VectorQueryResponse(BaseModel):
    rows: List[Dict[str, Any]]
    row_count: int
    column: str
    metric: str
    top_k: int


class DatasetUpsertRequest(BaseModel):
    records: List[Dict[str, Any]]
    merge: bool = True


class KnowledgeGraphComponent:
    """Bundle FastAPI routes that expose the Lance knowledge graph."""

    def __init__(self, config: Optional[KnowledgeGraphConfig] = None):
        self._config = config or KnowledgeGraphConfig.default()
        self._service: Optional[LanceKnowledgeGraph] = None
        self.router = APIRouter(tags=["knowledge-graph"])
        self._setup_routes()

    def _get_service(self) -> LanceKnowledgeGraph:
        if self._service is None:
            try:
                self._service = _create_service(self._config)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
        return self._service

    def _setup_routes(self) -> None:
        @self.router.get("/health")
        async def health() -> Dict[str, str]:
            return {"status": "healthy", "service": "lance-knowledge-graph"}

        @self.router.get("/datasets")
        async def list_datasets() -> Dict[str, List[str]]:
            service = self._get_service()
            names = list(service.dataset_names())
            return {"datasets": names}

        @self.router.get("/datasets/{name}")
        async def get_dataset(name: str, limit: int = 100) -> Dict[str, Any]:
            service = self._get_service()
            if not service.has_dataset(name):
                raise HTTPException(
                    status_code=404, detail=f"Dataset '{name}' not found"
                )

            table = service.load_table(name)
            rows = table.to_pylist()
            if limit is not None:
                rows = rows[:limit]
            return {"name": name, "row_count": len(rows), "rows": rows}

        @self.router.post("/datasets/{name}")
        async def upsert_dataset(
            name: str, request: DatasetUpsertRequest
        ) -> Dict[str, Any]:
            if not request.records:
                raise HTTPException(status_code=400, detail="records cannot be empty")

            table = pa.Table.from_pylist(request.records)
            service = self._get_service()
            service.upsert_table(name, table, merge=request.merge)
            return {"status": "ok", "dataset": name, "row_count": table.num_rows}

        @self.router.post("/query", response_model=QueryResponse)
        async def execute_query(request: QueryRequest) -> QueryResponse:
            service = self._get_service()
            result = service.query(request.query)
            rows = result.to_pylist()
            return QueryResponse(rows=rows, row_count=len(rows))

        @self.router.get("/schema")
        async def get_schema() -> Dict[str, Any]:
            schema_path = self._config.resolved_schema_path()
            if not schema_path.exists():
                raise HTTPException(status_code=404, detail="Schema file not found")
            with schema_path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
            return {"path": str(schema_path), "schema": payload}

        @self.router.post("/query/vector", response_model=VectorQueryResponse)
        async def execute_vector_query(
            request: VectorQueryRequest,
        ) -> VectorQueryResponse:
            """Execute a Cypher query with vector similarity reranking.

            Supply ``vector`` (raw floats) or ``query_text`` (auto-embedded).
            """
            if request.vector is None and request.query_text is None:
                raise HTTPException(
                    status_code=400,
                    detail="Either 'vector' or 'query_text' must be provided.",
                )
            if request.vector is not None and request.query_text is not None:
                raise HTTPException(
                    status_code=400,
                    detail="Provide only one of 'vector' or 'query_text', not both.",
                )

            service = self._get_service()

            try:
                if request.query_text is not None:
                    # Text: service internally calls EmbeddingGenerator
                    result = service.query_by_text(
                        request.query,
                        request.query_text,
                        request.column,
                        top_k=request.top_k,
                        metric=request.metric,
                        include_distance=request.include_distance,
                        embedding_model=request.embedding_model,
                    )
                else:
                    # Vector: Constructing VectorSearch directly
                    from lance_graph import DistanceMetric, VectorSearch

                    _metric_map = {
                        "cosine": DistanceMetric.Cosine,
                        "l2": DistanceMetric.L2,
                        "dot": DistanceMetric.Dot,
                    }
                    vs = (
                        VectorSearch(request.column)
                        .query_vector(request.vector)
                        .metric(_metric_map[request.metric])
                        .top_k(request.top_k)
                        .include_distance(request.include_distance)
                    )
                    result = service.run_with_vector_rerank(request.query, vs)

            except RuntimeError as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            rows = result.to_pylist()
            return VectorQueryResponse(
                rows=rows,
                row_count=len(rows),
                column=request.column,
                metric=request.metric,
                top_k=request.top_k,
            )

    def close(self) -> None:
        """Release retained resources."""
        self._service = None


def _create_service(config: KnowledgeGraphConfig) -> LanceKnowledgeGraph:
    graph_config = config.load_graph_config()
    storage = LanceGraphStore(config)
    service = LanceKnowledgeGraph(graph_config, storage=storage)
    service.ensure_initialized()
    return service
