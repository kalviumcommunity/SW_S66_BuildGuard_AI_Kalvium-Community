"""Pydantic request/response contracts for the RAG backend API.

These models define the JSON surface of the service and give FastAPI the
information it needs to validate incoming requests and emit clear, consistent
error bodies — Task 2 (structured JSON) and Task 3 (validation / errors).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryRequest(BaseModel):
    """Incoming question plus optional per-request pipeline overrides."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=1,
        description="Natural-language NBC question to answer.",
        examples=["What is the fire resistance rating for boiler room walls?"],
    )
    candidate_k: int | None = Field(
        default=None, ge=1, le=50,
        description="Retrieval pool size before re-ranking (defaults to env config).",
    )
    final_k: int | None = Field(
        default=None, ge=1, le=20,
        description="Chunks kept after re-ranking for context (defaults to env config).",
    )
    filter_heading: str | None = Field(
        default=None,
        description="Optional NBC section heading to scope retrieval to.",
    )
    hybrid: bool | None = Field(
        default=None,
        description="Enable lexical keyword boosting alongside dense vector search.",
    )
    dry_run: bool | None = Field(
        default=None,
        description="Skip the LLM call and return a synthetic answer (no API key).",
    )

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty or whitespace only")
        return value.strip()


class Source(BaseModel):
    """A single retrieved-and-cited chunk backing the answer."""

    rank: int
    id: str
    source: str
    volume: Any = ""
    page: Any = "?"
    heading: str = ""
    cosine_similarity: float | None = None
    rerank_score: float | None = None
    text_preview: str = ""


class QueryMetadata(BaseModel):
    """Operational metadata describing how the request was served."""

    model: str
    embedding_model: str
    candidate_k: int
    final_k: int
    rerank_method: str
    hybrid: bool
    dry_run: bool
    source_count: int
    context_chars: int
    pipeline_elapsed_ms: float
    stages: dict[str, Any] = Field(default_factory=dict)
    generated_at: str


class QueryResponse(BaseModel):
    """Structured, grounded answer with sources and status."""

    status: str = Field(description="'success', 'no_context', or 'error'.")
    answer: str
    query: str
    sources: list[Source] = Field(default_factory=list)
    metadata: QueryMetadata | None = None


class HealthResponse(BaseModel):
    """Liveness/config-readiness probe output."""

    status: str
    config_loaded: bool
    chat_configured: bool
    missing_chat_settings: list[str] = Field(default_factory=list)
    embedding_model: str
    collection_name: str


class ErrorResponse(BaseModel):
    """Uniform error envelope returned for 4xx/5xx responses."""

    status: str = "error"
    error: str
    detail: str | None = None
    fields: list[str] | None = None


class IndexingSummary(BaseModel):
    """Outcome of ingesting, embedding, and indexing one uploaded document."""

    document: str
    media_type: str
    bytes: int
    pages: int
    chunks_created: int
    chunks_indexed: int
    collection: str
    embedding_model: str
    total_records_after: int
    doc_hash: str
    elapsed_ms: float


class UploadResponse(BaseModel):
    """Response for a successful document upload + indexing (Task 2/3)."""

    status: str = "indexed"
    document: str
    stored_path: str
    indexing: IndexingSummary
    chunk_ids: list[str] = Field(default_factory=list)
