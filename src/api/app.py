"""FastAPI application exposing the BuildGuard RAG pipeline over HTTP.

Endpoints
---------
POST /query   — accept a question, run the RAG pipeline, return a grounded
                answer with structured sources + metadata as JSON.
GET  /health  — report service liveness and whether env config is present.

Tasks addressed
---------------
  Task 1 — POST /query accepts a question and returns an answer + sources.
  Task 2 — The endpoint calls ``run_pipeline`` and returns structured JSON
           (``answer``, ``sources``, ``status``, ``metadata``).
  Task 3 — Request validation (Pydantic) with clear status codes: 422 for bad
           or missing fields, 503 for a misconfigured live generation, 500 for
           unexpected server/pipeline errors — all with uniform error envelopes.
  Task 4 — Every setting (API keys, model names, vector-DB location, pipeline
           defaults, server binding) is loaded from the environment via
           ``src.api.config``.

Run locally
-----------
    uv run uvicorn src.api.app:app --reload
    # or
    uv run python -m src.api.app
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.rag_pipeline import run_pipeline

from .config import APIConfig, get_config
from .models import (
    ErrorResponse,
    HealthResponse,
    QueryMetadata,
    QueryRequest,
    QueryResponse,
    Source,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="BuildGuard RAG API",
    version="1.0.0",
    description="Grounded National Building Code question-answering service.",
)


# ---------------------------------------------------------------------------
# Error handlers — uniform JSON envelopes (Task 3)
# ---------------------------------------------------------------------------


@app.exception_handler(RequestValidationError)
async def _validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Turn FastAPI/Pydantic validation errors into a clear 422 body."""
    fields: list[str] = []
    for err in exc.errors():
        loc = [str(part) for part in err.get("loc", []) if part != "body"]
        fields.append(".".join(loc) or "body")
    body = ErrorResponse(
        error="invalid_request",
        detail="Request body failed validation.",
        fields=sorted(set(fields)),
    )
    return JSONResponse(status_code=422, content=body.model_dump())


@app.exception_handler(ValidationError)
async def _pydantic_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """Cover model validation raised inside handlers."""
    fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
    body = ErrorResponse(
        error="invalid_request", detail="Request body failed validation.", fields=fields
    )
    return JSONResponse(status_code=422, content=body.model_dump())


@app.exception_handler(Exception)
async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return a 500 envelope for unexpected pipeline/server failures."""
    logger.exception("Unhandled error during request: %s", exc)
    body = ErrorResponse(
        error="internal_error",
        detail="The RAG pipeline failed to process this request.",
    )
    return JSONResponse(status_code=500, content=body.model_dump())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_dry_run(request: QueryRequest, config: APIConfig) -> bool:
    return config.default_dry_run if request.dry_run is None else request.dry_run


def _build_metadata(result: Any, *, config: APIConfig, dry_run: bool) -> QueryMetadata:
    stages = result.stages or {}
    embed_stage = stages.get("embed", {})
    retrieve_stage = stages.get("retrieve", {})
    rerank_stage = stages.get("rerank", {})
    assemble_stage = stages.get("assemble", {})
    generate_stage = stages.get("generate", {})
    return QueryMetadata(
        model=generate_stage.get("model", config.chat_model or "unknown"),
        embedding_model=embed_stage.get("model", config.embedding_model),
        candidate_k=retrieve_stage.get("candidate_k", config.candidate_k),
        final_k=rerank_stage.get("final_k", config.final_k),
        rerank_method=rerank_stage.get("method", config.rerank_method),
        hybrid=bool(retrieve_stage.get("hybrid", True)),
        dry_run=dry_run,
        source_count=len(result.sources),
        context_chars=assemble_stage.get("context_chars", len(result.context_block or "")),
        pipeline_elapsed_ms=result.pipeline_elapsed_ms,
        stages=stages,
        generated_at=result.generated_at,
    )


def _to_sources(raw_sources: list[dict[str, Any]]) -> list[Source]:
    items: list[Source] = []
    for s in raw_sources:
        items.append(
            Source(
                rank=int(s.get("rank", 0)),
                id=str(s.get("id", "")),
                source=str(s.get("source", "")),
                volume=s.get("volume", ""),
                page=s.get("page", "?"),
                heading=str(s.get("heading", "") or ""),
                cosine_similarity=s.get("cosine_similarity"),
                rerank_score=s.get("rerank_score"),
                text_preview=str(s.get("text_preview", "") or ""),
            )
        )
    return items


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report liveness and whether required env config is present."""
    config = get_config()
    missing = config.missing_chat_settings()
    return HealthResponse(
        status="ok",
        config_loaded=True,
        chat_configured=not missing,
        missing_chat_settings=missing,
        embedding_model=config.embedding_model,
        collection_name=config.collection_name,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Run the RAG pipeline for *request.question* and return grounded JSON."""
    config = get_config()
    dry_run = _resolve_dry_run(request, config)

    # A live generation needs chat credentials; a misconfiguration is 503.
    if not dry_run:
        missing = config.missing_chat_settings()
        if missing:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Live generation is not configured. Missing environment "
                    "variable(s): " + ", ".join(missing)
                ),
            )

    metadata_filter = None
    if request.filter_heading:
        from src.embeddings.filtered_retrieval import build_metadata_filter

        metadata_filter = build_metadata_filter(heading=request.filter_heading)

    logger.info("POST /query  question=%r  dry_run=%s", request.question[:80], dry_run)

    result = run_pipeline(
        request.question,
        candidate_k=request.candidate_k or config.candidate_k,
        final_k=request.final_k or config.final_k,
        metadata_filter=metadata_filter,
        hybrid=True if request.hybrid is None else request.hybrid,
        rerank_method=config.rerank_method,
        model_name=config.embedding_model,
        persist_dir=config.persist_dir,
        collection_name=config.collection_name,
        dry_run=dry_run,
    )

    sources = _to_sources(result.sources)
    status = "success" if sources else "no_context"

    return QueryResponse(
        status=status,
        answer=result.answer,
        query=result.query,
        sources=sources,
        metadata=_build_metadata(result, config=config, dry_run=dry_run),
    )


def main() -> None:
    """Start the API server with settings from the environment."""
    import uvicorn

    config = get_config()
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("src.api.app:app", host=config.host, port=config.port)


if __name__ == "__main__":
    main()
