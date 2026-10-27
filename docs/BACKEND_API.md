# Backend API for the RAG Service

## 1. Overview & Objectives

The BuildGuard RAG pipeline already answers National Building Code (NBC)
questions end-to-end, but only from the command line. This service wraps that
pipeline in a **FastAPI** HTTP backend so a frontend or another service can
`POST` a natural-language question and receive a **grounded answer with
structured source citations as JSON**.

### Goals
- **Usable over the network** — a single `POST /query` endpoint drives the full
  embed → retrieve → rerank → assemble → generate pipeline.
- **Structured, contract-first JSON** — Pydantic request/response models define
  the exact shape of answers, sources, status, and operational metadata.
- **Predictable validation & errors** — clear HTTP status codes and a uniform
  error envelope for bad input, missing fields, misconfiguration, and server
  failures.
- **Environment-driven configuration** — API keys, model names, vector-database
  location, pipeline defaults, and server binding all come from environment
  variables — never hardcoded.

---

## 2. Task Coverage

| Task | Where implemented |
|:---|:---|
| **Task 1 — Query endpoint** | `POST /query` in [`src/api/app.py`](../src/api/app.py) |
| **Task 2 — Structured JSON** | `QueryResponse` / `Source` / `QueryMetadata` in [`src/api/models.py`](../src/api/models.py) |
| **Task 3 — Validation & errors** | Pydantic validation + exception handlers in [`src/api/app.py`](../src/api/app.py) |
| **Task 4 — Env configuration** | [`src/api/config.py`](../src/api/config.py) (`APIConfig.from_env`) |
| **Task 5 — Sample request/response** | [`outputs/rag_api/sample_request.json`](../outputs/rag_api/sample_request.json), [`sample_response.json`](../outputs/rag_api/sample_response.json) |

---

## 3. Endpoints

### `POST /query`

Runs the RAG pipeline for a question and returns a grounded answer.

**Request body (`QueryRequest`)**

| Field | Type | Required | Default | Description |
|:---|:---|:---|:---|:---|
| `question` | `string` | ✅ | — | Natural-language NBC question (non-blank). |
| `candidate_k` | `int` (1–50) | ❌ | env `RAG_CANDIDATE_K` | Retrieval pool size before re-ranking. |
| `final_k` | `int` (1–20) | ❌ | env `RAG_FINAL_K` | Chunks kept after re-ranking. |
| `filter_heading` | `string` | ❌ | `null` | Scope retrieval to an NBC section heading. |
| `hybrid` | `bool` | ❌ | `true` | Enable lexical keyword boosting. |
| `dry_run` | `bool` | ❌ | env `RAG_DRY_RUN` | Skip the LLM call; return a synthetic answer (no API key). |

Unknown extra fields are rejected (`extra="forbid"`).

**Response body (`QueryResponse`)**

```json
{
  "status": "success",
  "answer": "…grounded answer citing [Volume 1, PDF p. 314]…",
  "query": "…the original question…",
  "sources": [
    {
      "rank": 1,
      "id": "nbc-v1-p314-001012",
      "source": "nationalbuilding01.pdf",
      "volume": 1,
      "page": "314-315",
      "heading": "Part 4 — Fire and Life Safety",
      "cosine_similarity": 0.8264,
      "rerank_score": 4.0295,
      "text_preview": "120 min fire resistance rating wall…"
    }
  ],
  "metadata": {
    "model": "…chat model or 'dry-run'…",
    "embedding_model": "…",
    "candidate_k": 10,
    "final_k": 3,
    "rerank_method": "cross_encoder",
    "hybrid": true,
    "dry_run": false,
    "source_count": 3,
    "context_chars": 6938,
    "pipeline_elapsed_ms": 183982.0,
    "stages": { "embed": {"elapsed_ms": 78282.4}, "retrieve": {"elapsed_ms": 2788.0}, "…": "…" },
    "generated_at": "2026-10-27T06:20:13Z"
  }
}
```

**`status` semantics**
- `"success"` — the pipeline returned at least one cited source.
- `"no_context"` — the pipeline ran but retrieved no supporting chunks (the
  grounded answer is a refusal). HTTP `200`.
- `"error"` — returned only in the error envelope for `4xx`/`5xx` responses.

### `GET /health`

Liveness + configuration-readiness probe (`HealthResponse`): reports whether the
chat/LLM environment settings are present, plus the active embedding model and
Chroma collection name.

---

## 4. HTTP Status Codes

| Code | Meaning | Trigger |
|:---|:---|:---|
| `200` | Success | Valid question; answer produced (with or without sources). |
| `422` | Unprocessable Entity | Missing `question`, blank question, out-of-range `candidate_k`/`final_k`, or unknown fields. Body lists offending `fields`. |
| `503` | Service Unavailable | A **live** (non-dry-run) request while `API_BASE_URL` / `API_KEY` / `CHAT_MODEL` are not configured. |
| `500` | Internal Server Error | Unexpected pipeline/server failure. Uniform `{"status": "error", …}` envelope. |

Error envelope (`ErrorResponse`):

```json
{ "status": "error", "error": "invalid_request", "detail": "Request body failed validation.", "fields": ["question"] }
```

---

## 5. Configuration (Environment Variables)

All settings load through `APIConfig.from_env()` in `src/api/config.py`, which
reads a local `.env` (via `python-dotenv`) with sensible fallbacks matching the
CLI defaults. See `.env.example`.

| Variable | Default (fallback) | Purpose |
|:---|:---|:---|
| `API_BASE_URL` | — (required for live) | OpenAI-compatible chat endpoint. |
| `API_KEY` | — (required for live) | Chat API key. |
| `CHAT_MODEL` | — (required for live) | Generation model name. |
| `EMBEDDING_MODEL` | `sentence-transformers/multi-qa-MiniLM-L6-dot-v1` | Query embedding model. |
| `CHROMA_PERSIST_DIR` | `outputs/vector_db` | ChromaDB persistence directory. |
| `CHROMA_COLLECTION` | `nbc_v1_multi_qa_minilm_l6_dot_v1` | ChromaDB collection name. |
| `RAG_CANDIDATE_K` | `10` | Default retrieval pool size. |
| `RAG_FINAL_K` | `3` | Default chunks kept after re-ranking. |
| `RAG_RERANK_METHOD` | `cross_encoder` | Re-ranking strategy. |
| `RAG_DRY_RUN` | `false` | Default generation mode when `dry_run` is omitted. |
| `API_HOST` | `127.0.0.1` | Uvicorn bind host. |
| `API_PORT` | `8000` | Uvicorn bind port. |

Secrets are **never** read from code; the service can boot and serve `dry_run`
requests even with no chat credentials configured.

---

## 6. Running & Testing

```bash
# Start the server (host/port from env)
uv run uvicorn src.api.app:app --reload
# or
uv run python -m src.api.app

# Ask a question (dry-run — no API key needed)
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the fire resistance rating for boiler room walls?", "dry_run": true}'

# Regenerate the committed sample request/response
uv run python -m src.api.generate_sample

# Unit / contract tests (stubbed pipeline)
uv run python -m pytest tests/test_rag_api.py
```

The committed [`sample_response.json`](../outputs/rag_api/sample_response.json)
was produced by `generate_sample`, which drives the **real** endpoint
(embed → retrieve → rerank → assemble) against the local Chroma index and skips
only the LLM call, so the citations reflect genuine retrieval.

---

## 7. Design Notes

- **Pipeline is reused, not duplicated.** `POST /query` delegates to
  `src.rag_pipeline.run_pipeline`, keeping a single source of truth for the RAG
  logic; the API layer only adds HTTP contract, validation, and error mapping.
- **Lazy, patchable import.** `run_pipeline` is imported at module scope so
  tests can monkeypatch it without loading embedding models or the vector store.
- **Config is cached.** `get_config()` loads environment settings once; tests
  and the health probe can override it deterministically.
- **Per-request overrides vs env defaults.** Request fields (`candidate_k`,
  `final_k`, `hybrid`, `dry_run`) fall back to environment-configured defaults
  when omitted, so callers can rely on server-side configuration or override
  it per call.
