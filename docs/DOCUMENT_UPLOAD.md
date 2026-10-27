# Document Upload & Indexing Endpoint

## 1. Overview & Objectives

The RAG API answers questions from an existing NBC corpus. This feature lets the
knowledge base **grow at runtime**: an authenticated caller uploads a document,
and the service ingests it, cleans and chunks its text, embeds the chunks with
the **same model** used to index the original corpus, and writes them into the
**same ChromaDB collection** the `/query` endpoint already searches — so the new
content becomes answerable **immediately, with no restart**.

### Goals
- **Runtime ingestion** — `POST /documents/upload` accepts a file and indexes it live.
- **Reuse the existing pipeline** — the embedded chunks land in the query endpoint's
  collection using the identical L2-normalized, cosine-space convention, so
  hybrid retrieval + cross-encoder re-ranking work unchanged.
- **Graceful, explicit errors** — unsupported formats, empty files, oversized
  files, and processing failures map to distinct HTTP status codes.
- **Safe storage** — uploads are sanitized (no path traversal) and stored under a
  configurable directory with a content-hash-prefixed name.

---

## 2. Task Coverage

| Task | Where implemented |
|:---|:---|
| **Task 1 — Upload endpoint, safe storage** | `POST /documents/upload` ([`src/api/app.py`](../src/api/app.py)); `store_upload` / `sanitize_filename` ([`src/api/upload.py`](../src/api/upload.py)) |
| **Task 2 — Ingest → embed → index** | `index_document` → `extract_pages` → `chunk_pages` → `embed_and_index` ([`src/api/upload.py`](../src/api/upload.py)) |
| **Task 3 — Runtime searchability** | Upsert into the query collection; proven by [`outputs/rag_api/upload/`](../outputs/rag_api/upload) |
| **Task 4 — Upload error handling** | `UploadError` subclasses + HTTPException envelope ([`src/api/app.py`](../src/api/app.py)) |
| **Task 5 — Committed sample run** | [`outputs/rag_api/upload/*.json`](../outputs/rag_api/upload) |

---

## 3. Endpoint

### `POST /documents/upload`

`multipart/form-data` with a single `file` field.

**Success — `201 Created` (`UploadResponse`)**

```json
{
  "status": "indexed",
  "document": "buildguard_addendum_2026.txt",
  "stored_path": "<upload_dir>/45d2654352-buildguard_addendum_2026.txt",
  "indexing": {
    "document": "buildguard_addendum_2026.txt",
    "media_type": ".txt",
    "bytes": 356,
    "pages": 1,
    "chunks_created": 1,
    "chunks_indexed": 1,
    "collection": "nbc_v1_multi_qa_minilm_l6_dot_v1",
    "embedding_model": "sentence-transformers/multi-qa-MiniLM-L6-dot-v1",
    "total_records_after": 1234,
    "doc_hash": "45d2654352f2944a",
    "elapsed_ms": 22387.8
  },
  "chunk_ids": ["upload-45d2654352f2944a-000000"]
}
```

---

## 4. Ingestion Flow (Task 2)

```
Upload (bytes + filename)
   │
   ├─ validate_upload        → format / empty / size checks (4xx on failure)
   ├─ sha256(content)        → deterministic doc_hash + chunk ids (idempotent upsert)
   ├─ extract_pages          → PDF via PyMuPDF; txt/md direct; html tags stripped; cleaned
   ├─ chunk_pages            → token-aware (≤ 500 tokens), per-page boundaries
   ├─ store_upload           → sanitized, hash-prefixed file under UPLOAD_DIR
   └─ embed_and_index        → SentenceTransformer(normalize=True) → Chroma upsert
                                       │
                                       ▼
                             SAME collection /query reads → searchable now
```

Chunk ids are `upload-<hash16>-<index>`, so re-uploading the same document
**upserts** (replaces) rather than creating duplicates.

---

## 5. Runtime Searchability (Task 3)

Each request opens the Chroma collection fresh, so records added by an upload are
visible to the very next `/query`. The committed [`sample_upload_run.json`](../outputs/rag_api/upload/sample_upload_run.json)
drives **one in-process app** through:

1. `POST /documents/upload` — a `BuildGuard Internal Addendum 2026` stating a
   rooftop-helipad minimum static load of **1500 kg/m²** (content absent from the
   NBC corpus).
2. `POST /query` — *"What is the minimum static load a rooftop helipad must
   support on a Type-2 residential building?"*

The follow-up query returns the uploaded chunk as **rank 1**
(`"id": "upload-45d2654352f2944a-000000"`), confirming the new content is
retrievable without restarting the app. The demo writes to an isolated temp
Chroma directory so the project's real corpus is untouched.

---

## 6. Error Handling (Task 4)

| Code | `error` label | Trigger |
|:---|:---|:---|
| `400` | `empty_file` / `bad_request` | 0-byte upload, missing filename, no chunks extractable. |
| `413` | `file_too_large` | Upload exceeds `MAX_UPLOAD_BYTES`. |
| `415` | `unsupported_format` | Suffix not in `ALLOWED_EXTENSIONS`. |
| `422` | `invalid_request` | No `file` part / malformed multipart body. |
| `500` | `processing_error` / `internal_error` | Embedding, model, or vector-DB failure (e.g. a collection built with a different embedding model). |

Errors share the uniform envelope:

```json
{ "status": "error", "error": "unsupported_format", "detail": "Unsupported file type '.exe'. Allowed: .htm, .html, .md, .pdf, .txt." }
```

---

## 7. Configuration (Environment Variables)

Upload settings join the existing env config in `src/api/config.py`.

| Variable | Default | Purpose |
|:---|:---|:---|
| `UPLOAD_DIR` | `outputs/uploads` | Directory where accepted files are stored. |
| `MAX_UPLOAD_BYTES` | `10000000` (10 MB) | Rejected above this size (`413`). |
| `ALLOWED_EXTENSIONS` | `.pdf,.txt,.md,.markdown,.html,.htm` | Accepted file types (`415` otherwise). |
| `CHROMA_PERSIST_DIR` / `CHROMA_COLLECTION` / `EMBEDDING_MODEL` | (see `.env.example`) | Uploads index into this exact collection so `/query` finds them. |

Uploaded files live under `outputs/uploads/` (runtime data, not committed). The
demo sample artifacts under `outputs/rag_api/upload/` are committed.

---

## 8. Running & Testing

```bash
# Start the server
uv run uvicorn src.api.app:app --reload

# Upload a document
curl -X POST http://127.0.0.1:8000/documents/upload \
  -F "file=@./my_addendum.txt"

# Ask the running app about it (no restart)
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the minimum static load for a rooftop helipad?", "dry_run": true}'

# Regenerate the committed upload sample
uv run python -m src.api.generate_upload_sample

# Tests (pure text handling + endpoint with the indexing step stubbed)
uv run python -m pytest tests/test_upload_api.py
```
