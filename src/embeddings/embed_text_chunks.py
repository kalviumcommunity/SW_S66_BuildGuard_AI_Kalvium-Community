"""Embed reviewed text chunks and store them in a persistent Chroma collection.

The source JSONL is never modified. Re-runs skip chunks whose ID, text, and
stored vector are already present. Changed text is embedded and replaces the
old record under the same ID.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "outputs" / "chunk_review" / "text_chunks.headings_rebuilt.jsonl"
DEFAULT_PERSIST_DIR = ROOT / "outputs" / "vector_db"
RUN_DIR = ROOT / "outputs" / "embeddings"
DEFAULT_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-dot-v1"
DEFAULT_COLLECTION = "nbc_v1_multi_qa_minilm_l6_dot_v1"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_SECONDS = 1.0
DEFAULT_COST_PER_MILLION_TOKENS_USD = 0.0
T = TypeVar("T")


def _safe_collection_name(value: str) -> str:
    """Return a Chroma-compatible collection name."""
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_-").lower()
    if not 3 <= len(name) <= 63:
        raise ValueError("Collection name must contain 3 to 63 valid characters")
    if not name[0].isalnum() or not name[-1].isalnum():
        raise ValueError("Collection name must start and end with a letter or number")
    return name


def _read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            record_id = record.get("id")
            text = record.get("text")
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError(f"Missing string id on line {line_number}")
            if record_id in seen_ids:
                raise ValueError(f"Duplicate chunk id {record_id!r} on line {line_number}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Missing or empty text for chunk {record_id!r}")
            if not isinstance(record.get("metadata", {}), dict):
                raise ValueError(f"Metadata must be an object for chunk {record_id!r}")
            seen_ids.add(record_id)
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def _chroma_metadata(record: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Flatten metadata into Chroma's supported scalar values."""
    raw = dict(record.get("metadata") or {})
    raw["chunk_id"] = record["id"]
    clean: dict[str, str | int | float | bool] = {}
    for key, value in raw.items():
        if value is None:
            continue
        if isinstance(value, bool):
            clean[key] = value
        elif isinstance(value, int):
            clean[key] = value
        elif isinstance(value, float):
            if math.isfinite(value):
                clean[key] = value
        elif isinstance(value, str):
            clean[key] = value
        elif isinstance(value, (list, tuple)):
            clean[key] = ", ".join(str(item) for item in value if item is not None)
        else:
            clean[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return clean


def _configure_logging() -> Path:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RUN_DIR / "embedding.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return log_path


def _is_transient_error(exc: Exception) -> bool:
    """Identify common rate-limit and temporary service/storage failures."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    status = getattr(exc, "status_code", getattr(exc, "status", None))
    try:
        if int(status) in {408, 425, 429} or int(status) >= 500:
            return True
    except (TypeError, ValueError):
        pass
    name = type(exc).__name__.lower()
    message = str(exc).lower()
    transient_markers = (
        "rate limit",
        "ratelimit",
        "temporarily unavailable",
        "temporary failure",
        "connection reset",
        "connection aborted",
        "timed out",
        "timeout",
        "database is locked",
        "service unavailable",
    )
    return any(marker in name or marker in message for marker in transient_markers)


def _retry_call(
    action: Callable[[], T],
    *,
    label: str,
    max_retries: int,
    base_delay_seconds: float,
    counters: dict[str, int],
) -> T:
    """Retry transient failures with capped exponential backoff."""
    for attempt in range(max_retries + 1):
        try:
            return action()
        except Exception as exc:
            if attempt >= max_retries or not _is_transient_error(exc):
                raise
            delay = min(base_delay_seconds * (2**attempt), 30.0)
            counters["retry_attempts"] += 1
            logging.warning(
                "%s failed transiently (%s). Retry %d/%d in %.1f seconds",
                label,
                exc,
                attempt + 1,
                max_retries,
                delay,
            )
            time.sleep(delay)

    raise RuntimeError(f"Retry loop ended unexpectedly for {label}")


def _estimated_tokens(record: dict[str, Any]) -> int:
    """Use chunk token metadata, with a whitespace estimate as fallback."""
    token_count = (record.get("metadata") or {}).get("token_count")
    if isinstance(token_count, int) and token_count >= 0:
        return token_count
    return len(record["text"].split())


def run(
    *,
    input_path: Path,
    persist_dir: Path,
    collection_name: str,
    model_name: str,
    batch_size: int,
    device: str | None,
    limit: int | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_base_seconds: float = DEFAULT_RETRY_BASE_SECONDS,
    cost_per_million_tokens_usd: float = DEFAULT_COST_PER_MILLION_TOKENS_USD,
) -> dict[str, Any]:
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    log_path = _configure_logging()
    if not input_path.is_file():
        raise FileNotFoundError(f"Chunk input not found: {input_path}")
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("Limit must be at least 1")
    if max_retries < 0:
        raise ValueError("Maximum retries cannot be negative")
    if retry_base_seconds < 0 or not math.isfinite(retry_base_seconds):
        raise ValueError("Retry base delay must be a finite nonnegative number")
    if cost_per_million_tokens_usd < 0 or not math.isfinite(cost_per_million_tokens_usd):
        raise ValueError("Cost rate must be a finite nonnegative number")

    source_records = _read_records(input_path)
    source_total = len(source_records)
    records = source_records
    if limit is not None:
        records = records[:limit]
    collection_name = _safe_collection_name(collection_name)

    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer

    logging.info("Embedding model: %s", model_name)
    logging.info("Input: %s (%d chunks)", input_path, len(records))
    logging.info("Persistent database: %s", persist_dir)
    logging.info("Collection: %s", collection_name)
    model_kwargs = {"device": device} if device else {}
    model = SentenceTransformer(model_name, **model_kwargs)
    dimensions = int(model.get_sentence_embedding_dimension())
    client = PersistentClient(path=str(persist_dir))
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "embedding_model": model_name},
    )
    stored_model = (collection.metadata or {}).get("embedding_model")
    if stored_model and stored_model != model_name:
        raise ValueError(
            f"Collection {collection_name!r} was created with model {stored_model!r}; "
            "choose a different --collection for another model"
        )

    existing = collection.get(include=["documents", "embeddings"])
    existing_documents = existing.get("documents")
    existing_vectors = existing.get("embeddings")
    existing_by_id = {
        record_id: (
            existing_documents[index] if existing_documents is not None else None,
            existing_vectors[index] if existing_vectors is not None else None,
        )
        for index, record_id in enumerate(existing["ids"])
    }
    pending = []
    for record in records:
        stored_document, stored_vector = existing_by_id.get(record["id"], (None, None))
        has_embedding = stored_vector is not None and len(stored_vector) > 0
        if stored_document == record["text"] and has_embedding:
            continue
        pending.append(record)
    skipped_chunks = len(records) - len(pending)
    counters = {"retry_attempts": 0}
    failures: list[dict[str, Any]] = []
    embeddings_generated = 0
    chunks_stored = 0
    tokens_embedded = 0
    batch_total = math.ceil(len(pending) / batch_size)
    logging.info(
        "Input chunks: %d; unchanged chunks skipped: %d; batches to process: %d",
        len(records),
        skipped_chunks,
        batch_total,
    )

    for batch_number, start in enumerate(range(0, len(pending), batch_size), start=1):
        batch = pending[start : start + batch_size]
        batch_ids = [record["id"] for record in batch]
        logging.info("Embedding batch %d/%d (%d chunks)", batch_number, batch_total, len(batch))
        try:
            vectors = _retry_call(
                lambda: model.encode(
                    [record["text"] for record in batch],
                    batch_size=min(batch_size, len(batch)),
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ),
                label=f"Embedding batch {batch_number}/{batch_total}",
                max_retries=max_retries,
                base_delay_seconds=retry_base_seconds,
                counters=counters,
            )
        except Exception as exc:
            logging.exception("Embedding batch %d/%d failed", batch_number, batch_total)
            failures.append({"batch": batch_number, "stage": "embedding", "ids": batch_ids, "error": str(exc)})
            continue

        embeddings_generated += len(batch)
        tokens_embedded += sum(_estimated_tokens(record) for record in batch)
        try:
            _retry_call(
                lambda: collection.upsert(
                    ids=batch_ids,
                    documents=[record["text"] for record in batch],
                    metadatas=[_chroma_metadata(record) for record in batch],
                    embeddings=vectors.tolist(),
                ),
                label=f"Saving batch {batch_number}/{batch_total} to Chroma",
                max_retries=max_retries,
                base_delay_seconds=retry_base_seconds,
                counters=counters,
            )
            chunks_stored += len(batch)
            logging.info("Saved batch %d/%d; indexed %d/%d pending chunks", batch_number, batch_total, chunks_stored, len(pending))
        except Exception as exc:
            logging.exception("Saving batch %d/%d failed", batch_number, batch_total)
            failures.append({"batch": batch_number, "stage": "storage", "ids": batch_ids, "error": str(exc)})

    estimated_cost = (tokens_embedded / 1_000_000) * cost_per_million_tokens_usd
    failed_chunks = sum(len(failure["ids"]) for failure in failures)

    report = {
        "pipeline": "embed_chunks",
        "version": "2.0",
        "status": "completed_with_failures" if failed_chunks else "completed",
        "started_at": started_at,
        "input": str(input_path),
        "persist_dir": str(persist_dir),
        "collection": collection_name,
        "embedding_model": model_name,
        "embedding_dimensions": dimensions,
        "distance_metric": "cosine",
        "batch_size": batch_size,
        "batches_attempted": batch_total,
        "total_chunks": source_total,
        "selected_chunks": len(records),
        "unselected_chunks_due_to_limit": source_total - len(records),
        "embeddings_generated": embeddings_generated,
        "chunks_stored": chunks_stored,
        "skipped_chunks": skipped_chunks,
        "failed_chunks": failed_chunks,
        "failed_batches": failures,
        "retry_attempts": counters["retry_attempts"],
        "estimated_input_tokens": tokens_embedded,
        "estimated_token_count_basis": (
            "Chunk metadata token_count; whitespace count is used only when metadata is missing."
        ),
        "cost_rate_per_million_tokens_usd": cost_per_million_tokens_usd,
        "estimated_embedding_cost_usd": round(estimated_cost, 8),
        "cost_estimate_basis": (
            "Local Sentence Transformers inference has no per-token API charge. "
            "A nonzero configured rate is a user-supplied estimate for a comparable hosted embedding service."
        ),
        "collection_count_after_run": collection.count(),
        "log": str(log_path),
        "note": "Chunks are embedded locally in batches and unchanged IDs/text are skipped.",
    }
    report_path = RUN_DIR / "embedding_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logging.info("Finished. Report: %s", report_path)
    return report


def query_collection(
    *,
    query: str,
    persist_dir: Path,
    collection_name: str,
    model_name: str,
    top_k: int,
    device: str | None,
) -> dict[str, Any]:
    """Search an existing collection with a text query and save the matches."""
    log_path = _configure_logging()
    if not query.strip():
        raise ValueError("Query cannot be empty")
    if top_k < 1:
        raise ValueError("Top-k must be at least 1")

    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer

    collection_name = _safe_collection_name(collection_name)
    client = PersistentClient(path=str(persist_dir))
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as exc:
        raise ValueError(
            f"Collection {collection_name!r} was not found in {persist_dir}. "
            "Run the embedding command first."
        ) from exc

    stored_model = (collection.metadata or {}).get("embedding_model")
    if stored_model and stored_model != model_name:
        raise ValueError(
            f"Collection {collection_name!r} uses model {stored_model!r}, "
            f"but query requested {model_name!r}"
        )
    count = collection.count()
    if count == 0:
        raise ValueError(f"Collection {collection_name!r} is empty")

    logging.info("Loading query model: %s", model_name)
    model_kwargs = {"device": device} if device else {}
    model = SentenceTransformer(model_name, **model_kwargs)
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    response = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    ids = response["ids"][0]
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]
    results = [
        {
            "rank": rank,
            "id": record_id,
            "cosine_distance": float(distances[rank - 1]),
            "text": documents[rank - 1],
            "metadata": metadatas[rank - 1] or {},
        }
        for rank, record_id in enumerate(ids, start=1)
    ]
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RUN_DIR / "query_results.json"
    report = {
        "query": query,
        "collection": collection_name,
        "embedding_model": model_name,
        "top_k": len(results),
        "results": results,
        "log": str(log_path),
    }
    result_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logging.info("Query returned %d matches. Results: %s", len(results), result_path)
    for result in results:
        metadata = result["metadata"]
        logging.info(
            "#%d %s distance=%.4f source=%s page=%s",
            result["rank"],
            result["id"],
            result["cosine_distance"],
            metadata.get("source", ""),
            metadata.get("page_start", ""),
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Chunk JSONL to embed")
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR, help="Chroma persistence directory")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence Transformers model name")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Retries for transient batch failures")
    parser.add_argument("--retry-base-seconds", type=float, default=DEFAULT_RETRY_BASE_SECONDS, help="Initial exponential backoff delay")
    parser.add_argument(
        "--cost-per-million-tokens-usd",
        type=float,
        default=DEFAULT_COST_PER_MILLION_TOKENS_USD,
        help="Optional user-supplied hosted embedding rate; defaults to $0 for local inference",
    )
    parser.add_argument("--device", help="Optional Sentence Transformers device, for example cpu or cuda")
    parser.add_argument("--limit", type=int, help="Optional maximum number of chunks (useful for a small initial run)")
    parser.add_argument("--query", help="Search the existing collection instead of embedding chunks")
    parser.add_argument("--top-k", type=int, default=5, help="Number of matches to return with --query")
    args = parser.parse_args()
    try:
        if args.query is not None:
            report = query_collection(
                query=args.query,
                persist_dir=args.persist_dir.resolve(),
                collection_name=args.collection,
                model_name=args.model,
                top_k=args.top_k,
                device=args.device,
            )
            for result in report["results"]:
                metadata = result["metadata"]
                print(
                    f"\n#{result['rank']} {result['id']} "
                    f"(distance={result['cosine_distance']:.4f}; "
                    f"{metadata.get('source', '')}, page {metadata.get('page_start', '')})\n"
                    f"{result['text'][:700]}"
                )
        else:
            report = run(
                input_path=args.input.resolve(),
                persist_dir=args.persist_dir.resolve(),
                collection_name=args.collection,
                model_name=args.model,
                batch_size=args.batch_size,
                device=args.device,
                limit=args.limit,
                max_retries=args.max_retries,
                retry_base_seconds=args.retry_base_seconds,
                cost_per_million_tokens_usd=args.cost_per_million_tokens_usd,
            )
            print(
                "Run summary: "
                f"selected={report['selected_chunks']}/{report['total_chunks']}, "
                f"generated={report['embeddings_generated']}, "
                f"skipped={report['skipped_chunks']}, failed={report['failed_chunks']}, "
                f"retries={report['retry_attempts']}, "
                f"estimated_cost=${report['estimated_embedding_cost_usd']:.8f}"
            )
            if report["failed_chunks"]:
                return 2
    except Exception as exc:
        logging.exception("Embedding run failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
