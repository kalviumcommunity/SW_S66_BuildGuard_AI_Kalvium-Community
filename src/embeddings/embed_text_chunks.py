"""Embed reviewed text chunks and store them in a persistent Chroma collection.

The source JSONL is never modified. Re-running the command upserts the same
chunk IDs, so an unchanged input can be indexed again without duplicates.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "outputs" / "chunk_review" / "text_chunks.headings_rebuilt.jsonl"
DEFAULT_PERSIST_DIR = ROOT / "outputs" / "vector_db"
RUN_DIR = ROOT / "outputs" / "embeddings"
DEFAULT_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-dot-v1"
DEFAULT_COLLECTION = "nbc_v1_multi_qa_minilm_l6_dot_v1"
DEFAULT_BATCH_SIZE = 64


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


def run(
    *,
    input_path: Path,
    persist_dir: Path,
    collection_name: str,
    model_name: str,
    batch_size: int,
    device: str | None,
    limit: int | None = None,
) -> dict[str, Any]:
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    log_path = _configure_logging()
    if not input_path.is_file():
        raise FileNotFoundError(f"Chunk input not found: {input_path}")
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1")
    if limit is not None and limit < 1:
        raise ValueError("Limit must be at least 1")

    records = _read_records(input_path)
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

    total = len(records)
    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]
        vectors = model.encode(
            [record["text"] for record in batch],
            batch_size=min(batch_size, len(batch)),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        collection.upsert(
            ids=[record["id"] for record in batch],
            documents=[record["text"] for record in batch],
            metadatas=[_chroma_metadata(record) for record in batch],
            embeddings=vectors.tolist(),
        )
        logging.info("Indexed %d/%d chunks", min(start + len(batch), total), total)

    report = {
        "pipeline": "embed_chunks",
        "version": "1.0",
        "status": "completed",
        "started_at": started_at,
        "input": str(input_path),
        "persist_dir": str(persist_dir),
        "collection": collection_name,
        "embedding_model": model_name,
        "embedding_dimensions": dimensions,
        "distance_metric": "cosine",
        "batch_size": batch_size,
        "chunks_indexed_this_run": total,
        "collection_count_after_run": collection.count(),
        "log": str(log_path),
        "note": "Text and metadata are stored in Chroma; embeddings are generated locally.",
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
            run(
                input_path=args.input.resolve(),
                persist_dir=args.persist_dir.resolve(),
                collection_name=args.collection,
                model_name=args.model,
                batch_size=args.batch_size,
                device=args.device,
                limit=args.limit,
            )
    except Exception as exc:
        logging.exception("Embedding run failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
