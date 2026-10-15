"""Evaluate text-embedding similarity, distance, and stored-vector quality.

This script is read-only: it never changes the Chroma collection. It embeds one
query to rank the most and least similar chunks, then checks stored vectors and
metadata for common indexing problems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

from .embed_text_chunks import DEFAULT_COLLECTION, DEFAULT_MODEL, DEFAULT_PERSIST_DIR
from .rank_text_chunks import (
    DEFAULT_QUERY,
    cosine_similarity,
    display_path,
    evenly_sample_indices,
    rank_collection,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs" / "embeddings" / "text_embedding_evaluation.json"
DEFAULT_PAIR_SAMPLE_SIZE = 128
NORMALIZED_TOLERANCE = 0.01


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "p05": None, "mean": None, "median": None, "p95": None, "maximum": None}
    return {
        "minimum": min(values),
        "p05": _percentile(values, 5),
        "mean": sum(values) / len(values),
        "median": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "maximum": max(values),
    }


def _vector_digest(vector: Sequence[float]) -> str:
    """Hash float values deterministically to detect exact stored duplicates."""
    payload = ",".join(format(float(value), ".9g") for value in vector)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _load_collection_records(
    persist_dir: Path, collection_name: str
) -> tuple[Any, list[dict[str, Any]]]:
    from chromadb import PersistentClient

    client = PersistentClient(path=str(persist_dir))
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as exc:
        raise ValueError(
            f"Collection {collection_name!r} was not found in {persist_dir}. "
            "Run text embedding first."
        ) from exc
    stored = collection.get(include=["documents", "metadatas", "embeddings"])
    documents = stored.get("documents")
    metadatas = stored.get("metadatas")
    vectors = stored.get("embeddings")
    if documents is None:
        documents = []
    if metadatas is None:
        metadatas = []
    if vectors is None:
        vectors = []
    records: list[dict[str, Any]] = []
    for index, record_id in enumerate(stored.get("ids", [])):
        records.append(
            {
                "id": record_id,
                "document": documents[index] if index < len(documents) else None,
                "metadata": metadatas[index] if index < len(metadatas) else None,
                "vector": vectors[index] if index < len(vectors) else None,
            }
        )
    return collection, records


def _quality_checks(records: list[dict[str, Any]], pair_sample_size: int) -> dict[str, Any]:
    """Check vector integrity, normalization, duplicate content, and metadata."""
    dimensions: Counter[int] = Counter()
    vector_digests: dict[str, list[str]] = {}
    document_digests: dict[str, list[str]] = {}
    invalid_vectors: list[str] = []
    zero_vectors: list[str] = []
    normalization_failures: list[str] = []
    metadata_failures: list[str] = []
    norms: list[float] = []
    valid_vectors: list[tuple[dict[str, Any], list[float]]] = []

    for record in records:
        record_id = str(record["id"])
        document = record["document"]
        metadata = record["metadata"]
        vector = record["vector"]
        if not isinstance(document, str) or not document.strip():
            metadata_failures.append(record_id)
        else:
            document_digests.setdefault(
                hashlib.sha256(document.encode("utf-8")).hexdigest(), []
            ).append(record_id)
        if not isinstance(metadata, dict):
            metadata_failures.append(record_id)
        else:
            source = metadata.get("source")
            volume = metadata.get("volume")
            page_start = metadata.get("page_start")
            page_end = metadata.get("page_end")
            if (
                not isinstance(source, str)
                or not source.strip()
                or not isinstance(volume, (str, int))
                or not isinstance(page_start, int)
                or not isinstance(page_end, int)
                or page_start < 1
                or page_end < page_start
            ):
                metadata_failures.append(record_id)
        if vector is None:
            invalid_vectors.append(record_id)
            continue
        values = [float(value) for value in vector]
        dimensions[len(values)] += 1
        if not values or not all(math.isfinite(value) for value in values):
            invalid_vectors.append(record_id)
            continue
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            zero_vectors.append(record_id)
            continue
        norms.append(norm)
        if abs(norm - 1.0) > NORMALIZED_TOLERANCE:
            normalization_failures.append(record_id)
        vector_digests.setdefault(_vector_digest(values), []).append(record_id)
        valid_vectors.append((record, values))

    ordered = sorted(
        valid_vectors,
        key=lambda item: (
            str((item[0]["metadata"] or {}).get("volume", "")),
            (item[0]["metadata"] or {}).get("page_start", 0),
            item[0]["id"],
        ),
    )
    chosen = evenly_sample_indices(len(ordered), pair_sample_size)
    sampled = [ordered[index] for index in chosen]
    similarities: list[float] = []
    for (_, left), (_, right) in combinations(sampled, 2):
        similarities.append(cosine_similarity(left, right))
    distances = [1.0 - similarity for similarity in similarities]

    duplicate_vectors = [ids for ids in vector_digests.values() if len(ids) > 1]
    duplicate_documents = [ids for ids in document_digests.values() if len(ids) > 1]
    dimensions_ok = len(dimensions) == 1 and not invalid_vectors
    passed = (
        dimensions_ok
        and not zero_vectors
        and not normalization_failures
        and not metadata_failures
    )
    warnings: list[str] = []
    if duplicate_vectors:
        warnings.append("Exact duplicate embedding vectors were found.")
    if duplicate_documents:
        warnings.append("Exact duplicate chunk texts were found.")
    if similarities and _summary(similarities)["mean"] is not None and _summary(similarities)["mean"] > 0.95:
        warnings.append("Sampled vectors are unusually similar on average; inspect chunk diversity.")

    return {
        "status": "passed" if passed else "needs_attention",
        "records_checked": len(records),
        "vector_dimensions": dict(dimensions),
        "invalid_vector_count": len(invalid_vectors),
        "zero_vector_count": len(zero_vectors),
        "normalization_tolerance": NORMALIZED_TOLERANCE,
        "norms": _summary(norms),
        "vectors_outside_normalization_tolerance": len(normalization_failures),
        "invalid_vector_examples": invalid_vectors[:20],
        "zero_vector_examples": zero_vectors[:20],
        "normalization_failure_examples": normalization_failures[:20],
        "metadata_failure_count": len(set(metadata_failures)),
        "metadata_failure_examples": list(dict.fromkeys(metadata_failures))[:20],
        "exact_duplicate_vector_groups": len(duplicate_vectors),
        "exact_duplicate_vector_examples": duplicate_vectors[:10],
        "exact_duplicate_text_groups": len(duplicate_documents),
        "exact_duplicate_text_examples": duplicate_documents[:10],
        "pairwise_sample_size": len(sampled),
        "pairwise_comparisons": len(similarities),
        "pairwise_cosine_similarity": _summary(similarities),
        "pairwise_cosine_distance": _summary(distances),
        "metric_note": "Cosine distance is defined as 1 - cosine similarity; lower distance means closer vectors.",
        "warnings": warnings,
    }


def evaluate(
    *,
    query: str,
    persist_dir: Path,
    collection_name: str,
    model_name: str,
    top_k: int,
    least_k: int,
    pair_sample_size: int,
    device: str | None,
) -> dict[str, Any]:
    if pair_sample_size < 2:
        raise ValueError("--pair-sample-size must be at least 2")
    collection, records = _load_collection_records(persist_dir, collection_name)
    if not records:
        raise ValueError(f"Collection {collection_name!r} is empty")
    ranking = rank_collection(
        query=query,
        persist_dir=persist_dir,
        collection_name=collection_name,
        model_name=model_name,
        sample_size=0,
        top_k=top_k,
        least_k=least_k,
        device=device,
    )
    return {
        "pipeline": "evaluate_text_embeddings",
        "read_only": True,
        "vector_database": display_path(persist_dir),
        "collection": collection_name,
        "collection_metadata": collection.metadata or {},
        "collection_record_count": collection.count(),
        "similarity_and_distance": ranking,
        "quality_checks": _quality_checks(records, pair_sample_size),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Question used to rank stored chunks")
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--least-k", type=int, default=5)
    parser.add_argument(
        "--pair-sample-size",
        type=int,
        default=DEFAULT_PAIR_SAMPLE_SIZE,
        help="Stored vectors sampled for pairwise similarity and distance checks (default: 128)",
    )
    parser.add_argument("--device", help="Optional Sentence Transformers device, for example cpu or cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = evaluate(
        query=args.query,
        persist_dir=args.persist_dir.resolve(),
        collection_name=args.collection,
        model_name=args.model,
        top_k=args.top_k,
        least_k=args.least_k,
        pair_sample_size=args.pair_sample_size,
        device=args.device,
    )
    report["output"] = display_path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    checks = report["quality_checks"]
    print(f"Collection: {report['collection']} ({report['collection_record_count']} records)")
    print(f"Quality checks: {checks['status']}")
    print(
        "Vectors: "
        f"invalid={checks['invalid_vector_count']}, zero={checks['zero_vector_count']}, "
        f"outside normalization tolerance={checks['vectors_outside_normalization_tolerance']}, "
        f"metadata failures={checks['metadata_failure_count']}"
    )
    pairwise = checks["pairwise_cosine_similarity"]
    print(
        f"Pairwise cosine similarity: mean={pairwise['mean']}, "
        f"min={pairwise['minimum']}, max={pairwise['maximum']}"
    )
    print(f"Saved evaluation: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
