"""Rank stored chunk embeddings against a text query with cosine similarity.

Chunk vectors are read from Chroma. Only the query is encoded at search time,
using the same Sentence Transformers model recorded on the collection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .embed_text_chunks import (
    DEFAULT_COLLECTION,
    DEFAULT_MODEL,
    DEFAULT_PERSIST_DIR,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "examples" / "embedding_similarity" / "sample_rankings.json"
DEFAULT_QUERY = "What fire safety measures are required to keep exits and escape routes safe?"
DEFAULT_SAMPLE_SIZE = 0


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for two nonzero vectors of equal dimension."""
    if len(left) != len(right):
        raise ValueError("Cannot compare embedding vectors with different dimensions")
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Cannot calculate cosine similarity for a zero vector")
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    return dot / (left_norm * right_norm)


def evenly_sample_indices(record_count: int, sample_size: int) -> list[int]:
    """Select positions at regular intervals, preserving page-order coverage."""
    if sample_size < 0:
        raise ValueError("Sample size cannot be negative")
    if sample_size == 0 or sample_size >= record_count:
        return list(range(record_count))
    if sample_size == 1:
        return [record_count // 2]
    last = record_count - 1
    return [round(index * last / (sample_size - 1)) for index in range(sample_size)]


def display_path(path: Path) -> str:
    """Prefer portable repository-relative paths in saved reports."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def rank_collection(
    *,
    query: str,
    persist_dir: Path,
    collection_name: str,
    model_name: str,
    sample_size: int,
    top_k: int,
    least_k: int,
    device: str | None,
) -> dict[str, Any]:
    """Compare a query vector with stored vectors in a Chroma collection."""
    if not query.strip():
        raise ValueError("Query cannot be empty")
    if sample_size < 0 or top_k < 1 or least_k < 1:
        raise ValueError("Sample size cannot be negative; top-k and least-k must be at least 1")

    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer

    client = PersistentClient(path=str(persist_dir))
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as exc:
        raise ValueError(
            f"Collection {collection_name!r} was not found in {persist_dir}. "
            "Run the text embedding command first."
        ) from exc

    stored_model = (collection.metadata or {}).get("embedding_model")
    if stored_model and stored_model != model_name:
        raise ValueError(
            f"Collection {collection_name!r} uses model {stored_model!r}, "
            f"but query requested {model_name!r}"
        )

    collection_count = collection.count()
    if collection_count == 0:
        raise ValueError(f"Collection {collection_name!r} is empty")
    stored = collection.get(include=["documents", "metadatas", "embeddings"])
    documents = stored.get("documents")
    metadatas = stored.get("metadatas")
    vectors = stored.get("embeddings")
    if documents is None or metadatas is None or vectors is None:
        raise ValueError("The collection must contain stored documents, metadata, and embeddings")
    entries = []
    for index, record_id in enumerate(stored["ids"]):
        document = documents[index]
        metadata = metadatas[index] or {}
        vector = vectors[index]
        if document is None or vector is None:
            continue
        order = (
            metadata.get("volume", 0),
            metadata.get("page_start", 0),
            metadata.get("chunk_index", 0),
            record_id,
        )
        entries.append((order, record_id, document, metadata, vector))
    entries.sort(key=lambda item: item[0])
    if not entries:
        raise ValueError(f"Collection {collection_name!r} has no records with text and vectors")

    selected = evenly_sample_indices(len(entries), sample_size)
    sampled = [entries[index] for index in selected]

    model_kwargs = {"device": device} if device else {}
    model = SentenceTransformer(model_name, **model_kwargs)
    query_vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    ranked = []
    for _, record_id, document, metadata, vector in sampled:
        ranked.append(
            {
                "id": record_id,
                "score": cosine_similarity(query_vector, vector),
                "text": document,
                "metadata": metadata,
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["id"]))
    return {
        "query": query,
        "embedding_model": model_name,
        "vector_database": display_path(persist_dir),
        "collection": collection_name,
        "vector_source": "Embeddings stored in the Chroma collection; only the query was newly encoded.",
        "metric": "cosine_similarity",
        "metric_justification": (
            "Cosine similarity compares the direction of the query and chunk vectors. "
            "The model outputs normalized vectors, so their dot product is cosine similarity; "
            "higher scores indicate greater semantic similarity."
        ),
        "collection_record_count": collection_count,
        "sampled_record_count": len(sampled),
        "sampling": (
            "All records in volume/page/chunk order from the stored collection."
            if sample_size == 0 or sample_size >= len(entries)
            else "Evenly spaced records in volume/page/chunk order from the stored collection."
        ),
        "most_similar": ranked[: min(top_k, len(ranked))],
        "least_similar": list(reversed(ranked[-min(least_k, len(ranked)) :])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Text query to compare with stored chunks")
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of evenly spaced records to compare; 0 (default) compares the full collection",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--least-k", type=int, default=5)
    parser.add_argument("--device", help="Optional model device, for example cpu or cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSON report path")
    args = parser.parse_args()

    report = rank_collection(
        query=args.query,
        persist_dir=args.persist_dir.resolve(),
        collection_name=args.collection,
        model_name=args.model,
        sample_size=args.sample_size,
        top_k=args.top_k,
        least_k=args.least_k,
        device=args.device,
    )
    report["output"] = display_path(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Query: {report['query']}")
    print(
        f"Metric: cosine similarity; ranked {report['sampled_record_count']} of "
        f"{report['collection_record_count']} stored chunks."
    )
    for label in ("most_similar", "least_similar"):
        print(f"\n{label.replace('_', ' ').title()}")
        for item in report[label]:
            metadata = item["metadata"]
            print(
                f"{item['score']:+.4f}  {item['id']}  "
                f"volume={metadata.get('volume')} pages="
                f"{metadata.get('page_start')}-{metadata.get('page_end')}"
            )
            print(f"  {item['text'][:350].replace(chr(10), ' ')}")
    print(f"\nSaved rankings: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
