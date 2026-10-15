"""Retrieval step: embed a user query and search the vector store for top-k chunks.

Usage examples
--------------
# k = 3
uv run python -m src.embeddings.retrieve \
    --query "What fire safety measures protect building exits?" \
    --top-k 3

# k = 7  (more context for the grounding step)
uv run python -m src.embeddings.retrieve \
    --query "What fire safety measures protect building exits?" \
    --top-k 7

# Run both k values in one shot and save combined results
uv run python -m src.embeddings.retrieve \
    --query "What fire safety measures protect building exits?" \
    --top-k 3 7

The script embeds the query with the same model used to embed the document
chunks (sentence-transformers/multi-qa-MiniLM-L6-dot-v1), searches the
persistent Chroma collection, and returns the top-k results with:
  - rank                  sequential position in the result list
  - id                    unique chunk identifier
  - cosine_similarity     1 - Chroma cosine distance (higher = more relevant)
  - cosine_distance       raw Chroma distance for reference
  - text                  source text of the chunk
  - metadata              all stored metadata (source document, chunk_index, …)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

# Reuse the defaults that were used when building the vector store
from src.embeddings.embed_text_chunks import (  # noqa: E402
    DEFAULT_COLLECTION,
    DEFAULT_MODEL,
    DEFAULT_PERSIST_DIR,
    _safe_collection_name,
)

DEFAULT_TOP_K: list[int] = [3, 7]
DEFAULT_SAMPLE_QUERY = "What fire safety measures are required to keep exits and escape routes safe?"
OUTPUT_DIR = ROOT / "outputs" / "retrieval"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core retrieval helpers
# ---------------------------------------------------------------------------


def _embed_query(
    query: str,
    *,
    model_name: str,
    device: str | None,
) -> list[float]:
    """Return a normalized embedding vector for *query*.

    Uses the identical model and normalization settings as ``embed_text_chunks``
    so that the cosine similarity is meaningful.
    """
    from sentence_transformers import SentenceTransformer  # local import keeps startup fast

    model_kwargs: dict[str, str] = {"device": device} if device else {}
    model = SentenceTransformer(model_name, **model_kwargs)
    logger.info("Embedding query with model: %s", model_name)
    vector = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    logger.info("Query embedding dimension: %d", len(vector))
    return vector.tolist()


def _search_collection(
    *,
    query_vector: list[float],
    collection: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    """Run a top-k ANN query on *collection* and return structured results."""
    effective_k = min(top_k, collection.count())
    if effective_k == 0:
        return []

    response = collection.query(
        query_embeddings=[query_vector],
        n_results=effective_k,
        include=["documents", "metadatas", "distances"],
    )

    ids: list[str] = (response.get("ids") or [[]])[0]
    documents: list[str] = (response.get("documents") or [[]])[0]
    metadatas: list[dict] = (response.get("metadatas") or [[]])[0]
    distances: list[float] = (response.get("distances") or [[]])[0]

    results: list[dict[str, Any]] = []
    for rank, chunk_id in enumerate(ids, start=1):
        distance = float(distances[rank - 1]) if (rank - 1) < len(distances) else None
        similarity = (1.0 - distance) if distance is not None else None
        results.append(
            {
                "rank": rank,
                "id": chunk_id,
                "cosine_similarity": similarity,
                "cosine_distance": distance,
                "text": documents[rank - 1] if (rank - 1) < len(documents) else None,
                "metadata": metadatas[rank - 1] if (rank - 1) < len(metadatas) else {},
            }
        )
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrieve(
    *,
    query: str,
    top_k: int,
    persist_dir: Path,
    collection_name: str,
    model_name: str,
    device: str | None,
) -> dict[str, Any]:
    """Embed *query*, search the vector store, and return the top-*k* results.

    Returns a dict with keys:
      query, top_k, embedding_model, collection, retrieved_at,
      total_collection_records, results
    Each result contains rank, id, cosine_similarity, cosine_distance, text,
    metadata.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    import chromadb  # local import

    safe_name = _safe_collection_name(collection_name)
    client = chromadb.PersistentClient(path=str(persist_dir))

    try:
        collection = client.get_collection(name=safe_name)
    except Exception as exc:
        raise ValueError(
            f"Collection {safe_name!r} not found in {persist_dir}. "
            "Run the embedding pipeline first."
        ) from exc

    stored_model = (collection.metadata or {}).get("embedding_model")
    if stored_model and stored_model != model_name:
        raise ValueError(
            f"Collection {safe_name!r} was built with model {stored_model!r} "
            f"but retrieval requested {model_name!r}. "
            "Use the same model or choose a different collection."
        )

    count = collection.count()
    if count == 0:
        raise ValueError(f"Collection {safe_name!r} is empty — nothing to search.")

    logger.info(
        "Searching collection '%s' (%d records) with top_k=%d",
        safe_name,
        count,
        top_k,
    )

    query_vector = _embed_query(query, model_name=model_name, device=device)
    results = _search_collection(
        query_vector=query_vector,
        collection=collection,
        top_k=top_k,
    )

    return {
        "query": query,
        "top_k": top_k,
        "embedding_model": model_name,
        "collection": safe_name,
        "vector_database": str(persist_dir),
        "retrieved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_collection_records": count,
        "results": results,
        "score_note": (
            "cosine_similarity = 1 - Chroma cosine distance. "
            "Scores range from -1 (opposite) to 1 (identical). "
            "Higher values mean greater semantic similarity to the query."
        ),
    }


def retrieve_multi_k(
    *,
    query: str,
    k_values: list[int],
    persist_dir: Path,
    collection_name: str,
    model_name: str,
    device: str | None,
) -> dict[str, Any]:
    """Run retrieval for several k values and return a comparison report.

    The query is embedded **once** and reused for all k lookups so that
    differences in the result sets are purely due to k, not embedding variance.
    """
    if not k_values:
        raise ValueError("k_values must contain at least one value")

    if not query.strip():
        raise ValueError("Query cannot be empty")

    import chromadb

    safe_name = _safe_collection_name(collection_name)
    client = chromadb.PersistentClient(path=str(persist_dir))

    try:
        collection = client.get_collection(name=safe_name)
    except Exception as exc:
        raise ValueError(
            f"Collection {safe_name!r} not found in {persist_dir}."
        ) from exc

    count = collection.count()
    logger.info("Collection '%s' has %d records.", safe_name, count)

    query_vector = _embed_query(query, model_name=model_name, device=device)

    runs: list[dict[str, Any]] = []
    for k in sorted(set(k_values)):
        logger.info("Running top-k=%d search …", k)
        results = _search_collection(
            query_vector=query_vector,
            collection=collection,
            top_k=k,
        )
        runs.append(
            {
                "k": k,
                "results_returned": len(results),
                "results": results,
            }
        )

    return {
        "query": query,
        "embedding_model": model_name,
        "collection": safe_name,
        "vector_database": str(persist_dir),
        "retrieved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_collection_records": count,
        "k_values_tested": sorted(set(k_values)),
        "runs": runs,
        "score_note": (
            "cosine_similarity = 1 - Chroma cosine distance. "
            "Higher values indicate greater semantic relevance. "
            "All runs use the same query vector so score differences are solely "
            "due to the changed k cutoff, not embedding variance."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )


def _print_run(run: dict[str, Any], query: str) -> None:
    k = run["k"]
    print(f"\n{'='*70}")
    print(f"  top-k = {k}   (returned {run['results_returned']} results)")
    print(f"  query : {query}")
    print(f"{'='*70}")
    for item in run["results"]:
        meta = item.get("metadata") or {}
        source = meta.get("source", meta.get("chunk_id", item["id"]))
        chunk_idx = meta.get("chunk_index", "?")
        page = meta.get("page_start", meta.get("page", "?"))
        sim = item["cosine_similarity"]
        dist = item["cosine_distance"]
        print(
            f"\n#{item['rank']:>2}  id={item['id']}"
            f"\n     similarity={sim:.6f}  distance={dist:.6f}"
            f"\n     source={source}  chunk_index={chunk_idx}  page_start={page}"
            f"\n     text: {(item['text'] or '')[:400].replace(chr(10), ' ')}"
        )


def main() -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_SAMPLE_QUERY,
        help="User query to embed and search (default: fire-safety sample)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=DEFAULT_TOP_K,
        metavar="K",
        help="One or more k values to test (default: 3 7)",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_PERSIST_DIR,
        help="Chroma persistence directory",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Chroma collection name",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Sentence Transformers model (must match the collection)",
    )
    parser.add_argument("--device", help="Optional torch device, e.g. cpu or cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where JSON results are written",
    )
    args = parser.parse_args()

    k_values: list[int] = args.top_k
    if any(k < 1 for k in k_values):
        logger.error("All k values must be at least 1.")
        return 1

    try:
        report = retrieve_multi_k(
            query=args.query,
            k_values=k_values,
            persist_dir=args.persist_dir.resolve(),
            collection_name=args.collection,
            model_name=args.model,
            device=args.device,
        )
    except Exception as exc:
        logger.exception("Retrieval failed: %s", exc)
        return 1

    # Print human-readable summary
    print(f"\nQuery   : {report['query']}")
    print(f"Model   : {report['embedding_model']}")
    print(f"Store   : {report['vector_database']}")
    print(f"Records : {report['total_collection_records']}")
    for run in report["runs"]:
        _print_run(run, report["query"])

    # Persist full JSON report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ks_tag = "_".join(str(k) for k in report["k_values_tested"])
    out_path = args.output_dir / f"retrieval_k{ks_tag}.json"
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Results saved → %s", out_path.resolve())
    print(f"\nSaved   : {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
