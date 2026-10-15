"""Search the unified NBC text and image collections with one text query.

Both Volume 1 and Volume 2 are searched together. Text and image vectors use
different models and dimensions, so they remain in separate Chroma collections
but are returned in one result list. Similarity scores are meaningful within a
modality; the displayed combined order alternates their per-modality ranks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .embed_image_chunks import (
    DEFAULT_COLLECTION as DEFAULT_IMAGE_COLLECTION,
    DEFAULT_MODEL as DEFAULT_IMAGE_MODEL,
)
from .embed_text_chunks import (
    DEFAULT_COLLECTION as DEFAULT_TEXT_COLLECTION,
    DEFAULT_MODEL as DEFAULT_TEXT_MODEL,
    DEFAULT_PERSIST_DIR,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs" / "search" / "latest_nbc_search.json"


def _query_collection(
    *,
    collection: Any,
    query_vector: list[float],
    modality: str,
    top_k: int,
) -> list[dict[str, Any]]:
    response = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    ids = (response.get("ids") or [[]])[0]
    documents = (response.get("documents") or [[]])[0]
    metadatas = (response.get("metadatas") or [[]])[0]
    distances = (response.get("distances") or [[]])[0]
    results: list[dict[str, Any]] = []
    for rank, record_id in enumerate(ids, start=1):
        distance = float(distances[rank - 1]) if rank - 1 < len(distances) else None
        results.append(
            {
                "modality": modality,
                "modality_rank": rank,
                "id": record_id,
                "cosine_distance": distance,
                "cosine_similarity": None if distance is None else 1.0 - distance,
                "text": documents[rank - 1] if rank - 1 < len(documents) else None,
                "metadata": metadatas[rank - 1] if rank - 1 < len(metadatas) else {},
            }
        )
    return results


def _interleave(text_results: list[dict[str, Any]], image_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one list while avoiding a misleading comparison across models."""
    combined: list[dict[str, Any]] = []
    for index in range(max(len(text_results), len(image_results))):
        if index < len(text_results):
            combined.append(text_results[index])
        if index < len(image_results):
            combined.append(image_results[index])
    for rank, item in enumerate(combined, start=1):
        item["combined_rank"] = rank
    return combined


def search(
    *,
    query: str,
    persist_dir: Path,
    text_collection_name: str,
    image_collection_name: str,
    text_model_name: str,
    image_model_name: str,
    top_k: int,
    device: str | None,
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("Query cannot be empty")
    if top_k < 1:
        raise ValueError("--top-k must be at least 1")

    import chromadb
    from sentence_transformers import SentenceTransformer

    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        text_collection = client.get_collection(text_collection_name)
        image_collection = client.get_collection(image_collection_name)
    except Exception as exc:
        raise ValueError(
            "Both text and image collections must exist. Run text and image embedding first."
        ) from exc
    if text_collection.count() == 0 or image_collection.count() == 0:
        raise ValueError("Both text and image collections must contain at least one record.")

    model_kwargs = {"device": device} if device else {}
    text_model = SentenceTransformer(text_model_name, **model_kwargs)
    image_model = SentenceTransformer(image_model_name, **model_kwargs)
    text_vector = text_model.encode(query, normalize_embeddings=True, convert_to_numpy=True).tolist()
    # CLIP maps text and images into the same image-model vector space.
    image_query_vector = image_model.encode(query, normalize_embeddings=True, convert_to_numpy=True).tolist()

    text_results = _query_collection(
        collection=text_collection,
        query_vector=text_vector,
        modality="text",
        top_k=top_k,
    )
    image_results = _query_collection(
        collection=image_collection,
        query_vector=image_query_vector,
        modality="image",
        top_k=top_k,
    )
    return {
        "query": query,
        "vector_database": str(persist_dir),
        "searched_volumes": ["volume_1", "volume_2"],
        "text_collection": {
            "name": text_collection_name,
            "records": text_collection.count(),
            "model": text_model_name,
            "results": text_results,
        },
        "image_collection": {
            "name": image_collection_name,
            "records": image_collection.count(),
            "model": image_model_name,
            "results": image_results,
        },
        "combined_results": _interleave(text_results, image_results),
        "score_note": (
            "Cosine similarity is 1 - Chroma cosine distance. Text and image scores use "
            "different models, so compare scores within each modality. Combined results alternate "
            "the text and image ranks."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Question or phrase to search in both NBC collections")
    parser.add_argument("--persist-dir", type=Path, default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--text-collection", default=DEFAULT_TEXT_COLLECTION)
    parser.add_argument("--image-collection", default=DEFAULT_IMAGE_COLLECTION)
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL)
    parser.add_argument("--top-k", type=int, default=5, help="Results to retrieve per modality")
    parser.add_argument("--device", help="Optional Sentence Transformers device, for example cpu or cuda")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = search(
        query=args.query,
        persist_dir=args.persist_dir.resolve(),
        text_collection_name=args.text_collection,
        image_collection_name=args.image_collection,
        text_model_name=args.text_model,
        image_model_name=args.image_model,
        top_k=args.top_k,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Searched both volumes: {report['query']}")
    for item in report["combined_results"]:
        metadata = item["metadata"] or {}
        page = metadata.get("page_start", metadata.get("page_number", metadata.get("page")))
        print(
            f"{item['combined_rank']:>2}. {item['modality']:<5} "
            f"rank {item['modality_rank']} | similarity={item['cosine_similarity']:.4f} "
            f"| volume={metadata.get('volume')} | page={page} | {item['id']}"
        )
    print(f"Saved results: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
