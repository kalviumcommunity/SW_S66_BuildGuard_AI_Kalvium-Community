"""Demonstrate local semantic embeddings with Hugging Face."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "embeddings"
OUTPUT_JSON = OUTPUT_DIR / "embedding_demo_output.json"
EXPLANATION_PATH = OUTPUT_DIR / "embedding_explanation.txt"
MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-dot-v1"
EXPECTED_VECTOR_DIMENSION = 384
SAMPLE_TEXTS = [
    "How do I request leave for several consecutive working days?",
    "What is the process for submitting a multi-day leave request?",
    "The cafeteria menu is updated every Thursday afternoon.",
]


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    """Return cosine similarity for deterministic helper/test use."""
    if len(first) != len(second):
        raise ValueError("cosine similarity requires vectors with equal dimensions")
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")
    return sum(a * b for a, b in zip(first, second)) / (first_norm * second_norm)


def dot_product(first: Sequence[float], second: Sequence[float]) -> float:
    """Return the recommended unnormalized dot-product similarity."""
    if len(first) != len(second):
        raise ValueError("dot product requires vectors with equal dimensions")
    return sum(a * b for a, b in zip(first, second))


def validate_embedding_dimensions(embeddings: Sequence[Sequence[float]]) -> tuple[bool, int]:
    if not embeddings:
        raise ValueError("embedding response contained no vectors")
    dimension = len(embeddings[0])
    if dimension == 0 or any(len(vector) != dimension for vector in embeddings):
        raise ValueError("embedding vectors do not all have the same non-zero dimension")
    return True, dimension


def compare_embeddings(embeddings: Sequence[Sequence[float]]) -> dict[str, float | bool | str]:
    """Compare the similar leave pair with the unrelated cafeteria pair by dot product."""
    if len(embeddings) < 3:
        raise ValueError("at least three embeddings are required for this comparison")
    similar_score = dot_product(embeddings[0], embeddings[1])
    unrelated_score = dot_product(embeddings[0], embeddings[2])
    return {
        "similarity_similar_leave_pair": similar_score,
        "similarity_unrelated_pair": unrelated_score,
        "similarity_metric": "dot_product",
        "similar_score_higher": similar_score > unrelated_score,
    }


def embedding_result(
    model: str,
    texts: Sequence[str],
    embeddings: Sequence[Sequence[float]],
) -> dict[str, Any]:
    same_length, dimension = validate_embedding_dimensions(embeddings)
    comparison = compare_embeddings(embeddings)
    return {
        "status": "success",
        "model": model,
        "expected_vector_dimension": EXPECTED_VECTOR_DIMENSION,
        "vector_dimension": dimension,
        "same_length": same_length,
        "texts": [
            {
                "id": index,
                "text": text,
                "vector_length": len(embeddings[index]),
                "vector_preview": list(embeddings[index][:5]),
            }
            for index, text in enumerate(texts)
        ],
        **comparison,
        "comparison_result": "PASS" if comparison["similar_score_higher"] else "FAIL",
    }


def create_embedding_model() -> Any:
    """Create the local model lazily so importing this module never downloads it."""
    huggingface = import_module("langchain_huggingface")
    return huggingface.HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        encode_kwargs={"normalize_embeddings": False},
    )


def write_explanation() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPLANATION_PATH.write_text(
        "Embedding explanation\n\n"
        f"This demo uses the local Hugging Face model {MODEL_NAME}. It produces "
        f"{EXPECTED_VECTOR_DIMENSION}-dimensional vectors without an API key or network API request. "
        "The comparison uses the model's recommended unnormalized dot-product similarity. "
        "Embedding vectors represent semantic meaning learned from language; they are not "
        "random IDs and are not keyword counts.\n",
        encoding="utf-8",
    )


def write_error_artifact(error: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps({"status": "error", "model": MODEL_NAME, "error": error}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_explanation()


def run_embedding_demo(
    embedding_model: Any,
    texts: Sequence[str] = SAMPLE_TEXTS,
    model_name: str = MODEL_NAME,
    *,
    write_outputs: bool = True,
) -> dict[str, Any]:
    """Embed texts with an injected model, allowing deterministic tests."""
    embeddings = embedding_model.embed_documents(list(texts))
    result = embedding_result(model_name, texts, embeddings)
    if write_outputs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_explanation()
    return result


def main() -> int:
    try:
        result = run_embedding_demo(create_embedding_model())
    except Exception as error:  # noqa: BLE001 - model download/runtime errors need a clear artifact
        write_error_artifact(str(error))
        print(f"Local embedding demo unavailable: {error}")
        return 1
    print(f"Model: {result['model']}")
    print(f"Vector dimension: {result['vector_dimension']} (expected {EXPECTED_VECTOR_DIMENSION})")
    print(f"Same vector length: {result['same_length']}")
    print(f"Dot product, similar leave pair: {result['similarity_similar_leave_pair']:.6f}")
    print(f"Dot product, unrelated pair: {result['similarity_unrelated_pair']:.6f}")
    print(f"Comparison result: {result['comparison_result']}")
    return 0 if result["similar_score_higher"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
