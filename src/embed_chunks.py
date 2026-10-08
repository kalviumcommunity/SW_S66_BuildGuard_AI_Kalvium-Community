"""Embed prepared token chunks locally with Hugging Face embeddings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    from document_loader import load_corpus
    from text_cleaner import clean_document
    from token_chunking import TokenChunk, token_chunk_documents
except ImportError:  # Support imports through the src namespace in tests.
    from .document_loader import load_corpus
    from .text_cleaner import clean_document
    from .token_chunking import TokenChunk, token_chunk_documents

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = PROJECT_ROOT / "examples" / "sample_corpus"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "embeddings"
OUTPUT_JSON = OUTPUT_DIR / "stored_chunk_embeddings.json"
SUMMARY_PATH = OUTPUT_DIR / "chunk_embedding_summary.txt"
DEFAULT_MODEL = "sentence-transformers/multi-qa-MiniLM-L6-dot-v1"


def prepare_chunks(corpus_path: str | Path = CORPUS_PATH) -> list[TokenChunk]:
    """Load, clean, and token-chunk the prepared corpus."""
    loaded = load_corpus(corpus_path)
    cleaned = [clean_document(document) for document in loaded]
    return token_chunk_documents(cleaned)


def validate_vector_dimensions(vectors: list[list[float]], expected_dimension: int | None = None) -> int:
    """Validate nonempty, equal-length vectors and return their dimension."""
    if not vectors:
        raise ValueError("no embedding vectors were returned")
    dimension = len(vectors[0])
    if dimension == 0 or any(len(vector) != dimension for vector in vectors):
        raise ValueError("embedding vectors do not all have the same non-zero dimension")
    if expected_dimension is not None and dimension != expected_dimension:
        raise ValueError(
            f"embedding dimension {dimension} does not match expected {expected_dimension}"
        )
    return dimension


def build_embedding_records(
    chunks: list[TokenChunk],
    vectors: list[list[float]],
    expected_dimension: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Pair full vectors with their source text and scalar chunk metadata."""
    if len(chunks) != len(vectors):
        raise ValueError("the number of vectors does not match the number of chunks")
    dimension = validate_vector_dimensions(vectors, expected_dimension)
    records = [
        {
            "embedding": list(vector),
            "vector_preview": list(vector[:5]),
            "vector_length": len(vector),
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
        for chunk, vector in zip(chunks, vectors)
    ]
    return records, dimension


def format_summary(model_name: str, records: list[dict[str, Any]], dimension: int) -> str:
    """Create a readable verification summary."""
    sample_values = records[0]["vector_preview"] if records else []
    return (
        "Local chunk embedding verification\n"
        f"Model: {model_name}\n"
        f"Chunks embedded: {len(records)}\n"
        f"Vector dimension: {dimension}\n"
        f"Sample vector values: {sample_values}\n"
        "Provider: LangChain HuggingFaceEmbeddings (local; no API call)\n"
    )


def create_embedding_model(model_name: str) -> Any:
    """Create the local Hugging Face embedding model lazily."""
    huggingface = __import__("langchain_huggingface", fromlist=["HuggingFaceEmbeddings"])
    return huggingface.HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": False},
    )


def run_embedding_pipeline(
    embedding_model: Any,
    chunks: list[TokenChunk],
    model_name: str,
    expected_dimension: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Embed injected model inputs and return storage-ready records."""
    vectors = [list(vector) for vector in embedding_model.embed_documents([chunk.text for chunk in chunks])]
    return build_embedding_records(chunks, vectors, expected_dimension)


def main() -> int:
    load_dotenv()
    model_name = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    try:
        chunks = prepare_chunks()
        embedding_model = create_embedding_model(model_name)
        records, dimension = run_embedding_pipeline(embedding_model, chunks, model_name)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_JSON.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        SUMMARY_PATH.write_text(format_summary(model_name, records, dimension), encoding="utf-8")
    except Exception as error:  # noqa: BLE001 - model/runtime failures need a clear CLI result
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(f"Local chunk embedding failed\nError: {error}\n", encoding="utf-8")
        print(f"Local chunk embedding failed: {error}")
        return 1

    print(f"Chunks embedded: {len(records)}")
    print(f"Vector dimension: {dimension}")
    print(f"Sample vector values: {records[0]['vector_preview'] if records else []}")
    print(f"Stored embeddings: {OUTPUT_JSON}")
    print(f"Summary: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
