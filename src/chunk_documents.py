"""Chunk cleaned corpus documents with reusable LangChain splitters."""

from dataclasses import dataclass
from typing import Literal

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

try:
    from document_loader import LoadedDocument
except ImportError:  # Support imports through the src namespace in tests.
    from .document_loader import LoadedDocument

RECURSIVE_CHUNK_SIZE = 500
RECURSIVE_CHUNK_OVERLAP = 75
RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
FIXED_CHUNK_SIZE = 500
FIXED_CHUNK_OVERLAP = 0
ChunkStrategy = Literal["recursive", "fixed"]


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    chunk_id: int
    text: str


def recursive_splitter() -> RecursiveCharacterTextSplitter:
    """Create the boundary-aware recursive splitter configuration."""
    return RecursiveCharacterTextSplitter(
        chunk_size=RECURSIVE_CHUNK_SIZE,
        chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
        separators=RECURSIVE_SEPARATORS,
    )


def fixed_splitter() -> CharacterTextSplitter:
    """Create the fixed 500-character, zero-overlap splitter configuration."""
    return CharacterTextSplitter(
        separator=" ",
        chunk_size=FIXED_CHUNK_SIZE,
        chunk_overlap=FIXED_CHUNK_OVERLAP,
    )


def _splitter(strategy: ChunkStrategy):
    if strategy == "recursive":
        return recursive_splitter()
    if strategy == "fixed":
        return fixed_splitter()
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def chunk_documents(
    documents: list[LoadedDocument],
    strategy: ChunkStrategy = "recursive",
) -> list[DocumentChunk]:
    """Chunk documents and assign source-local sequential IDs."""
    splitter = _splitter(strategy)
    chunks: list[DocumentChunk] = []
    for document in documents:
        chunk_id = 0
        for text in splitter.split_text(document.text):
            if text.strip():
                chunks.append(DocumentChunk(document.source, chunk_id, text))
                chunk_id += 1
    return chunks


def recursive_chunks(documents: list[LoadedDocument]) -> list[DocumentChunk]:
    return chunk_documents(documents, strategy="recursive")


def fixed_chunks(documents: list[LoadedDocument]) -> list[DocumentChunk]:
    return chunk_documents(documents, strategy="fixed")


def chunk_stats(chunks: list[DocumentChunk]) -> dict[str, int | float]:
    """Return count and character-length statistics for chunks."""
    lengths = [len(chunk.text) for chunk in chunks]
    if not lengths:
        return {"count": 0, "average_length": 0.0, "min_length": 0, "max_length": 0}
    return {
        "count": len(lengths),
        "average_length": sum(lengths) / len(lengths),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }
