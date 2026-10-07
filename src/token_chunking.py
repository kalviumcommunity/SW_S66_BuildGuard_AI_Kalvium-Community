"""Token-aware chunking with controlled token overlap."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import tiktoken

try:
    from document_loader import LoadedDocument
except ImportError:  # Support imports through the src namespace in tests.
    from .document_loader import LoadedDocument

TOKEN_CHUNK_SIZE = 256
TOKEN_OVERLAP = 32
DEFAULT_ENCODING = "cl100k_base"
_HEADING = re.compile(r"(?m)^#{1,3}\s+(.+?)\s*$")


@dataclass(frozen=True)
class TokenChunk:
    source: str
    section: str
    page: int
    chunk_index: int
    start_token: int
    end_token: int
    token_count: int
    text: str
    token_ids: tuple[int, ...]

    @property
    def chunk_id(self) -> int:
        return self.chunk_index

    @property
    def page_content(self) -> str:
        return self.text

    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "source": self.source,
            "section": self.section,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "start_token": self.start_token,
            "end_token": self.end_token,
            "token_count": self.token_count,
        }


def get_tokenizer(model_name: str | None = None) -> tuple[Any, str]:
    """Select a model encoding, falling back to cl100k_base for unknown models."""
    requested = (model_name or DEFAULT_ENCODING).strip()
    try:
        return tiktoken.encoding_for_model(requested), requested
    except (KeyError, ValueError):
        return tiktoken.get_encoding(DEFAULT_ENCODING), DEFAULT_ENCODING


def _section_at(text: str, character_position: int) -> str:
    headings = [(match.start(), match.group(1).strip()) for match in _HEADING.finditer(text)]
    for position, section in reversed(headings):
        if position <= character_position:
            return section
    return "General"


def token_chunk_documents(
    documents: list[LoadedDocument],
    *,
    model_name: str | None = None,
    chunk_size: int = TOKEN_CHUNK_SIZE,
    overlap: int = TOKEN_OVERLAP,
) -> list[TokenChunk]:
    """Split documents into token windows with exact adjacent-token overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    encoding, _ = get_tokenizer(model_name)
    chunks: list[TokenChunk] = []
    for document in documents:
        token_ids = encoding.encode(document.text)
        start = 0
        chunk_index = 0
        while start < len(token_ids):
            end = min(start + chunk_size, len(token_ids))
            window = token_ids[start:end]
            if not window:
                break
            text = encoding.decode(window)
            start_char = len(encoding.decode(token_ids[:start]))
            chunks.append(
                TokenChunk(
                    source=document.source,
                    section=_section_at(document.text, start_char),
                    page=1,
                    chunk_index=chunk_index,
                    start_token=start,
                    end_token=end,
                    token_count=len(window),
                    text=text,
                    token_ids=tuple(window),
                )
            )
            chunk_index += 1
            if end == len(token_ids):
                break
            start = end - overlap
    return chunks


def token_chunk_stats(chunks: list[TokenChunk]) -> dict[str, int | float]:
    """Return token-count statistics for token chunks."""
    counts = [chunk.token_count for chunk in chunks]
    if not counts:
        return {"count": 0, "average_tokens": 0.0, "min_tokens": 0, "max_tokens": 0}
    return {
        "count": len(counts),
        "average_tokens": sum(counts) / len(counts),
        "min_tokens": min(counts),
        "max_tokens": max(counts),
    }


# Explicit aliases make the reusable API easy to discover alongside character chunking.
chunk_documents = token_chunk_documents
chunk_stats = token_chunk_stats
