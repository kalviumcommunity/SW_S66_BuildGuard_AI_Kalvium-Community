"""Chunk cleaned corpus documents with metadata-aware LangChain splitters."""

import re
from dataclasses import dataclass
from typing import Literal

from langchain_core.documents import Document
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
DEFAULT_PAGE = 1  # TXT, Markdown, and HTML corpus documents are non-paginated.
ChunkStrategy = Literal["recursive", "fixed"]
_HEADING = re.compile(r"(?m)^#{1,3}\s+(.+?)\s*$")
_METADATA_KEYS = (
    "source",
    "section",
    "page",
    "chunk_index",
    "start_char",
    "end_char",
)


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    section: str
    page: int
    chunk_index: int
    start_char: int
    end_char: int
    text: str

    @property
    def chunk_id(self) -> int:
        """Backward-compatible alias for the source-local chunk index."""
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
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


def recursive_splitter() -> RecursiveCharacterTextSplitter:
    """Create the boundary-aware recursive splitter configuration."""
    return RecursiveCharacterTextSplitter(
        chunk_size=RECURSIVE_CHUNK_SIZE,
        chunk_overlap=RECURSIVE_CHUNK_OVERLAP,
        separators=RECURSIVE_SEPARATORS,
        add_start_index=True,
    )


def fixed_splitter() -> CharacterTextSplitter:
    """Create the fixed 500-character, zero-overlap splitter configuration."""
    return CharacterTextSplitter(
        separator=" ",
        chunk_size=FIXED_CHUNK_SIZE,
        chunk_overlap=FIXED_CHUNK_OVERLAP,
        add_start_index=True,
    )


def _splitter(strategy: ChunkStrategy):
    if strategy == "recursive":
        return recursive_splitter()
    if strategy == "fixed":
        return fixed_splitter()
    raise ValueError(f"Unknown chunking strategy: {strategy}")


def _section_at(text: str, start_char: int) -> str:
    sections = [(match.start(), match.group(1).strip()) for match in _HEADING.finditer(text)]
    for position, section in reversed(sections):
        if position <= start_char:
            return section
    return "General"


def chunk_documents(
    documents: list[LoadedDocument],
    strategy: ChunkStrategy = "recursive",
) -> list[DocumentChunk]:
    """Chunk documents with source-local IDs and source-text positions."""
    splitter = _splitter(strategy)
    chunks: list[DocumentChunk] = []
    for document in documents:
        source_index = 0
        split_documents = splitter.create_documents([document.text])
        for split_document in split_documents:
            text = split_document.page_content
            if not text.strip():
                continue
            start_char = int(split_document.metadata.get("start_index", 0))
            end_char = start_char + len(text)
            chunks.append(
                DocumentChunk(
                    source=document.source,
                    section=_section_at(document.text, start_char),
                    page=DEFAULT_PAGE,
                    chunk_index=source_index,
                    start_char=start_char,
                    end_char=end_char,
                    text=text,
                )
            )
            source_index += 1
    return chunks


def recursive_chunks(documents: list[LoadedDocument]) -> list[DocumentChunk]:
    return chunk_documents(documents, strategy="recursive")


def fixed_chunks(documents: list[LoadedDocument]) -> list[DocumentChunk]:
    return chunk_documents(documents, strategy="fixed")


def to_langchain_documents(chunks: list[DocumentChunk]) -> list[Document]:
    """Convert chunks to Chroma-compatible LangChain Documents."""
    return [Document(page_content=chunk.text, metadata=chunk.metadata) for chunk in chunks]


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
