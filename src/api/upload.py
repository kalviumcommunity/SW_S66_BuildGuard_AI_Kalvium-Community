"""Runtime document ingestion: store, extract, clean, chunk, embed, and index.

This module lets the knowledge base grow while the API is running. An uploaded
document is validated, stored safely under the configured upload directory, its
text extracted and cleaned, split into token-aware chunks, embedded with the
SAME model used to index the original corpus, and upserted into the SAME
ChromaDB collection the query endpoint searches — so the new content is
immediately retrievable without restarting the app.

The embedding convention (L2-normalized vectors, cosine space) matches
``src.embeddings.embed_text_chunks`` and ``retrieve._embed_query`` so the
existing hybrid retrieval and re-ranking stages work unchanged.

Tasks addressed
---------------
  Task 1 — ``store_upload`` saves the accepted file safely.
  Task 2 — ``index_document`` runs ingestion → cleaning → chunking → embedding
           → vector-DB indexing end to end.
  Task 4 — ``UploadError`` subclasses carry an HTTP status + message for
           unsupported formats, empty files, oversized files, and failures.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Chunks target the same 500-token ceiling used by the offline pipeline.
MAX_TOKENS = 500
EMBEDDING_BATCH_SIZE = 64

_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TAG = re.compile(r"<[^>]+>")

_ENCODING: Any = None


def _encoding():
    """Lazily load the tiktoken encoding (avoids cost at import time)."""
    global _ENCODING
    if _ENCODING is None:
        import tiktoken

        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


# ---------------------------------------------------------------------------
# Errors — each maps to a clear HTTP status code (Task 4)
# ---------------------------------------------------------------------------


class UploadError(Exception):
    """Base ingestion error carrying an HTTP status and a client message."""

    status_code = 500
    error_code = "upload_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class UnsupportedFormatError(UploadError):
    status_code = 415
    error_code = "unsupported_format"


class EmptyFileError(UploadError):
    status_code = 400
    error_code = "empty_file"


class FileTooLargeError(UploadError):
    status_code = 413
    error_code = "file_too_large"


class ProcessingError(UploadError):
    status_code = 500
    error_code = "processing_error"


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class UploadResult:
    """Summary of an ingest-and-index run, returned to the API layer."""

    document: str
    stored_path: str
    media_type: str
    bytes: int
    pages: int
    chunks_created: int
    chunks_indexed: int
    collection: str
    embedding_model: str
    total_records_after: int
    doc_hash: str
    elapsed_ms: float
    status: str = "indexed"
    chunk_ids: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Validation & filename safety
# ---------------------------------------------------------------------------


def sanitize_filename(name: str | None) -> str:
    """Return a safe single-component filename (no path traversal)."""
    if not name:
        raise EmptyFileError("Uploaded file must include a filename.")
    # Take only the final path component and drop any unsafe characters.
    base = Path(name.replace("\\", "/")).name
    base = _CONTROL_CHARS.sub("", base).strip()
    if not base:
        raise EmptyFileError("Uploaded filename resolved to an empty string.")
    return base


def validate_upload(
    *,
    filename: str,
    size_bytes: int,
    allowed_extensions: frozenset[str],
    max_upload_bytes: int,
) -> str:
    """Validate an upload and return its lowercased suffix.

    Raises :class:`EmptyFileError`, :class:`FileTooLargeError`, or
    :class:`UnsupportedFormatError` for empty, oversized, or disallowed files.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in allowed_extensions:
        supported = ", ".join(sorted(allowed_extensions))
        raise UnsupportedFormatError(
            f"Unsupported file type '{suffix or filename}'. Allowed: {supported}."
        )
    if size_bytes <= 0:
        raise EmptyFileError("Uploaded file is empty.")
    if size_bytes > max_upload_bytes:
        limit_mb = max_upload_bytes / 1_000_000
        raise FileTooLargeError(
            f"File of {size_bytes} bytes exceeds the {limit_mb:.1f} MB upload limit."
        )
    return suffix


# ---------------------------------------------------------------------------
# Text extraction + cleaning + chunking
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_CHARS.sub(" ", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    return re.sub(r"[ \t]+", " ", text).strip()


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", text)
    text = _TAG.sub(" ", text)
    return unicodedata.normalize("NFC", text)


def extract_pages(content: bytes, suffix: str) -> list[tuple[int, str]]:
    """Extract ``(page_number, text)`` pairs from raw file bytes.

    PDFs are read page-by-page via PyMuPDF; ``.txt``/``.md`` are decoded as-is;
    HTML has tags stripped. All extracted text is cleaned.
    """
    if suffix == ".pdf":
        import pymupdf

        pages: list[tuple[int, str]] = []
        with pymupdf.open(stream=content, filetype="pdf") as doc:
            for index, page in enumerate(doc, start=1):
                cleaned = _clean_text(page.get_text("text"))
                if cleaned:
                    pages.append((index, cleaned))
        if not pages:
            raise EmptyFileError("No readable text could be extracted from the PDF.")
        return pages

    if suffix in {".html", ".htm"}:
        text = _clean_text(_strip_html(content.decode("utf-8", errors="replace")))
    else:  # .txt, .md, .markdown
        text = _clean_text(content.decode("utf-8", errors="replace"))

    if not text:
        raise EmptyFileError("The uploaded text document is empty.")
    return [(1, text)]


def chunk_pages(pages: list[tuple[int, str]], *, doc_hash: str, source_name: str) -> list[dict[str, Any]]:
    """Split cleaned pages into token-aware chunks with retrieval metadata.

    Sentence units are grouped per page up to ``MAX_TOKENS``; a new chunk starts
    at each page boundary. Chunk ids are deterministic from the document hash so
    re-uploading the same file upserts (replaces) rather than duplicates.
    """
    enc = _encoding()
    chunks: list[dict[str, Any]] = []
    idx = 0
    for page_no, page_text in pages:
        buffer: list[str] = []
        buffer_tokens = 0
        for sentence in (s.strip() for s in _SENTENCE.split(page_text) if s.strip()):
            count = len(enc.encode(sentence))
            if buffer and buffer_tokens + count > MAX_TOKENS:
                chunks.append(_make_chunk(buffer, idx, page_no, doc_hash, source_name, buffer_tokens))
                idx += 1
                buffer, buffer_tokens = [], 0
            if count > MAX_TOKENS:
                # Split an over-long sentence on token boundaries (no overlap).
                token_ids = enc.encode(sentence)
                for start in range(0, len(token_ids), MAX_TOKENS):
                    part = enc.decode(token_ids[start : start + MAX_TOKENS])
                    part_tokens = len(enc.encode(part))
                    chunks.append(_make_chunk([part], idx, page_no, doc_hash, source_name, part_tokens))
                    idx += 1
                continue
            buffer.append(sentence)
            buffer_tokens += count
        if buffer:
            chunks.append(_make_chunk(buffer, idx, page_no, doc_hash, source_name, buffer_tokens))
            idx += 1
    return chunks


def _make_chunk(
    sentences: list[str],
    index: int,
    page_no: int,
    doc_hash: str,
    source_name: str,
    token_count: int,
) -> dict[str, Any]:
    text = " ".join(sentences).strip()
    return {
        "id": f"upload-{doc_hash}-{index:06d}",
        "text": text,
        "metadata": {
            "modality": "text",
            "source": source_name,
            "volume": "",
            "page_start": page_no,
            "page_end": page_no,
            "clause": "",
            "heading": "",
            "chunk_index": index,
            "token_count": token_count,
            "doc_hash": doc_hash,
            "origin": "upload",
        },
    }



# ---------------------------------------------------------------------------
# Model cache + safe storage
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, Any] = {}


def _get_embed_model(model_name: str):
    """Return a cached SentenceTransformer for *model_name*."""
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model for uploads: %s", model_name)
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def store_upload(content: bytes, filename: str, upload_dir: Path, doc_hash: str) -> Path:
    """Write the uploaded bytes under *upload_dir* with a collision-safe name.

    The stored name is prefixed with the document hash so two different files
    that share a filename never overwrite each other, and the original (already
    sanitized) name is preserved for readability.
    """
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = sanitize_filename(filename)
    stored_path = upload_dir / f"{doc_hash[:10]}-{safe_name}"
    stored_path.write_bytes(content)
    return stored_path


# ---------------------------------------------------------------------------
# Embedding + indexing (Task 2)
# ---------------------------------------------------------------------------


def embed_and_index(
    chunks: list[dict[str, Any]],
    *,
    model_name: str,
    persist_dir: Path,
    collection_name: str,
) -> tuple[int, list[str]]:
    """Embed *chunks* and upsert them into the target Chroma collection.

    Returns ``(total_records_after, indexed_ids)``. Raises
    :class:`ProcessingError` if the collection was built with a different
    embedding model or if embedding/storage fails.
    """
    from chromadb import PersistentClient

    from src.embeddings.embed_text_chunks import _chroma_metadata, _safe_collection_name

    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]

    model = _get_embed_model(model_name)
    try:
        vectors = model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    except Exception as exc:  # pragma: no cover - model/runtime failure
        raise ProcessingError(f"Embedding failed: {exc}") from exc

    safe_name = _safe_collection_name(collection_name)
    client = PersistentClient(path=str(persist_dir))
    try:
        collection = client.get_or_create_collection(
            name=safe_name,
            metadata={"hnsw:space": "cosine", "embedding_model": model_name},
        )
    except Exception as exc:
        raise ProcessingError(f"Could not open collection '{safe_name}': {exc}") from exc

    stored_model = (collection.metadata or {}).get("embedding_model")
    if stored_model and stored_model != model_name:
        raise ProcessingError(
            f"Collection '{safe_name}' was built with model {stored_model!r}, "
            f"but uploads use {model_name!r}. Configure a matching collection."
        )

    try:
        collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=[_chroma_metadata(c) for c in chunks],
            embeddings=vectors.tolist(),
        )
    except Exception as exc:
        raise ProcessingError(f"Vector database indexing failed: {exc}") from exc

    return collection.count(), ids


def index_document(
    *,
    content: bytes,
    filename: str,
    config,
    collection_name: str | None = None,
    persist_dir: Path | None = None,
    model_name: str | None = None,
) -> UploadResult:
    """Validate, store, extract, chunk, embed, and index an uploaded document.

    *config* is an :class:`src.api.config.APIConfig`; its upload, vector-DB, and
    embedding settings drive the run. Raises an :class:`UploadError` subclass on
    any client- or server-side failure.
    """
    t0 = time.perf_counter()

    safe_name = sanitize_filename(filename)
    size_bytes = len(content)
    allowed = config.allowed_extensions
    max_bytes = config.max_upload_bytes
    # Empty-content check happens before format so a 0-byte valid type is caught.
    suffix = validate_upload(
        filename=safe_name,
        size_bytes=size_bytes,
        allowed_extensions=allowed,
        max_upload_bytes=max_bytes,
    )

    doc_hash = hashlib.sha256(content).hexdigest()
    effective_collection = collection_name or config.collection_name
    effective_persist = persist_dir or config.persist_dir
    effective_model = model_name or config.embedding_model

    pages = extract_pages(content, suffix)
    # Chunk ids use a short hash prefix for readability; the stored file name
    # keeps a slightly longer prefix. The full SHA-256 still guarantees global
    # uniqueness so re-uploading the same file upserts rather than duplicates.
    id_hash = doc_hash[:16]
    chunks = chunk_pages(pages, doc_hash=id_hash, source_name=safe_name)
    if not chunks:
        raise EmptyFileError("Document produced no indexable text chunks.")

    stored_path = store_upload(content, safe_name, config.upload_dir, doc_hash)

    total_after, indexed_ids = embed_and_index(
        chunks,
        model_name=effective_model,
        persist_dir=effective_persist,
        collection_name=effective_collection,
    )

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "Indexed upload %s: %d chunks → collection %s (%.0f ms)",
        safe_name, len(indexed_ids), effective_collection, elapsed,
    )
    return UploadResult(
        document=safe_name,
        stored_path=str(stored_path),
        media_type=suffix,
        bytes=size_bytes,
        pages=len(pages),
        chunks_created=len(chunks),
        chunks_indexed=len(indexed_ids),
        collection=effective_collection,
        embedding_model=effective_model,
        total_records_after=total_after,
        doc_hash=doc_hash[:16],
        elapsed_ms=round(elapsed, 1),
        status="indexed",
        chunk_ids=indexed_ids,
    )
