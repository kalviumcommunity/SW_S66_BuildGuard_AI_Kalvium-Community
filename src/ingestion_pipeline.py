"""Reusable corpus ingestion, reconciliation, and completeness validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from document_loader import load_document
    from text_cleaner import clean_document
    from token_chunking import token_chunk_documents
except ImportError:  # Support imports through the src namespace in tests.
    from .document_loader import load_document
    from .text_cleaner import clean_document
    from .token_chunking import token_chunk_documents

SUPPORTED_SUFFIXES = {".txt", ".md", ".html", ".htm", ".pdf"}


class IngestionValidationError(ValueError):
    """Raised when an ingestion report fails completeness reconciliation."""


def discover_files(corpus_directory: str | Path) -> list[Path]:
    root = Path(corpus_directory)
    if not root.is_dir():
        return []
    return sorted((path for path in root.rglob("*") if path.is_file()), key=str)


def _source(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _failure_reason(path: Path) -> str:
    if path.suffix.lower() in {".txt", ".md", ".html", ".htm"}:
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return "unreadable UTF-8 text"
        except OSError as error:
            return f"unreadable file: {error}"
        return "loader rejected file or produced no document"
    if path.suffix.lower() == ".pdf":
        return "PDF parser rejected the file, PDF support is unavailable, or it had no text"
    return "unsupported file type"


def validate_ingestion_report(report: dict[str, Any]) -> None:
    """Raise if source status reconciliation or chunk coverage is incomplete."""
    sources = report.get("sources", [])
    chunks = report.get("chunks", [])
    errors: list[str] = []
    source_names = [entry.get("source") for entry in sources]
    if len(source_names) != len(set(source_names)):
        errors.append("duplicate source statuses")
    statuses = {entry.get("status") for entry in sources}
    if not statuses.issubset({"ingested", "skipped"}):
        errors.append("source has an invalid status")
    ingested = [entry for entry in sources if entry.get("status") == "ingested"]
    skipped = [entry for entry in sources if entry.get("status") == "skipped"]
    if report.get("total_sources") != len(sources):
        errors.append("total_sources does not match source entries")
    if report.get("ingested") != len(ingested) or report.get("skipped") != len(skipped):
        errors.append("ingested/skipped counts do not match source statuses")
    if report.get("total_sources") != report.get("ingested", 0) + report.get("skipped", 0):
        errors.append("total sources do not reconcile to ingested plus skipped")
    ingested_names = {entry["source"] for entry in ingested}
    chunk_sources = {chunk.get("source") for chunk in chunks}
    if not chunk_sources.issubset(ingested_names):
        errors.append("chunk source is not an ingested source")
    chunk_counts: dict[str, int] = {}
    for chunk in chunks:
        chunk_counts[chunk["source"]] = chunk_counts.get(chunk["source"], 0) + 1
    for entry in ingested:
        if entry.get("chunk_count", 0) < 1 or chunk_counts.get(entry["source"], 0) < 1:
            errors.append(f"ingested source has no chunks: {entry['source']}")
    if report.get("total_chunks") != len(chunks):
        errors.append("total_chunks does not match chunk entries")
    if errors:
        raise IngestionValidationError("; ".join(errors))


def ingest_corpus(corpus_directory: str | Path, model_name: str | None = None) -> dict[str, Any]:
    """Discover, ingest, chunk, and validate every file in a corpus directory."""
    root = Path(corpus_directory)
    paths = discover_files(root)
    source_entries: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []

    for path in paths:
        source = _source(path, root)
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            source_entries.append({"source": source, "status": "skipped", "reason": "unsupported file type", "chunk_count": 0})
            continue
        try:
            loaded = load_document(path, root)
            if loaded is None:
                source_entries.append({"source": source, "status": "skipped", "reason": _failure_reason(path), "chunk_count": 0})
                continue
            cleaned = clean_document(loaded)
            token_chunks = token_chunk_documents([cleaned], model_name=model_name)
            if not token_chunks:
                source_entries.append({"source": source, "status": "skipped", "reason": "no non-empty token chunks", "chunk_count": 0})
                continue
            source_entries.append({"source": source, "status": "ingested", "reason": None, "chunk_count": len(token_chunks)})
            chunks.extend(
                {
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
                for chunk in token_chunks
            )
        except Exception as error:  # noqa: BLE001 - one bad source must not stop ingestion
            source_entries.append({"source": source, "status": "skipped", "reason": f"ingestion failed: {error}", "chunk_count": 0})

    report: dict[str, Any] = {
        "corpus_directory": str(root),
        "total_sources": len(paths),
        "ingested": sum(entry["status"] == "ingested" for entry in source_entries),
        "skipped": sum(entry["status"] == "skipped" for entry in source_entries),
        "total_chunks": len(chunks),
        "sources": source_entries,
        "chunks": chunks,
        "complete": False,
        "validation_error": None,
    }
    if not paths:
        report["validation_error"] = "corpus directory is missing or empty"
        return report
    try:
        validate_ingestion_report(report)
    except IngestionValidationError as error:
        report["validation_error"] = str(error)
        return report
    report["complete"] = True
    return report
