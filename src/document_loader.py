"""Load supported documents from a local corpus."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from html.parser import HTMLParser
from importlib import import_module
from pathlib import Path

logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".txt", ".md", ".html", ".htm", ".pdf"}


@dataclass(frozen=True)
class LoadedDocument:
    source: str
    text: str


class _HTMLTextParser(HTMLParser):
    """Collect visible HTML text without third-party dependencies."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"br", "p", "div", "li", "section", "article", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"p", "div", "li", "section", "article", "h1", "h2", "h3"}:
            self.parts.append("\n")


def _source_name(path: Path, corpus_root: Path) -> str:
    try:
        return path.resolve().relative_to(corpus_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _skip(path: Path, message: str) -> None:
    logger.warning("Skipping %s: %s", path, message)


def _load_html(path: Path) -> str:
    parser = _HTMLTextParser()
    parser.feed(path.read_text(encoding="utf-8"))
    parser.close()
    return " ".join("".join(parser.parts).split())


def _load_pdf(path: Path) -> str:
    pypdf = import_module("pypdf")
    reader = pypdf.PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def load_document(path: str | Path, corpus_root: str | Path) -> LoadedDocument | None:
    """Load one supported file, returning None and logging on any skip."""
    document_path = Path(path)
    root = Path(corpus_root)
    source = _source_name(document_path, root)

    if not document_path.is_file():
        _skip(document_path, "file is missing or not a regular file")
        return None

    suffix = document_path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        _skip(document_path, f"unsupported file type {suffix or '<none>'}")
        return None

    try:
        if suffix in {".txt", ".md"}:
            text = document_path.read_text(encoding="utf-8")
        elif suffix in {".html", ".htm"}:
            text = _load_html(document_path)
        else:
            text = _load_pdf(document_path)
    except ImportError as error:
        _skip(document_path, f"PDF support is unavailable: {error}")
        return None
    except (OSError, UnicodeError) as error:
        _skip(document_path, f"could not be read: {error}")
        return None
    except Exception as error:  # noqa: BLE001 - parser libraries use varied error types
        _skip(document_path, f"could not be parsed: {error}")
        return None

    if suffix == ".pdf" and not text.strip():
        _skip(document_path, "PDF contained no extractable text")
        return None
    return LoadedDocument(source=source, text=text)


def load_corpus(directory: str | Path) -> list[LoadedDocument]:
    """Recursively load supported files while continuing past bad files."""
    corpus_root = Path(directory)
    if not corpus_root.is_dir():
        _skip(corpus_root, "corpus directory is missing or not a directory")
        return []

    documents: list[LoadedDocument] = []
    for path in sorted((item for item in corpus_root.rglob("*") if item.is_file()), key=str):
        document = load_document(path, corpus_root)
        if document is not None:
            documents.append(document)
    return documents
