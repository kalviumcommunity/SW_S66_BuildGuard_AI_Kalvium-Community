"""Conservative text cleanup for loaded corpus documents."""

from __future__ import annotations

import logging
import re
import unicodedata
from importlib import import_module

try:
    from document_loader import LoadedDocument
except ImportError:  # Support imports through the src namespace in tests.
    from .document_loader import LoadedDocument

logger = logging.getLogger(__name__)
_PAGE_MARKER = re.compile(r"^\s*page\s+\d+(?:\s+of\s+\d+)?\s*$", re.IGNORECASE)
_KNOWN_BOILERPLATE = re.compile(
    r"^\s*(?:confidential|internal use only|staff handbook|employee handbook)\s*$",
    re.IGNORECASE,
)


def _fix_text(text: str) -> str:
    """Apply ftfy's mojibake and encoding repair."""
    try:
        ftfy = import_module("ftfy")
    except ModuleNotFoundError:
        logger.warning("ftfy is unavailable; skipping mojibake repair")
        return text
    return ftfy.fix_text(text)


def _repeated_edge_lines(lines: list[str]) -> set[str]:
    """Find repeated non-trivial lines that occur at document edges."""
    counts: dict[str, int] = {}
    edge_keys: set[str] = set()
    edge_size = min(3, len(lines))
    for index, line in enumerate(lines):
        key = line.casefold()
        if line and len(line) > 2:
            counts[key] = counts.get(key, 0) + 1
        if index < edge_size or index >= len(lines) - edge_size:
            edge_keys.add(key)
    return {key for key in edge_keys if counts.get(key, 0) > 1}


def _join_broken_lines(lines: list[str]) -> list[str]:
    joined: list[str] = []
    for line in lines:
        if joined and line and joined[-1] and not re.search(r"[.!?:;]$", joined[-1]) and re.match(r"[a-z]", line):
            joined[-1] = f"{joined[-1]} {line}"
        else:
            joined.append(line)
    return joined


def clean_text(text: str) -> str:
    """Repair encoding, remove noise, and preserve paragraph boundaries."""
    cleaned = unicodedata.normalize("NFKC", _fix_text(text))
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    raw_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.split("\n")]
    repeated = _repeated_edge_lines(raw_lines)

    useful_lines: list[str] = []
    for line in raw_lines:
        if not line:
            useful_lines.append("")
            continue
        if _PAGE_MARKER.match(line) or _KNOWN_BOILERPLATE.match(line):
            continue
        if line.casefold() in repeated:
            continue
        useful_lines.append(line)

    paragraphs: list[str] = []
    current: list[str] = []
    for line in _join_broken_lines(useful_lines):
        if line:
            current.append(line)
        elif current:
            paragraphs.append("\n".join(current))
            current = []
    if current:
        paragraphs.append("\n".join(current))
    return "\n\n".join(paragraphs).strip()


def clean_document(document: LoadedDocument) -> LoadedDocument:
    """Return a cleaned document while preserving its source identity."""
    return LoadedDocument(source=document.source, text=clean_text(document.text))
