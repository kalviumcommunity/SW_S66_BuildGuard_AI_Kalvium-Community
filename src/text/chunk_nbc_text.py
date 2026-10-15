"""Extract, chunk, clean, and validate text from one NBC 2016 PDF volume."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
import unicodedata
from collections import Counter
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
import tiktoken

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PDFS = [ROOT / "data" / "nationalbuilding01.pdf"]
OUTPUT_DIR = ROOT / "outputs" / "chunks"
LOG_PATH = OUTPUT_DIR / "chunking.log"
MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-dot-v1"
MAX_TOKENS = 500
SHORT_CHUNK_TOKENS = 10
BOILERPLATE_MAX_TOKENS = 12
BOILERPLATE_REPEAT_THRESHOLD = 3
OVERSIZE_OVERLAP = 50
DISTANCE_PERCENTILE = 85.0
EMBEDDING_BATCH_SIZE = 64
ENCODING = tiktoken.get_encoding("cl100k_base")

_CLAUSE = re.compile(
    r"^\s*((?:\d+\.)+\d*[A-Za-z]?(?:\([a-z0-9ivx]+\))?|(?:PART|SECTION|ANNEX|APPENDIX)\s+[A-Z0-9.()/-]+)\s*",
    re.I,
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
_COMMON = set(
    "a an and are as at be been being by can clause code for from has have in into is it may must no not of on or shall should that the their this to was were which with within where when will works".split()
)


def _volume_number(path: Path, fallback: int) -> int:
    match = re.search(r"nationalbuilding0?(\d+)", path.stem, re.I)
    return int(match.group(1)) if match else fallback


@dataclass
class PageText:
    volume: int
    source: str
    page: int
    text: str
    extraction: str


@dataclass
class Unit:
    text: str
    volume: int
    source: str
    page_start: int
    page_end: int
    clause: str | None
    heading: str | None
    extraction: str


_WORKER_DOCS: dict[str, pymupdf.Document] = {}


def _init_page_worker(paths: list[str]) -> None:
    global _WORKER_DOCS
    tess_dir = Path(r"C:\Program Files\Tesseract-OCR")
    if tess_dir.is_dir():
        os.environ["PATH"] = str(tess_dir) + os.pathsep + os.environ.get("PATH", "")
    _WORKER_DOCS = {path: pymupdf.open(path) for path in paths}


def _extract_one_page(task: tuple[int, str, int]) -> PageText:
    volume, source, page_index = task
    page = _WORKER_DOCS[source][page_index]
    text = page.get_text("text", sort=False)
    method = "native"
    if _needs_ocr(text):
        ocr = _ocr_page(page)
        if _quality(ocr) > _quality(text):
            text, method = ocr, "ocr"
    return PageText(volume, Path(source).name, page_index + 1, text, method)


def _quality(text: str) -> float:
    """Estimate whether PDF text is readable English rather than bad font maps."""
    if not text.strip():
        return 0.0
    words = re.findall(r"[A-Za-z]{2,}", text.lower())
    if not words:
        return 0.0
    common = sum(word in _COMMON for word in words) / len(words)
    nonspace = max(1, sum(not ch.isspace() for ch in text))
    alpha = sum(ch.isalpha() for ch in text) / nonspace
    replacement = text.count("\ufffd") / max(1, len(text))
    symbol_noise = sum(ch in "!@#$%^&*_+=[]{}|~`\\<>" for ch in text) / nonspace
    mixed_tokens = sum(
        any(ch.isalpha() for ch in word) and any(ch.isdigit() for ch in word)
        for word in re.findall(r"[A-Za-z0-9]+", text)
    ) / max(1, len(words))
    # Custom-font maps often turn every word into letter+digit codes (for
    # example, `YGKPLT9GVP9...`) while keeping a misleadingly high letter ratio.
    score = 0.58 * common + 0.22 * alpha + 0.20 * (1 - min(1.0, replacement * 20))
    return max(
        0.0, score - min(0.75, mixed_tokens * 0.85) - min(0.35, symbol_noise * 1.8)
    )


def _needs_ocr(text: str) -> bool:
    """Only invoke the slower OCR pass for strong signs of a bad text map."""
    if not text.strip():
        return True
    nonspace = max(1, sum(not ch.isspace() for ch in text))
    alpha = sum(ch.isalpha() for ch in text) / nonspace
    words = re.findall(r"[A-Za-z0-9]+", text)
    mixed = sum(
        any(ch.isalpha() for ch in word) and any(ch.isdigit() for ch in word)
        for word in words
    ) / max(1, len(words))
    symbols = sum(ch in "!@#$%^&*_+=[]{}|~`\\<>" for ch in text) / nonspace
    lexical_words = re.findall(r"[A-Za-z]{2,}", text.lower())
    common_ratio = sum(word in _COMMON for word in lexical_words) / max(
        1, len(lexical_words)
    )
    # Some NBC pages use custom fonts whose broken Unicode maps produce
    # plausible-looking letters mixed with punctuation, rather than replacement
    # characters. A low common-word ratio plus symbol noise catches those pages.
    encoded_looking = common_ratio < 0.10 and symbols > 0.04
    return (
        alpha < 0.35
        or mixed > 0.25
        or symbols > 0.09
        or encoded_looking
        or "\ufffd" in text
    )


def _ocr_page(page: pymupdf.Page) -> str:
    try:
        text_page = page.get_textpage_ocr(language="eng", dpi=180, full=True)
        return page.get_text(textpage=text_page)
    except (
        Exception
    ) as exc:  # OCR failures are reported per page; the run can continue.
        print(f"OCR warning page {page.number + 1}: {exc}", file=sys.stderr)
        return ""


def _extract_pages(
    paths: list[Path], page_limit: int | None = None
) -> tuple[list[PageText], dict[str, int]]:
    tasks: list[tuple[int, str, int]] = []
    for index, path in enumerate(paths, 1):
        volume = _volume_number(path, index)
        with pymupdf.open(path) as doc:
            count = min(doc.page_count, page_limit) if page_limit else doc.page_count
            tasks.extend((volume, str(path), index) for index in range(count))
    pages: list[PageText | None] = [None] * len(tasks)
    stats = {"pages": 0, "native": 0, "ocr": 0, "empty": 0}
    for index, path in enumerate(paths, 1):
        volume = _volume_number(path, index)
        with pymupdf.open(path) as doc:
            count = min(doc.page_count, page_limit) if page_limit else doc.page_count
            print(
                f"Reading volume {volume}: {path.name} ({count} pages{' sampled' if page_limit else ''})"
            )
    workers = min(4, os.cpu_count() or 2)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_page_worker,
        initargs=([str(p) for p in paths],),
    ) as pool:
        futures = {
            pool.submit(_extract_one_page, task): i for i, task in enumerate(tasks)
        }
        for future in as_completed(futures):
            index = futures[future]
            page_text = future.result()
            pages[index] = page_text
            stats["pages"] += 1
            if page_text.text.strip():
                stats[page_text.extraction] += 1
            else:
                stats["empty"] += 1
            if stats["pages"] % 100 == 0:
                print(
                    f"  processed {stats['pages']}/{len(tasks)} pages (native={stats['native']}, OCR={stats['ocr']})"
                )
    return [page for page in pages if page is not None], stats


def _clean_page(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    # Keep paragraph breaks and retain clause numbers, symbols, table rows, and units.
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _units(pages: list[PageText]) -> list[Unit]:
    out: list[Unit] = []
    current_heading: str | None = None
    for page in pages:
        cleaned = _clean_page(page.text)
        for raw in re.split(r"\n\s*\n", cleaned):
            paragraph = " ".join(
                line.strip() for line in raw.splitlines() if line.strip()
            ).strip()
            if not paragraph:
                continue
            heading_like = len(paragraph) < 180 and (
                paragraph.isupper()
                or re.match(r"^(PART|SECTION|ANNEX|APPENDIX)\b", paragraph, re.I)
            )
            if heading_like and not _needs_ocr(paragraph):
                current_heading = paragraph[:180]
            clause_match = _CLAUSE.match(paragraph)
            clause = clause_match.group(1).strip() if clause_match else None
            for sentence in (
                s.strip() for s in _SENTENCE.split(paragraph) if s.strip()
            ):
                out.append(
                    Unit(
                        sentence,
                        page.volume,
                        page.source,
                        page.page,
                        page.page,
                        clause,
                        current_heading,
                        page.extraction,
                    )
                )
    return out


def _cosine_distances(model: Any, units: list[Unit]) -> list[float]:
    if len(units) < 2:
        return []
    vectors = model.encode(
        [unit.text for unit in units],
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return [
        float(1.0 - (vectors[i] * vectors[i + 1]).sum())
        for i in range(len(vectors) - 1)
    ]


def _chunk(
    units: list[Unit], distances: list[float], percentile: float
) -> tuple[list[dict[str, Any]], float]:
    cutoff = (
        sorted(distances)[
            min(len(distances) - 1, int((len(distances) - 1) * percentile / 100))
        ]
        if distances
        else 1.0
    )
    chunks: list[dict[str, Any]] = []
    buffer: list[Unit] = []
    buffer_tokens = 0

    def emit() -> None:
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        text = " ".join(unit.text for unit in buffer).strip()
        first, last = buffer[0], buffer[-1]
        idx = len(chunks)
        chunks.append(
            {
                "id": f"nbc-v{first.volume}-p{first.page_start}-{idx:06d}",
                "text": text,
                "metadata": {
                    "modality": "text",
                    "source": first.source,
                    "volume": first.volume,
                    "page_start": first.page_start,
                    "page_end": last.page_end,
                    "clause": first.clause,
                    "heading": first.heading,
                    "extraction_methods": sorted({unit.extraction for unit in buffer}),
                    "chunk_index": idx,
                    "token_count": buffer_tokens,
                    "chunk_method": "semantic",
                    "distance_percentile": percentile,
                },
            }
        )
        buffer, buffer_tokens = [], 0

    for i, unit in enumerate(units):
        count = len(ENCODING.encode(unit.text))
        prev = units[i - 1] if i else None
        structural_break = bool(
            prev
            and (
                unit.volume != prev.volume
                or (unit.clause and unit.clause != prev.clause)
                or (unit.heading and unit.heading != prev.heading)
            )
        )
        semantic_break = bool(buffer and i > 0 and distances[i - 1] >= cutoff)
        if buffer and (
            structural_break or semantic_break or buffer_tokens + count > MAX_TOKENS
        ):
            emit()
        # Extra-long single sentences are split at token boundaries; only those
        # forced fragments receive the configured 50-token overlap.
        if count > MAX_TOKENS:
            ids = ENCODING.encode(unit.text)
            start = 0
            while start < len(ids):
                end = min(start + MAX_TOKENS, len(ids))
                fragment = Unit(
                    ENCODING.decode(ids[start:end]),
                    unit.volume,
                    unit.source,
                    unit.page_start,
                    unit.page_end,
                    unit.clause,
                    unit.heading,
                    unit.extraction,
                )
                buffer = [fragment]
                buffer_tokens = end - start
                emit()
                if end == len(ids):
                    break
                start = end - OVERSIZE_OVERLAP
            continue
        buffer.append(unit)
        buffer_tokens += count
    emit()
    return chunks, cutoff


def _split_at_word_boundaries(text: str) -> list[str] | None:
    """Split an over-limit chunk without dropping words."""
    parts: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if len(ENCODING.encode(candidate)) <= MAX_TOKENS:
            current = candidate
            continue
        if current:
            parts.append(current)
        if len(ENCODING.encode(word)) > MAX_TOKENS:
            return None
        current = word
    if current:
        parts.append(current)
    return parts


def _boilerplate_signature(text: str) -> str:
    """Match recurring page labels even when their page number changes."""
    normalized = re.sub(r"\d+", "#", text.casefold())
    return re.sub(r"[^a-z0-9#]+", " ", normalized).strip()


def _clean_and_validate(raw_chunks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Produce final chunks; invalid records are excluded and recorded in the report."""
    control_chars = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    counts: Counter[str] = Counter()
    warning_samples: dict[str, list[str]] = {}
    exclusion_samples: dict[str, list[str]] = {}
    warning_count = 0
    seen_ids: set[str] = set()
    output_ids: set[str] = set()
    cleaned: list[dict[str, Any]] = []

    def warn(category: str, chunk_id: str) -> None:
        nonlocal warning_count
        warning_count += 1
        counts[category] += 1
        samples = warning_samples.setdefault(category, [])
        if len(samples) < 20:
            samples.append(chunk_id)

    def exclude(category: str, chunk_id: str) -> None:
        counts[category] += 1
        samples = exclusion_samples.setdefault(category, [])
        if len(samples) < 20:
            samples.append(chunk_id)

    for record in raw_chunks:
        counts["raw_chunks"] += 1
        chunk_id = record.get("id")
        metadata = record.get("metadata")
        text = record.get("text")
        if (
            not isinstance(chunk_id, str)
            or not chunk_id
            or chunk_id in seen_ids
            or not isinstance(metadata, dict)
            or not isinstance(text, str)
            or metadata.get("modality") != "text"
            or not isinstance(metadata.get("source"), str)
            or not metadata["source"].strip()
            or not isinstance(metadata.get("volume"), int)
            or not isinstance(metadata.get("page_start"), int)
            or not isinstance(metadata.get("page_end"), int)
            or metadata["page_start"] > metadata["page_end"]
        ):
            counts["invalid_records_excluded"] += 1
            continue
        seen_ids.add(chunk_id)
        normalized = unicodedata.normalize("NFC", text)
        normalized, removed = control_chars.subn(" ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if removed:
            warn("control_characters_removed", chunk_id)
        if not normalized:
            counts["empty_records_excluded"] += 1
            continue
        parts = [normalized]
        if len(ENCODING.encode(normalized)) > MAX_TOKENS:
            parts = _split_at_word_boundaries(normalized) or []
            if not parts:
                counts["unsplittable_records_excluded"] += 1
                continue
            warn("over_limit_chunks_split", chunk_id)
        for part_index, part_text in enumerate(parts, start=1):
            token_count = len(ENCODING.encode(part_text))
            if token_count > MAX_TOKENS:
                counts["over_limit_records_excluded"] += 1
                continue
            part_id = chunk_id if part_index == 1 else f"{chunk_id}-part-{part_index:02d}"
            if part_id in output_ids:
                counts["duplicate_output_ids_excluded"] += 1
                continue
            output_ids.add(part_id)
            part_metadata = dict(metadata)
            part_metadata["token_count"] = token_count
            if len(parts) > 1:
                part_metadata["source_chunk_id"] = chunk_id
                part_metadata["subchunk_index"] = part_index
                part_metadata["subchunk_count"] = len(parts)
            if token_count <= SHORT_CHUNK_TOKENS:
                warn("short_chunks", part_id)
            if "\ufffd" in part_text:
                warn("replacement_characters", part_id)
            cleaned.append({"id": part_id, "text": part_text, "metadata": part_metadata})
            counts["final_chunks"] += 1

    signatures = Counter(
        _boilerplate_signature(record["text"])
        for record in cleaned
        if record["metadata"]["token_count"] <= BOILERPLATE_MAX_TOKENS
    )
    filtered: list[dict[str, Any]] = []
    for record in cleaned:
        signature = _boilerplate_signature(record["text"])
        if (
            record["metadata"]["token_count"] <= BOILERPLATE_MAX_TOKENS
            and signature
            and signatures[signature] >= BOILERPLATE_REPEAT_THRESHOLD
        ):
            exclude("repeated_boilerplate_excluded", record["id"])
            continue
        filtered.append(record)
    counts["final_chunks"] = len(filtered)

    return filtered, {
        "counts": dict(counts),
        "warning_samples": warning_samples,
        "exclusion_samples": exclusion_samples,
        "warnings": warning_count,
        "excluded_records": sum(
            value for key, value in counts.items() if key.endswith("_excluded")
        ),
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_run_log(lines: list[str], *, start_run: bool = False) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8", newline="\n") as handle:
        if start_run:
            handle.write(
                f"\n=== Run {datetime.now().astimezone().isoformat(timespec='seconds')} ===\n"
            )
        handle.write("\n".join(lines) + "\n")


def _print_preview(chunks: list[dict[str, Any]], sample: int) -> None:
    first = chunks[:sample]
    selected = first + chunks[max(sample, len(chunks) - sample) :]
    print("\n" + "=" * 88)
    print(f"CHUNK PREVIEW | showing {len(selected)} of {len(chunks)} text chunks")
    print("=" * 88)
    for position, record in enumerate(selected, 1):
        metadata = record["metadata"]
        print(f"\n[{position}] {record['id']}")
        print(f"    Source:      {metadata['source']} (volume {metadata['volume']})")
        print(f"    Pages:       {metadata['page_start']}-{metadata['page_end']}")
        print(f"    Clause:      {metadata.get('clause') or 'not detected'}")
        print(f"    Heading:     {metadata.get('heading') or 'not detected'}")
        print(f"    Tokens:      {metadata['token_count']} / {MAX_TOKENS}")
        print(f"    Text source: {', '.join(metadata.get('extraction_methods', [])) or 'unknown'}")
        print("    Text:")
        wrapped = textwrap.fill(
            " ".join(record["text"].split()),
            width=100,
            initial_indent="      ",
            subsequent_indent="      ",
            break_long_words=False,
            break_on_hyphens=False,
        )
        print(wrapped or "      [empty]")
        print("    " + "-" * 80)


def run(
    paths: list[Path],
    *,
    percentile: float,
    preview: bool,
    sample: int,
    page_limit: int | None = None,
) -> int:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing PDF(s): " + ", ".join(missing))
    from sentence_transformers import SentenceTransformer

    print(f"Loading semantic boundary model {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    pages, stats = _extract_pages(paths, page_limit)
    units = _units(pages)
    if not units:
        raise RuntimeError("No text units were extracted from the PDFs")
    print(f"Preparing semantic boundaries for {len(units)} sentences")
    distances = _cosine_distances(model, units)
    raw_chunks, cutoff = _chunk(units, distances, percentile)
    chunks, validation = _clean_and_validate(raw_chunks)
    print(
        f"Pages: {stats['pages']} (native {stats['native']}, OCR {stats['ocr']}, empty {stats['empty']})"
    )
    print(f"Sentences: {len(units)}; semantic cutoff: {cutoff:.4f} at p{percentile:g}")
    print(
        f"Final text chunks: {len(chunks)}; average tokens: {sum(c['metadata']['token_count'] for c in chunks) / max(1, len(chunks)):.1f}"
    )
    if preview:
        _print_preview(chunks, sample)
        return 0
    volumes = sorted({int(chunk["metadata"]["volume"]) for chunk in chunks})
    volume_label = "_".join(str(volume) for volume in volumes)
    output_path = OUTPUT_DIR / f"text_chunks_volume_{volume_label}.jsonl"
    _write_jsonl(output_path, chunks)
    print(f"Wrote text: {output_path}")
    token_counts = [chunk["metadata"]["token_count"] for chunk in chunks]
    batches = (len(units) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    summary = [
        "Status: completed successfully",
        f"Source PDFs: {', '.join(path.name for path in paths)}",
        f"Volume(s): {', '.join(str(_volume_number(path, index)) for index, path in enumerate(paths, 1))}",
        f"Pages processed: {stats['pages']} (native text {stats['native']}, OCR {stats['ocr']}, empty {stats['empty']})",
        f"Text sentences/units: {len(units)}",
        f"Semantic model: {MODEL_NAME}",
        f"Embedding batches: {batches} (batch size {EMBEDDING_BATCH_SIZE}, {len(units)} sentence vectors)",
        f"Semantic boundary cutoff: {cutoff:.4f} (adjacent-distance percentile p{percentile:g})",
        f"Raw chunks before validation: {len(raw_chunks)}",
        f"Final text chunks: {len(chunks)}",
        f"Validation exclusions: {validation['excluded_records']}; warnings: {validation['warnings']}",
        f"Chunk token counts: average {sum(token_counts) / max(1, len(token_counts)):.1f}, min {min(token_counts, default=0)}, max {max(token_counts, default=0)} (limit {MAX_TOKENS})",
        f"Text JSONL: {output_path}",
    ]
    print("\nRun summary:")
    for line in summary:
        print(f"  {line}")
    _append_run_log(summary)
    report_path = OUTPUT_DIR / f"text_pipeline_report_volume_{volume_label}.json"
    report = {
        "pipeline": "chunk_nbc_text",
        "status": "completed",
        "source_pdfs": [str(path) for path in paths],
        "volumes": volumes,
        "output": str(output_path),
        "pages": stats,
        "semantic_boundary": {"model": MODEL_NAME, "percentile": percentile, "cutoff": cutoff},
        "validation": validation,
        "final_chunks": len(chunks),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")
    print(f"Run details appended to: {LOG_PATH}")
    return 0



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, help="One PDF path (defaults to nationalbuilding01.pdf)")
    parser.add_argument("--percentile", type=float, default=DISTANCE_PERCENTILE, help="Adjacent sentence-distance percentile (default: 85)")
    parser.add_argument("--preview", action="store_true", help="Run extraction and print chunks without writing chunk files")
    parser.add_argument("--sample", type=int, default=8, help="Chunks to print from each end in preview mode")
    parser.add_argument("--limit-pages", type=int, help="Limit pages per PDF (preview mode only)")
    args = parser.parse_args()
    paths = [args.pdf.resolve()] if args.pdf else DEFAULT_PDFS
    if not 0 < args.percentile < 100:
        parser.error("--percentile must be between 0 and 100")
    if args.limit_pages is not None and (not args.preview or args.limit_pages < 1):
        parser.error("--limit-pages requires preview mode and a positive page count")
    return run(paths, percentile=args.percentile, preview=args.preview, sample=args.sample, page_limit=args.limit_pages)


if __name__ == "__main__":
    raise SystemExit(main())
