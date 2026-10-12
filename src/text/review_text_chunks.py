"""Prepare NBC chunk text and heading metadata for human review.

This pass does not rewrite OCR text. It rebuilds confident volume 1 heading
labels from the document's contents page ranges and emits review queues for
OCR and chunks that cross a heading boundary.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "outputs" / "cleaned_chunks" / "text_chunks.cleaned.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "chunk_review"
HEADING_MAP_VERSION = "nbc_volume1_contents_pages_v1"
OCR_SAMPLE_BAND_PAGES = 25
SYMBOLS = set("\\|~^_={}[]<>")

# One-based physical PDF pages, checked against the headings visible in the
# extracted chunks and the page totals in the volume 1 contents pages.
HEADING_RANGES: tuple[tuple[int, int, str], ...] = (
    (34, 45, "Part 0 — Integrated Approach"),
    (46, 73, "Part 1 — Definitions"),
    (74, 105, "Part 2 — Administration"),
    (106, 255, "Part 3 — Development Control Rules and General Building Requirements"),
    (256, 371, "Part 4 — Fire and Life Safety"),
    (372, 417, "Part 5 — Building Materials"),
    (418, 551, "Part 6 — Structural Design, Section 1 — Loads, Forces and Effects"),
    (552, 611, "Part 6 — Structural Design, Section 2 — Soils and Foundations"),
    (612, 667, "Part 6 — Structural Design, Section 3A — Timber"),
    (668, 697, "Part 6 — Structural Design, Section 3B — Bamboo"),
    (698, 783, "Part 6 — Structural Design, Section 4 — Masonry"),
    (784, 901, "Part 6 — Structural Design, Section 5A — Plain and Reinforced Concrete"),
    (902, 953, "Part 6 — Structural Design, Section 5B — Prestressed Concrete"),
    (954, 1091, "Part 6 — Structural Design, Section 6 — Steel"),
    (1092, 1133, "Part 6 — Structural Design, Section 7A — Prefabricated Concrete"),
    (1134, 1145, "Part 6 — Structural Design, Section 7B — Systems Building and Mixed/Composite Construction"),
    (1146, 1225, "Part 6 — Structural Design, Section 8 — Glass and Glazing"),
)


def _heading_for_page(page: int) -> str | None:
    for start, end, heading in HEADING_RANGES:
        if start <= page <= end:
            return heading
    return None


def _flags(record: dict[str, Any]) -> list[str]:
    metadata = record["metadata"]
    text = record["text"]
    reasons: list[str] = []
    if metadata.get("token_count", 0) <= 10:
        reasons.append("short_chunk")
    if "\ufffd" in text:
        reasons.append("replacement_character")
    if any(ord(char) < 32 and char not in "\t\r\n" for char in text):
        reasons.append("control_character")
    visible = [char for char in text if not char.isspace()]
    symbol_ratio = sum(char in SYMBOLS for char in visible) / max(1, len(visible))
    if symbol_ratio > 0.03:
        reasons.append("symbol_noise_candidate")
    letter_ratio = sum(char.isalpha() for char in visible) / max(1, len(visible))
    if letter_ratio < 0.45:
        reasons.append("low_letter_ratio_candidate")
    return reasons


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} is not a JSON object")
            records.append(record)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run(input_path: Path, output_dir: Path) -> int:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Cleaned text JSONL not found: {input_path}")
    if input_path.parent == output_dir:
        raise ValueError("Choose a separate output directory to preserve the input dataset")

    output_dir.mkdir(parents=True, exist_ok=True)
    records = _read_jsonl(input_path)
    output_records: list[dict[str, Any]] = []
    ocr_queue: list[dict[str, Any]] = []
    heading_queue: list[dict[str, Any]] = []
    page_bands: dict[int, list[tuple[dict[str, Any], list[str]]]] = {}
    counts: Counter[str] = Counter()
    ids: set[str] = set()

    for record in records:
        if not isinstance(record.get("text"), str) or not isinstance(record.get("metadata"), dict):
            raise ValueError(f"Invalid chunk record: {record.get('id', '<missing id>')}")
        if record.get("id") in ids:
            raise ValueError(f"Duplicate chunk ID: {record['id']}")
        ids.add(record["id"])

        updated = dict(record)
        metadata = dict(record["metadata"])
        updated["metadata"] = metadata
        start = metadata.get("page_start")
        end = metadata.get("page_end", start)
        if not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
            raise ValueError(f"Invalid page range for {record['id']}")

        mapped = {_heading_for_page(page) for page in range(start, end + 1)}
        if len(mapped) == 1 and None not in mapped:
            metadata["heading"] = next(iter(mapped))
            metadata["heading_review_status"] = "assigned_from_contents_page_ranges"
            counts["headings_assigned"] += 1
        elif None in mapped and len(mapped) == 1:
            metadata["heading"] = None
            metadata["heading_review_status"] = "outside_contents_page_ranges"
            heading_queue.append(
                {
                    "chunk_id": record["id"],
                    "page_start": start,
                    "page_end": end,
                    "current_heading": record["metadata"].get("heading"),
                    "proposed_heading": None,
                    "review_status": "pending",
                    "text_preview": record["text"][:500],
                }
            )
            counts["heading_review_outside_map"] += 1
        else:
            metadata["heading"] = None
            metadata["heading_review_status"] = "crosses_heading_boundary"
            heading_queue.append(
                {
                    "chunk_id": record["id"],
                    "page_start": start,
                    "page_end": end,
                    "current_heading": record["metadata"].get("heading"),
                    "proposed_heading": None,
                    "review_status": "pending",
                    "text_preview": record["text"][:500],
                }
            )
            counts["heading_review_boundary_crossings"] += 1
        metadata["heading_source"] = HEADING_MAP_VERSION

        methods = metadata.get("extraction_methods", [])
        if "ocr" in methods:
            counts["ocr_chunks"] += 1
            reasons = _flags(record)
            if reasons:
                ocr_queue.append(
                    {
                        "chunk_id": record["id"],
                        "source": metadata.get("source"),
                        "page_start": start,
                        "page_end": end,
                        "priority": "high" if len(reasons) > 1 else "review",
                        "flags": reasons,
                        "current_text": record["text"],
                        "verified_text": "",
                        "review_status": "pending",
                        "review_note": "",
                    }
                )
                counts["ocr_flagged"] += 1
            band = start // OCR_SAMPLE_BAND_PAGES
            page_bands.setdefault(band, []).append((record, reasons))
        output_records.append(updated)

    # Add a deterministic page-stratified sample so OCR without obvious noise
    # also receives human review. Avoid duplicating already flagged records.
    queued_ids = {item["chunk_id"] for item in ocr_queue}
    for band, candidates in sorted(page_bands.items()):
        center = band * OCR_SAMPLE_BAND_PAGES + OCR_SAMPLE_BAND_PAGES // 2
        sample, _ = min(
            candidates,
            key=lambda pair: (
                abs(pair[0]["metadata"]["page_start"] - center),
                pair[0]["id"],
            ),
        )
        if sample["id"] not in queued_ids:
            metadata = sample["metadata"]
            ocr_queue.append(
                {
                    "chunk_id": sample["id"],
                    "source": metadata.get("source"),
                    "page_start": metadata.get("page_start"),
                    "page_end": metadata.get("page_end"),
                    "priority": "sample",
                    "flags": ["stratified_sample"],
                    "current_text": sample["text"],
                    "verified_text": "",
                    "review_status": "pending",
                    "review_note": "",
                }
            )
            queued_ids.add(sample["id"])
            counts["ocr_stratified_samples"] += 1

    out_jsonl = output_dir / "text_chunks.headings_rebuilt.jsonl"
    ocr_jsonl = output_dir / "ocr_review_queue.jsonl"
    heading_jsonl = output_dir / "heading_review_queue.jsonl"
    report_path = output_dir / "review_report.json"
    log_path = output_dir / "chunk_review.log"
    _write_jsonl(out_jsonl, output_records)
    _write_jsonl(ocr_jsonl, ocr_queue)
    _write_jsonl(heading_jsonl, heading_queue)

    report = {
        "status": "review_files_created",
        "input": str(input_path),
        "heading_output": str(out_jsonl),
        "ocr_review_queue": str(ocr_jsonl),
        "heading_review_queue": str(heading_jsonl),
        "heading_map_version": HEADING_MAP_VERSION,
        "counts": dict(counts),
        "records_in": len(records),
        "records_out": len(output_records),
        "ocr_review_items": len(ocr_queue),
        "heading_review_items": len(heading_queue),
        "note": "OCR text remains unchanged until verified against the source PDF.",
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    log_lines = [
        f"\n=== Run {datetime.now().astimezone().isoformat(timespec='seconds')} ===",
        f"Input records: {len(records)}",
        f"Heading output records: {len(output_records)}",
        f"OCR chunks: {counts['ocr_chunks']}",
        f"OCR flagged plus sampled for review: {len(ocr_queue)}",
        f"Heading boundary/out-of-map items for review: {len(heading_queue)}",
        f"Confident page-range headings assigned: {counts['headings_assigned']}",
        "OCR corrections applied: 0 (review queue is pending manual verification)",
    ]
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(log_lines) + "\n")
    print(
        f"Created heading output for {len(output_records)} chunks; "
        f"OCR review items: {len(ocr_queue)}; heading review items: {len(heading_queue)}"
    )
    print(f"Review report: {report_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    input_path = args.input if args.input.is_absolute() else ROOT / args.input
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    try:
        return run(input_path, output_dir)
    except Exception as exc:
        print(f"Chunk review preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
