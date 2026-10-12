"""Conservatively clean and validate generated NBC text chunks.

The source JSONL is preserved. Cleaned records and diagnostics are written separately.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import tiktoken

ROOT = Path(__file__).resolve().parents[2]
ENCODING = tiktoken.get_encoding("cl100k_base")

DEFAULT_INPUT = ROOT / "outputs" / "chunks" / "text_chunks.jsonl"
DEFAULT_CLEAN_OUTPUT_DIR = ROOT / "outputs" / "cleaned_chunks"
PIPELINE_VERSION = "1.0"
SHORT_CHUNK_TOKENS = 10
HEADING_SPAN_WARNING_PAGES = 50
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
def _clean_text(text: str) -> tuple[str, int]:
    """Normalize encoding and whitespace without rewriting words or symbols."""
    text = unicodedata.normalize("NFC", text)
    text, removed = CONTROL_CHARS.subn(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, removed


def _split_at_word_boundaries(text: str, max_tokens: int) -> list[str] | None:
    """Split oversized text without dropping words or exceeding the token cap."""
    words = text.split(" ")
    parts: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(ENCODING.encode(candidate)) <= max_tokens:
            current = candidate
            continue
        if current:
            parts.append(current)
        if len(ENCODING.encode(word)) > max_tokens:
            return None
        current = word
    if current:
        parts.append(current)
    return parts


def _sample_issue(
    samples: dict[str, list[str]], category: str, chunk_id: str, limit: int = 20
) -> None:
    values = samples.setdefault(category, [])
    if len(values) < limit:
        values.append(chunk_id)


def _record_errors(
    record: Any, line_number: int, seen_ids: set[str]
) -> list[str]:
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["record must be a JSON object"]

    chunk_id = record.get("id")
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        errors.append("missing or invalid id")
    elif chunk_id in seen_ids:
        errors.append(f"duplicate id {chunk_id!r}")
    else:
        seen_ids.add(chunk_id)

    if not isinstance(record.get("text"), str):
        errors.append("text must be a string")
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        errors.append("metadata must be an object")
        return errors

    if metadata.get("modality") != "text":
        errors.append("metadata.modality must be 'text'")
    if not isinstance(metadata.get("source"), str) or not metadata["source"].strip():
        errors.append("metadata.source must be a nonempty string")
    volume = metadata.get("volume")
    if not isinstance(volume, int) or isinstance(volume, bool) or volume < 1:
        errors.append("metadata.volume must be a positive integer")

    for key in ("page_start", "page_end"):
        value = metadata.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            errors.append(f"metadata.{key} must be a positive integer")
    if all(isinstance(metadata.get(key), int) for key in ("page_start", "page_end")):
        if metadata["page_start"] > metadata["page_end"]:
            errors.append("metadata.page_start must not exceed page_end")

    token_count = metadata.get("token_count")
    if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count < 0:
        errors.append("metadata.token_count must be a nonnegative integer")

    if errors:
        return [f"line {line_number}: {message}" for message in errors]
    return errors


def _resolve_from_root(value: Path) -> Path:
    return value if value.is_absolute() else ROOT / value


def run(input_path: Path, output_dir: Path, max_tokens: int) -> int:
    input_path = input_path.resolve()
    output_dir = output_dir.resolve()
    output_path = output_dir / "text_chunks.cleaned.jsonl"
    log_path = output_dir / "cleaning_validation.log"
    report_path = output_dir / "validation_report.json"
    temp_path = output_dir / "text_chunks.cleaned.jsonl.tmp"

    if input_path == output_path:
        raise ValueError("Input and output paths must be different; raw input is preserved.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    if max_tokens < 1:
        raise ValueError("--max-tokens must be a positive integer")

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    samples: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    heading_pages: dict[str, list[int]] = {}
    hard_errors = 0
    warnings = 0
    output_ids: set[str] = set()
    total_tokens_before = 0
    total_tokens_after = 0

    def warn(category: str, chunk_id: str) -> None:
        nonlocal warnings
        warnings += 1
        counts[category] += 1
        _sample_issue(samples, category, chunk_id)

    def error(category: str, message: str) -> None:
        nonlocal hard_errors
        hard_errors += 1
        counts[category] += 1
        _sample_issue(samples, category, message)

    try:
        with input_path.open("r", encoding="utf-8-sig") as source, temp_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as cleaned_file:
            for line_number, line in enumerate(source, 1):
                counts["input_lines"] += 1
                if not line.strip():
                    error("blank_lines", f"line {line_number}")
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    error("invalid_json", f"line {line_number}: {exc.msg}")
                    continue

                record_errors = _record_errors(record, line_number, seen_ids)
                if record_errors:
                    for message in record_errors:
                        error("invalid_records", message)
                    continue

                chunk_id = record["id"]
                metadata = record["metadata"]
                original_text = record["text"]
                cleaned_text, removed_controls = _clean_text(original_text)
                if removed_controls:
                    counts["control_characters_removed"] += removed_controls
                    warn("records_with_control_characters", chunk_id)
                if not cleaned_text:
                    error("empty_cleaned_text", f"line {line_number}: {chunk_id}")
                    continue

                original_count = len(ENCODING.encode(original_text))
                cleaned_count = len(ENCODING.encode(cleaned_text))
                total_tokens_before += original_count
                if metadata["token_count"] != original_count:
                    # The chunker sums tokenized sentence lengths before joining
                    # them. Tokenization of the joined text can differ at boundaries.
                    counts["source_token_count_recalculations"] += 1

                parts = [cleaned_text]
                if cleaned_count > max_tokens:
                    parts = _split_at_word_boundaries(cleaned_text, max_tokens) or []
                    if not parts:
                        error(
                            "unsplittable_over_limit_chunks",
                            f"{chunk_id} ({cleaned_count} tokens)",
                        )
                        continue
                    warn("split_over_limit_chunks", chunk_id)

                for part_index, part_text in enumerate(parts, 1):
                    part_tokens = len(ENCODING.encode(part_text))
                    if part_tokens > max_tokens:
                        error(
                            "chunks_over_token_limit",
                            f"{chunk_id} part {part_index} ({part_tokens} tokens)",
                        )
                        continue

                    part_record = dict(record)
                    part_metadata = dict(metadata)
                    part_record["metadata"] = part_metadata
                    part_record["text"] = part_text
                    part_metadata["token_count"] = part_tokens
                    if len(parts) > 1:
                        part_record["id"] = (
                            chunk_id
                            if part_index == 1
                            else f"{chunk_id}-part-{part_index:02d}"
                        )
                        part_metadata["source_chunk_id"] = chunk_id
                        part_metadata["subchunk_index"] = part_index
                        part_metadata["subchunk_count"] = len(parts)

                    part_id = part_record["id"]
                    if part_id in output_ids:
                        error("duplicate_output_ids", part_id)
                        continue
                    output_ids.add(part_id)

                    if part_tokens <= SHORT_CHUNK_TOKENS:
                        warn("short_chunks", part_id)
                    total_tokens_after += part_tokens
                    cleaned_file.write(
                        json.dumps(part_record, ensure_ascii=False) + "\n"
                    )
                    counts["cleaned_records"] += 1
                    if len(parts) > 1:
                        counts["split_output_records"] += 1

                if "\ufffd" in cleaned_text:
                    warn("replacement_characters", chunk_id)

                heading = metadata.get("heading")
                start_page = metadata["page_start"]
                end_page = metadata["page_end"]
                if isinstance(heading, str) and heading.strip():
                    span = heading_pages.setdefault(heading, [start_page, end_page])
                    span[0] = min(span[0], start_page)
                    span[1] = max(span[1], end_page)

        stale_heading_candidates = [
            {"heading": heading, "page_start": pages[0], "page_end": pages[1]}
            for heading, pages in heading_pages.items()
            if pages[1] - pages[0] + 1 > HEADING_SPAN_WARNING_PAGES
        ]
        counts["distinct_nonempty_headings"] = len(heading_pages)
        counts["possible_stale_headings"] = len(stale_heading_candidates)

        if hard_errors:
            temp_path.unlink(missing_ok=True)
            status = "failed_validation"
        else:
            temp_path.replace(output_path)
            status = "completed_with_warnings" if warnings or stale_heading_candidates else "completed"

        report = {
            "pipeline": "clean_validate_text_chunks",
            "version": PIPELINE_VERSION,
            "status": status,
            "input": str(input_path),
            "output": str(output_path) if not hard_errors else None,
            "max_tokens": max_tokens,
            "tokens_before": total_tokens_before,
            "tokens_after": total_tokens_after,
            "counts": dict(counts),
            "warnings": warnings + len(stale_heading_candidates),
            "errors": hard_errors,
            "warning_samples": samples,
            "possible_stale_headings": stale_heading_candidates[:20],
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _append_log(log_path, report)
        print(
            f"{status}: {counts['cleaned_records']} records; "
            f"{warnings} record warnings; {hard_errors} errors"
        )
        print(f"Report: {report_path}")
        if not hard_errors:
            print(f"Cleaned JSONL: {output_path}")
        return 1 if hard_errors else 0
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _append_log(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "",
        f"=== Run {datetime.now().astimezone().isoformat(timespec='seconds')} ===",
        f"Status: {report['status']}",
        f"Input: {report['input']}",
        f"Output: {report['output'] or '(not written; validation failed)'}",
        f"Records read: {report['counts'].get('input_lines', 0)}",
        f"Records written: {report['counts'].get('cleaned_records', 0)}",
        f"Oversized source chunks split: {report['counts'].get('split_over_limit_chunks', 0)}",
        f"Records emitted from splits: {report['counts'].get('split_output_records', 0)}",
        f"Tokens before/after cleanup: {report['tokens_before']} / {report['tokens_after']}",
        f"Warnings: {report['warnings']}; errors: {report['errors']}",
        f"Short chunks (<= {SHORT_CHUNK_TOKENS} tokens): {report['counts'].get('short_chunks', 0)}",
        f"Replacement-character chunks: {report['counts'].get('replacement_characters', 0)}",
        f"Stored token counts recalculated: {report['counts'].get('source_token_count_recalculations', 0)}",
        f"Distinct nonempty headings: {report['counts'].get('distinct_nonempty_headings', 0)}",
        "Possible stale headings:",
    ]
    candidates = report["possible_stale_headings"]
    if candidates:
        lines.extend(
            f"  pages {item['page_start']}-{item['page_end']}: {item['heading']}"
            for item in candidates
        )
    else:
        lines.append("  none detected")
    if report["warning_samples"]:
        lines.append("Diagnostic sample references (up to 20 per category):")
        for category, chunk_ids in sorted(report["warning_samples"].items()):
            lines.append(f"  {category}: {', '.join(chunk_ids)}")
    with path.open("a", encoding="utf-8", newline="\n") as log:
        log.write("\n".join(lines) + "\n")



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source text-chunk JSONL")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder for cleaned chunks, report, and log")
    parser.add_argument("--max-tokens", type=int, default=500, help="Maximum tokens per cleaned chunk")
    args = parser.parse_args()
    try:
        return run_text_validation(_resolve_from_root(args.input), _resolve_from_root(args.output_dir), args.max_tokens)
    except Exception as exc:
        log_path = _resolve_from_root(args.output_dir) / "cleaning_validation.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8", newline="\n") as log:
                log.write(f"\n=== Run {datetime.now().astimezone().isoformat(timespec='seconds')} ===\nStatus: failed before validation completed\nError: {type(exc).__name__}: {exc}\n")
        except OSError:
            pass
        print(f"Cleaning/validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
