"""Validate complete, metadata-aware ingestion of a local corpus."""

import argparse
import json
import os
from pathlib import Path

from ingestion_pipeline import ingest_corpus

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ingestion"
SUMMARY_JSON = OUTPUT_DIR / "ingestion_summary.json"
SUMMARY_TEXT = OUTPUT_DIR / "ingestion_summary.txt"
SAMPLE_CHUNKS = OUTPUT_DIR / "sample_ingested_chunks.json"


def _summary_text(report: dict) -> str:
    lines = [
        f"Corpus: {report['corpus_directory']}",
        f"Total sources: {report['total_sources']}",
        f"Ingested: {report['ingested']}",
        f"Skipped/failed: {report['skipped']}",
        f"Total chunks: {report['total_chunks']}",
        f"Completeness check: {'PASS' if report['complete'] else 'FAIL'}",
    ]
    if report.get("validation_error"):
        lines.append(f"Validation error: {report['validation_error']}")
    lines.append("")
    lines.append("Source statuses:")
    for entry in report["sources"]:
        detail = entry.get("reason") or f"{entry['chunk_count']} chunks"
        lines.append(f"- {entry['source']}: {entry['status']} ({detail})")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", nargs="?", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--model", default=os.getenv("CHAT_MODEL"))
    args = parser.parse_args()

    report = ingest_corpus(args.corpus, model_name=args.model)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    SUMMARY_TEXT.write_text(_summary_text(report), encoding="utf-8")
    SAMPLE_CHUNKS.write_text(
        json.dumps(report["chunks"][:5], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Total sources: {report['total_sources']}")
    print(f"Ingested: {report['ingested']}")
    print(f"Skipped/failed: {report['skipped']}")
    print(f"Total chunks: {report['total_chunks']}")
    print(f"Completeness check: {'PASS' if report['complete'] else 'FAIL'}")
    if report.get("validation_error"):
        print(f"Reason: {report['validation_error']}")
    print(f"Summary JSON: {SUMMARY_JSON}")
    print(f"Summary text: {SUMMARY_TEXT}")
    print(f"Sample chunks: {SAMPLE_CHUNKS}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
