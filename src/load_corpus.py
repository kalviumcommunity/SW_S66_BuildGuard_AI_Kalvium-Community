"""Load a local document corpus and write an intake manifest."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from document_loader import load_corpus
from text_cleaner import clean_document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = PROJECT_ROOT / "examples" / "sample_corpus"
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "corpus_intake" / "intake_manifest.json"
CLEANING_EXAMPLES_PATH = PROJECT_ROOT / "outputs" / "corpus_cleaning" / "cleaning_examples.txt"


def normalized_sample(text: str, limit: int = 160) -> str:
    return " ".join(text.split())[:limit]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=DEFAULT_CORPUS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    documents = load_corpus(args.directory)
    file_count = (
        sum(1 for path in args.directory.rglob("*") if path.is_file())
        if args.directory.is_dir()
        else 0
    )
    entries = []
    cleaning_examples: list[str] = []
    for document in documents:
        cleaned = clean_document(document)
        before_sample = normalized_sample(document.text)
        after_sample = normalized_sample(cleaned.text)
        entries.append(
            {
                "source": cleaned.source,
                "text_length": len(cleaned.text),
                "sample": after_sample,
            }
        )
        cleaning_examples.append(
            f"Source: {document.source}\n"
            f"Before ({len(document.text)} characters): {before_sample}\n"
            f"After ({len(cleaned.text)} characters): {after_sample}\n"
        )
        print(
            f"{cleaned.source}: {len(cleaned.text)} characters | "
            f"before: {before_sample} | after: {after_sample}"
        )

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    CLEANING_EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLEANING_EXAMPLES_PATH.write_text("\n".join(cleaning_examples), encoding="utf-8")
    print(f"Loaded: {len(documents)}")
    print(f"Skipped: {max(file_count - len(documents), 0)}")
    print(f"Manifest: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
