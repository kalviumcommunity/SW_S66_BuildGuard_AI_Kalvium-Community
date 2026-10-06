"""Load a local document corpus and write an intake manifest."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from document_loader import load_corpus

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = PROJECT_ROOT / "examples" / "sample_corpus"
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "corpus_intake" / "intake_manifest.json"


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
    for document in documents:
        sample = normalized_sample(document.text)
        entries.append(
            {
                "source": document.source,
                "text_length": len(document.text),
                "sample": sample,
            }
        )
        print(f"{document.source}: {len(document.text)} characters | {sample}")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Loaded: {len(documents)}")
    print(f"Skipped: {max(file_count - len(documents), 0)}")
    print(f"Manifest: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
