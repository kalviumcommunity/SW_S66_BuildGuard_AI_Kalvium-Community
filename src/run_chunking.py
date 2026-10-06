"""Clean the sample corpus and compare two chunking strategies."""


import json
from pathlib import Path

from chunk_documents import (
    FIXED_CHUNK_OVERLAP,
    FIXED_CHUNK_SIZE,
    RECURSIVE_CHUNK_OVERLAP,
    RECURSIVE_CHUNK_SIZE,
    chunk_documents,
    chunk_stats,
)
from document_loader import load_corpus
from text_cleaner import clean_document

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = PROJECT_ROOT / "examples" / "sample_corpus"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "chunking"
STATS_PATH = OUTPUT_DIR / "chunk_stats.txt"
SAMPLES_PATH = OUTPUT_DIR / "sample_chunks.json"


def main() -> int:
    loaded_documents = load_corpus(CORPUS_PATH)
    documents = [clean_document(document) for document in loaded_documents]
    recursive = chunk_documents(documents, strategy="recursive")
    fixed = chunk_documents(documents, strategy="fixed")
    stats = {"recursive": chunk_stats(recursive), "fixed": chunk_stats(fixed)}

    print(f"Loaded and cleaned documents: {len(documents)}")
    for strategy, values in stats.items():
        print(
            f"{strategy}: {values['count']} chunks, "
            f"average {values['average_length']:.1f}, "
            f"min {values['min_length']}, max {values['max_length']} characters"
        )
    print(
        "Chosen strategy: recursive, because its paragraph/line/sentence separators "
        f"preserve boundaries with {RECURSIVE_CHUNK_OVERLAP}-character overlap; "
        f"fixed uses {FIXED_CHUNK_SIZE}-character chunks with {FIXED_CHUNK_OVERLAP} overlap."
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stats_text = (
        "Recursive strategy\n"
        f"Configuration: chunk_size={RECURSIVE_CHUNK_SIZE}, "
        f"chunk_overlap={RECURSIVE_CHUNK_OVERLAP}\n"
        f"Stats: {stats['recursive']}\n\n"
        "Fixed strategy\n"
        f"Configuration: separator=' ', chunk_size={FIXED_CHUNK_SIZE}, "
        f"chunk_overlap={FIXED_CHUNK_OVERLAP}\n"
        f"Stats: {stats['fixed']}\n\n"
        "Chosen strategy: recursive for better boundary preservation and controlled overlap.\n"
    )
    STATS_PATH.write_text(stats_text, encoding="utf-8")

    samples = [
        {"strategy": "recursive", **chunk.__dict__} for chunk in recursive[:3]
    ] + [
        {"strategy": "fixed", **chunk.__dict__} for chunk in fixed[:3]
    ]
    SAMPLES_PATH.write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Stats: {STATS_PATH}")
    print(f"Samples: {SAMPLES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
