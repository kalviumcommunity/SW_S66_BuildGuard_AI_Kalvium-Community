"""Compare token chunking with and without overlap."""

import argparse
import itertools
import json
import os
from pathlib import Path

from document_loader import LoadedDocument, load_corpus
from text_cleaner import clean_document
from token_chunking import (
    TOKEN_CHUNK_SIZE,
    TOKEN_OVERLAP,
    get_tokenizer,
    token_chunk_documents,
    token_chunk_stats,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = PROJECT_ROOT / "examples" / "sample_corpus"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "token_chunking"
STATS_PATH = OUTPUT_DIR / "token_chunk_stats.txt"
SAMPLES_PATH = OUTPUT_DIR / "sample_token_chunks.json"
BOUNDARY_PATH = OUTPUT_DIR / "overlap_boundary_example.txt"


def _serialize_chunks(chunks) -> list[dict[str, object]]:
    return [
        {
            "text": chunk.text,
            "page_content": chunk.page_content,
            "metadata": chunk.metadata,
            "token_ids": list(chunk.token_ids),
        }
        for chunk in chunks[:3]
    ]


def _boundary_demo(model_name: str) -> str:
    encoding, _ = get_tokenizer(model_name)
    phrase = "urgent leave approval"
    phrase_ids = encoding.encode(f" {phrase}")
    selected: tuple[str, list, list] | None = None

    def contains_phrase(chunks) -> bool:
        return any(
            any(
                chunk.token_ids[index : index + len(phrase_ids)] == tuple(phrase_ids)
                for index in range(chunk.token_count - len(phrase_ids) + 1)
            )
            for chunk in chunks
        )
    for prefix_length in range(1, 80):
        prefix = " ".join("context" for _ in range(prefix_length))
        suffix = " ".join("followup" for _ in range(20))
        candidate = f"{prefix} {phrase} {suffix}"
        no_overlap = token_chunk_documents(
            [LoadedDocument("boundary.txt", candidate)],
            model_name=model_name,
            chunk_size=12,
            overlap=0,
        )
        overlap = token_chunk_documents(
            [LoadedDocument("boundary.txt", candidate)],
            model_name=model_name,
            chunk_size=12,
            overlap=4,
        )
        no_phrase = contains_phrase(no_overlap)
        overlap_phrase = contains_phrase(overlap)
        if not no_phrase and overlap_phrase:
            selected = (candidate, no_overlap, overlap)
            break

    if selected is None:
        raise RuntimeError("Could not construct a token-aware boundary demonstration")
    candidate, no_overlap, overlap = selected
    repeated_pair = next(
        (
            (previous, current)
            for previous, current in itertools.pairwise(overlap)
            if previous.token_ids[-4:] == current.token_ids[:4]
        ),
        None,
    )
    repeated = repeated_pair is not None
    repeated_ids = list(repeated_pair[0].token_ids[-4:]) if repeated_pair else []
    next_ids = list(repeated_pair[1].token_ids[:4]) if repeated_pair else []
    return (
        "Deterministic boundary demonstration\n"
        f"Boundary phrase: {phrase}\n"
        f"Candidate text: {candidate}\n\n"
        f"No-overlap full phrase in one chunk: {contains_phrase(no_overlap)}\n"
        f"No-overlap chunks: {[chunk.text for chunk in no_overlap]}\n\n"
        f"Overlap full phrase in one chunk: {contains_phrase(overlap)}\n"
        f"Overlap chunks: {[chunk.text for chunk in overlap]}\n\n"
        f"Exact repeated token overlap (4 IDs): {repeated}\n"
        f"Previous chunk final IDs: {repeated_ids}\n"
        f"Next chunk initial IDs: {next_ids}\n"
        f"Tokenizer encoding: {encoding.name}\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("CHAT_MODEL"))
    args = parser.parse_args()
    encoding, encoding_name = get_tokenizer(args.model)

    documents = [clean_document(document) for document in load_corpus(CORPUS_PATH)]
    with_overlap = token_chunk_documents(
        documents,
        model_name=args.model,
        chunk_size=TOKEN_CHUNK_SIZE,
        overlap=TOKEN_OVERLAP,
    )
    no_overlap = token_chunk_documents(
        documents,
        model_name=args.model,
        chunk_size=TOKEN_CHUNK_SIZE,
        overlap=0,
    )
    stats = {
        "overlap_32": token_chunk_stats(with_overlap),
        "overlap_0": token_chunk_stats(no_overlap),
    }

    print(f"Tokenizer: {encoding_name} ({encoding.name})")
    print(f"Chunk size: {TOKEN_CHUNK_SIZE} tokens")
    print(f"Overlap 32: {stats['overlap_32']}")
    print(f"Overlap 0: {stats['overlap_0']}")
    boundary = _boundary_demo(args.model or encoding_name)
    print(boundary)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(
        f"Tokenizer: {encoding_name}\n"
        f"Chunk size: {TOKEN_CHUNK_SIZE}\n"
        f"Overlap 32: {stats['overlap_32']}\n"
        f"Overlap 0: {stats['overlap_0']}\n",
        encoding="utf-8",
    )
    samples = {
        "overlap_32": _serialize_chunks(with_overlap),
        "overlap_0": _serialize_chunks(no_overlap),
    }
    SAMPLES_PATH.write_text(json.dumps(samples, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    BOUNDARY_PATH.write_text(boundary, encoding="utf-8")
    print(f"Stats: {STATS_PATH}")
    print(f"Samples: {SAMPLES_PATH}")
    print(f"Boundary: {BOUNDARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
