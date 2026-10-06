from itertools import pairwise

from src.chunk_documents import (
    RECURSIVE_CHUNK_OVERLAP,
    RECURSIVE_CHUNK_SIZE,
    RECURSIVE_SEPARATORS,
    chunk_documents,
    chunk_stats,
)
from src.document_loader import LoadedDocument


def sample_documents() -> list[LoadedDocument]:
    text = "\n\n".join(
        f"Paragraph {index}. This sentence contains enough repeated content for chunking."
        for index in range(30)
    )
    return [LoadedDocument(source="nested/guide.txt", text=text)]


def test_both_strategies_create_nonempty_chunks_with_source_ids() -> None:
    documents = sample_documents()
    for strategy in ("recursive", "fixed"):
        chunks = chunk_documents(documents, strategy=strategy)
        assert chunks
        assert all(chunk.source == "nested/guide.txt" for chunk in chunks)
        assert all(chunk.text.strip() for chunk in chunks)
        assert [chunk.chunk_id for chunk in chunks] == list(range(len(chunks)))


def test_chunk_stats_are_correct() -> None:
    chunks = chunk_documents(sample_documents(), strategy="recursive")
    stats = chunk_stats(chunks)
    lengths = [len(chunk.text) for chunk in chunks]
    assert stats["count"] == len(chunks)
    assert stats["min_length"] == min(lengths)
    assert stats["max_length"] == max(lengths)
    assert stats["average_length"] == sum(lengths) / len(lengths)


def test_recursive_configuration_and_overlap_are_reasonable() -> None:
    assert RECURSIVE_CHUNK_SIZE == 500
    assert RECURSIVE_CHUNK_OVERLAP == 75
    assert RECURSIVE_SEPARATORS == ["\n\n", "\n", ". ", " ", ""]
    chunks = chunk_documents(sample_documents(), strategy="recursive")
    assert all(len(chunk.text) <= RECURSIVE_CHUNK_SIZE for chunk in chunks)
    assert any(
        previous.text[-20:] in current.text
        for previous, current in pairwise(chunks)
    )


def test_chunk_ids_restart_for_each_source() -> None:
    documents = [
        LoadedDocument(source="a.txt", text="A " * 400),
        LoadedDocument(source="b.txt", text="B " * 400),
    ]
    chunks = chunk_documents(documents, strategy="fixed")
    assert [chunk.chunk_id for chunk in chunks if chunk.source == "a.txt"] == [0, 1]
    assert [chunk.chunk_id for chunk in chunks if chunk.source == "b.txt"] == [0, 1]
