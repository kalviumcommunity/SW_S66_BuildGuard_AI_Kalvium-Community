from types import SimpleNamespace

from src.document_loader import LoadedDocument
from src.embed_chunks import (
    build_embedding_records,
    format_summary,
    run_embedding_pipeline,
    validate_vector_dimensions,
)
from src.token_chunking import token_chunk_documents


def test_records_preserve_chunk_text_metadata_and_full_vectors() -> None:
    chunks = token_chunk_documents([LoadedDocument("guide.txt", "A workplace procedure. " * 30)])
    vectors = [[float(index), 1.0, 2.0] for index, _ in enumerate(chunks)]

    records, dimension = build_embedding_records(chunks, vectors, expected_dimension=3)

    assert dimension == 3
    assert len(records) == len(chunks)
    for chunk, record, vector in zip(chunks, records, vectors):
        assert record["embedding"] == vector
        assert record["vector_preview"] == vector[:5]
        assert record["vector_length"] == 3
        assert record["text"] == chunk.text
        assert record["metadata"] == chunk.metadata


def test_vector_dimension_validation_and_count_mismatch() -> None:
    assert validate_vector_dimensions([[1.0, 2.0], [3.0, 4.0]], expected_dimension=2) == 2
    for invalid in ([], [[1.0], [1.0, 2.0]]):
        try:
            validate_vector_dimensions(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid vector dimensions were accepted")

    chunks = token_chunk_documents([LoadedDocument("guide.txt", "Text " * 1000)])
    try:
        build_embedding_records(chunks, [[1.0, 2.0]])
    except ValueError as error:
        assert "number of vectors" in str(error)
    else:
        raise AssertionError("vector/chunk count mismatch was accepted")


def test_injected_model_pipeline_and_summary_are_deterministic() -> None:
    chunks = token_chunk_documents([LoadedDocument("guide.txt", "Text " * 100)])
    fake_model = SimpleNamespace(
        embed_documents=lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
    )

    records, dimension = run_embedding_pipeline(fake_model, chunks, "fake-model", expected_dimension=3)
    summary = format_summary("fake-model", records, dimension)

    assert len(records) == len(chunks)
    assert dimension == 3
    assert "Chunks embedded:" in summary
    assert "Vector dimension: 3" in summary
    assert "local; no API call" in summary
