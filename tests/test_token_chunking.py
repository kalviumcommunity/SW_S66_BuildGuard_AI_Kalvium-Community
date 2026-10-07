import itertools

import tiktoken

from src.document_loader import LoadedDocument
from src.token_chunking import get_tokenizer, token_chunk_documents, token_chunk_stats


def token_document() -> list[LoadedDocument]:
    return [LoadedDocument(source="token-guide.txt", text=("boundary phrase and workplace guidance. " * 300))]


def test_token_size_limits_and_metadata() -> None:
    chunks = token_chunk_documents(token_document(), chunk_size=32, overlap=8)
    assert chunks
    assert all(0 < chunk.token_count <= 32 for chunk in chunks)
    assert all(chunk.source == "token-guide.txt" for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.end_token - chunk.start_token == chunk.token_count for chunk in chunks)
    assert all(chunk.metadata["page"] == 1 for chunk in chunks)


def test_exact_token_overlap() -> None:
    chunks = token_chunk_documents(token_document(), chunk_size=32, overlap=8)
    for previous, current in itertools.pairwise(chunks):
        assert previous.token_ids[-8:] == current.token_ids[:8]


def test_zero_overlap() -> None:
    chunks = token_chunk_documents(token_document(), chunk_size=32, overlap=0)
    for previous, current in itertools.pairwise(chunks):
        assert previous.token_ids[-1:] != current.token_ids[:1]


def test_stats_and_unknown_model_fallback() -> None:
    chunks = token_chunk_documents(token_document(), model_name="unknown-model", chunk_size=32, overlap=0)
    stats = token_chunk_stats(chunks)
    assert stats["count"] == len(chunks)
    assert stats["average_tokens"] == sum(chunk.token_count for chunk in chunks) / len(chunks)
    encoding, name = get_tokenizer("unknown-model")
    assert name == "cl100k_base"
    assert encoding.name == "cl100k_base"


def test_boundary_evidence_is_traceable_to_token_ids() -> None:
    encoding = tiktoken.get_encoding("cl100k_base")
    document = [LoadedDocument(source="boundary.txt", text="prefix " * 20 + "UNIQUE_BOUNDARY_PHRASE " + "suffix " * 20)]
    chunks = token_chunk_documents(document, chunk_size=12, overlap=4)
    assert len(chunks) > 1
    assert any("UNIQUE_BOUNDARY_PHRASE" in chunk.text for chunk in chunks)
    assert any(
        previous.token_ids[-4:] == current.token_ids[:4]
        for previous, current in itertools.pairwise(chunks)
    )
    assert encoding.encode(chunks[0].text)  # The sample is nonempty and tokenizable.


def test_invalid_token_parameters() -> None:
    try:
        token_chunk_documents(token_document(), chunk_size=0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero chunk size was accepted")

    try:
        token_chunk_documents(token_document(), chunk_size=8, overlap=8)
    except ValueError:
        pass
    else:
        raise AssertionError("overlap equal to chunk size was accepted")
