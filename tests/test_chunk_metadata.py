from src.chunk_documents import chunk_documents, to_langchain_documents
from src.document_loader import LoadedDocument

METADATA_KEYS = {
    "source",
    "section",
    "page",
    "chunk_index",
    "start_char",
    "end_char",
}


def test_metadata_is_consistent_and_traceable() -> None:
    text = "# Leave Procedure\nRequest leave through the manager review process.\n\n"
    text += "Additional details should include dates and a handoff summary. " * 12
    documents = [LoadedDocument(source="leave.txt", text=text)]

    chunks = chunk_documents(documents, strategy="recursive")

    assert chunks
    assert all(set(chunk.metadata) == METADATA_KEYS for chunk in chunks)
    assert all(chunk.source == "leave.txt" for chunk in chunks)
    assert all(chunk.page == 1 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(0 <= chunk.start_char < chunk.end_char <= len(text) for chunk in chunks)
    assert all(text[chunk.start_char : chunk.end_char] == chunk.text for chunk in chunks)


def test_markdown_heading_section_and_general_fallback() -> None:
    documents = [
        LoadedDocument(
            source="sections.md",
            text="# Leave\nLeave content.\n\n## Safety\nSafety content. " * 20,
        ),
        LoadedDocument(source="plain.txt", text="Plain content without a heading. " * 20),
    ]

    chunks = chunk_documents(documents, strategy="recursive")

    leave_chunks = [chunk for chunk in chunks if chunk.source == "sections.md"]
    plain_chunks = [chunk for chunk in chunks if chunk.source == "plain.txt"]
    assert any(chunk.section in {"Leave", "Safety"} for chunk in leave_chunks)
    assert all(chunk.section == "General" for chunk in plain_chunks)


def test_langchain_document_conversion_uses_scalar_metadata() -> None:
    chunks = chunk_documents(
        [LoadedDocument(source="guide.md", text="# Guide\nUseful content. " * 30)],
        strategy="fixed",
    )

    langchain_documents = to_langchain_documents(chunks)

    assert len(langchain_documents) == len(chunks)
    for chunk, document in zip(chunks, langchain_documents):
        assert document.page_content == chunk.text
        assert document.metadata == chunk.metadata
        assert all(isinstance(value, (str, int)) for value in document.metadata.values())
