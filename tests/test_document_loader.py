from pathlib import Path

from src.document_loader import load_corpus, load_document


def test_txt_and_md_loading(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("Plain text", encoding="utf-8")
    (tmp_path / "guide.md").write_text("# Guide\nMarkdown text", encoding="utf-8")

    documents = load_corpus(tmp_path)

    assert {document.source for document in documents} == {"guide.md", "note.txt"}
    assert next(document for document in documents if document.source == "note.txt").text == "Plain text"


def test_html_becomes_plain_text(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "page.html"
    path.parent.mkdir()
    path.write_text("<h1>Title</h1><p>Hello <b>staff</b>.</p>", encoding="utf-8")

    document = load_document(path, tmp_path)

    assert document is not None
    assert document.source == "nested/page.html"
    assert document.text == "Title Hello staff."


def test_source_identity_includes_nested_relative_path(tmp_path: Path) -> None:
    path = tmp_path / "department" / "leave.txt"
    path.parent.mkdir()
    path.write_text("Leave process", encoding="utf-8")

    document = load_document(path, tmp_path)

    assert document is not None
    assert document.source == "department/leave.txt"


def test_unsupported_missing_and_unreadable_inputs_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "image.csv").write_text("not supported", encoding="utf-8")
    (tmp_path / "broken.txt").write_bytes(b"\xff\xfe")

    assert load_document(tmp_path / "missing.txt", tmp_path) is None
    assert load_document(tmp_path / "image.csv", tmp_path) is None
    assert load_document(tmp_path / "broken.txt", tmp_path) is None


def test_malformed_pdf_is_skipped_when_pdf_support_is_available(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not a PDF")

    assert load_document(path, tmp_path) is None


def test_corpus_continues_after_bad_file(tmp_path: Path) -> None:
    (tmp_path / "good.txt").write_text("Loaded", encoding="utf-8")
    (tmp_path / "bad.bin").write_bytes(b"unsupported")
    (tmp_path / "bad.txt").write_bytes(b"\xff")

    documents = load_corpus(tmp_path)

    assert [document.source for document in documents] == ["good.txt"]
