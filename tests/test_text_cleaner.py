import importlib.util
import unittest

if importlib.util.find_spec("ftfy") is None:
    raise unittest.SkipTest("ftfy is not installed")

from src.document_loader import LoadedDocument
from src.text_cleaner import clean_document, clean_text


def test_ftfy_repairs_mojibake() -> None:
    assert "café" in clean_text("cafÃ©")


def test_nfkc_normalizes_compatibility_characters() -> None:
    assert "office" in clean_text("office ﬁle")


def test_whitespace_and_page_markers_are_cleaned() -> None:
    cleaned = clean_text("Page 1 of 2\n\n  Too   much   space.  \n\nPage 2 of 2")
    assert cleaned == "Too much space."


def test_repeated_boilerplate_is_removed() -> None:
    cleaned = clean_text("STAFF HANDBOOK\n\nBody text.\n\nSTAFF HANDBOOK")
    assert cleaned == "Body text."


def test_paragraphs_and_content_are_preserved() -> None:
    cleaned = clean_text("First broken\nline.\n\nSecond paragraph remains.")
    assert cleaned == "First broken line.\n\nSecond paragraph remains."


def test_clean_document_preserves_source() -> None:
    document = LoadedDocument(source="nested/guide.txt", text="cafÃ©")
    cleaned = clean_document(document)
    assert cleaned.source == document.source
    assert cleaned.text == "café"
