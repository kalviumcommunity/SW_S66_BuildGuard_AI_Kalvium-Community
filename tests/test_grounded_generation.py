"""Unit tests for grounded answer generation, source accuracy, and fallback."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.grounded_generation import (
    INSUFFICIENT_CONTEXT_FALLBACK,
    ComparisonResult,
    GenerationResult,
    SourceAccuracyReport,
    assemble_context_block,
    generate_grounded_answer,
    generate_grounded_answers_markdown,
    generate_without_retrieval,
    get_grounding_sample_chunks,
    run_with_and_without_comparison,
    save_grounded_results,
    verify_source_accuracy,
)


class TestSourceAccuracyVerification(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = get_grounding_sample_chunks()

    def test_verify_source_accuracy_fully_grounded(self) -> None:
        answer = (
            "Under Clause 6.1.1.2 [1], walls separating a boiler room must provide a minimum fire "
            "resistance rating of 120 min. Openings must be protected by self-closing fire doors [Volume 1, PDF p. 314-315]."
        )
        report = verify_source_accuracy(answer, self.chunks)
        self.assertTrue(report.is_grounded)
        self.assertGreaterEqual(report.grounding_score, 0.70)
        self.assertIn("[1]", report.valid_citations)
        self.assertEqual(len(report.invalid_citations), 0)
        self.assertIn("120 min", report.verified_key_terms)
        self.assertEqual(report.verdict, "VERIFIED_GROUNDED")

    def test_verify_source_accuracy_hallucinated_citations_and_terms(self) -> None:
        # Citations point to chunk 99 (doesn't exist) and mentions 480 min (not in text)
        answer = "Boiler room walls require 480 min rating according to [99]."
        report = verify_source_accuracy(answer, self.chunks)
        self.assertFalse(report.is_grounded)
        self.assertLess(report.grounding_score, 0.50)
        self.assertGreater(len(report.invalid_citations), 0)
        self.assertIn("UNGROUNDED", report.verdict)

    def test_verify_source_accuracy_fallback(self) -> None:
        report = verify_source_accuracy(INSUFFICIENT_CONTEXT_FALLBACK, [], is_fallback=True)
        self.assertTrue(report.is_grounded)
        self.assertEqual(report.grounding_score, 1.0)
        self.assertEqual(report.verdict, "FALLBACK_TRIGGERED")


class TestGroundedGeneration(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = get_grounding_sample_chunks()
        self.query = "What is the fire resistance rating for boiler room walls?"

    def test_generate_grounded_answer_offline(self) -> None:
        res = generate_grounded_answer(self.query, self.chunks, offline=True)
        self.assertIsInstance(res, GenerationResult)
        self.assertEqual(res.mode, "grounded_rag")
        self.assertFalse(res.is_fallback)
        self.assertIn("[1]", res.answer)
        self.assertIn("120", res.answer)
        self.assertIsNotNone(res.accuracy_report)
        self.assertTrue(res.accuracy_report.is_grounded)

    def test_generate_grounded_answer_mocked_llm(self) -> None:
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "According to Clause 6.1.1.2 [1], walls separating a boiler room require a 120 min "
            "fire resistance rating [Volume 1, PDF p. 314-315]."
        )
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("src.assistant.api_client.get_chat_client", return_value=mock_client), \
             patch("src.assistant.api_client.get_chat_model", return_value="gemini-2.5-flash"):
            res = generate_grounded_answer(self.query, self.chunks, offline=False)
            self.assertEqual(res.mode, "grounded_rag")
            self.assertEqual(res.model, "gemini-2.5-flash")
            self.assertIn("120 min", res.answer)
            self.assertTrue(res.accuracy_report.is_grounded)

    def test_missing_context_fallback_empty_chunks(self) -> None:
        res = generate_grounded_answer("Any query", [], offline=True)
        self.assertEqual(res.mode, "fallback")
        self.assertTrue(res.is_fallback)
        self.assertEqual(res.answer, INSUFFICIENT_CONTEXT_FALLBACK)
        self.assertEqual(res.accuracy_report.verdict, "FALLBACK_TRIGGERED")

    def test_missing_context_fallback_low_relevance_threshold(self) -> None:
        # Chunks with similarity lower than relevance_threshold (e.g., 0.30 vs 0.55)
        low_sim_chunks = [
            {"id": "chunk_x", "text": "Unrelated topic", "cosine_similarity": 0.30, "metadata": {}}
        ]
        res = generate_grounded_answer(
            "Out of domain question", low_sim_chunks, offline=True, relevance_threshold=0.55
        )
        self.assertTrue(res.is_fallback)
        self.assertEqual(res.answer, INSUFFICIENT_CONTEXT_FALLBACK)


class TestComparisonWithAndWithoutRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = get_grounding_sample_chunks()
        self.query = "What is the fire resistance rating for boiler room walls?"

    def test_generate_without_retrieval(self) -> None:
        res = generate_without_retrieval(self.query, offline=True)
        self.assertEqual(res.mode, "unanchored_baseline")
        self.assertFalse(res.is_fallback)
        self.assertEqual(len(res.supporting_chunks), 0)
        self.assertEqual(res.accuracy_report.verdict, "UNGROUNDED")

    def test_run_with_and_without_comparison(self) -> None:
        comp = run_with_and_without_comparison(self.query, self.chunks, offline=True)
        self.assertIsInstance(comp, ComparisonResult)
        self.assertEqual(comp.query, self.query)
        self.assertEqual(comp.grounded_result.mode, "grounded_rag")
        self.assertEqual(comp.unanchored_result.mode, "unanchored_baseline")
        self.assertGreater(len(comp.grounding_differences), 0)
        self.assertTrue(comp.grounded_result.accuracy_report.is_grounded)
        self.assertFalse(comp.unanchored_result.accuracy_report.is_grounded)


class TestPersistenceAndReports(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = get_grounding_sample_chunks()
        self.query = "What is the fire resistance rating for boiler room walls?"

    def test_save_grounded_results(self) -> None:
        grounded = generate_grounded_answer(self.query, self.chunks, offline=True)
        fallback = generate_grounded_answer("Non-NBC question", [], offline=True)
        comp = run_with_and_without_comparison(self.query, self.chunks, offline=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            json_path, md_path = save_grounded_results(
                grounded, fallback, comp, output_dir=out_dir
            )

            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())

            # Check JSON
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertIn("sample_grounded_answer", data)
            self.assertIn("sample_fallback", data)
            self.assertIn("retrieval_comparison", data)
            self.assertEqual(data["sample_grounded_answer"]["mode"], "grounded_rag")
            self.assertEqual(data["sample_fallback"]["mode"], "fallback")

            # Check Markdown
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("# Grounded Answer Generation & Retrieval Comparison Report", md_text)
            self.assertIn("With vs. Without Retrieval Comparison", md_text)
            self.assertIn("Source Accuracy Audit", md_text)
            self.assertIn("Missing-Context Fallback Evaluation", md_text)


if __name__ == "__main__":
    unittest.main()
