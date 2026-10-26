"""Unit tests for RAG Evaluation & Answer Quality Scoring Engine."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.rag_evaluation import (
    DEFAULT_TEST_SET_PATH,
    CitationAuditResult,
    CorrectnessScore,
    EvaluationSummary,
    GroundingScore,
    ScoredEvaluationItem,
    TestCase,
    evaluate_test_case,
    generate_evaluation_markdown_summary,
    load_test_set,
    run_full_evaluation,
    save_evaluation_artifacts,
    score_correctness,
    score_grounding,
    verify_citation_accuracy,
)


class TestTestSetLoader(unittest.TestCase):
    def test_load_default_test_set(self) -> None:
        self.assertTrue(DEFAULT_TEST_SET_PATH.exists(), f"Missing {DEFAULT_TEST_SET_PATH}")
        test_cases = load_test_set(DEFAULT_TEST_SET_PATH)
        self.assertGreaterEqual(len(test_cases), 8)

        # Verify essential fields
        for tc in test_cases:
            self.assertTrue(tc.test_id)
            self.assertTrue(tc.category)
            self.assertTrue(tc.question)
            if not tc.should_refuse:
                self.assertGreater(len(tc.required_facts), 0)

    def test_test_case_deserialization(self) -> None:
        sample_dict = {
            "test_id": "test_01",
            "category": "Test Cat",
            "question": "Sample Question?",
            "expected_answer": "Sample Answer",
            "expected_sources": ["[Volume 1, PDF p. 10]"],
            "required_facts": ["Sample Fact"],
            "should_refuse": False,
        }
        tc = TestCase.from_dict(sample_dict)
        self.assertEqual(tc.test_id, "test_01")
        self.assertEqual(tc.category, "Test Cat")
        self.assertEqual(tc.required_facts, ["Sample Fact"])


class TestCorrectnessScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.test_case = TestCase(
            test_id="tc_sample",
            category="Fire Safety",
            question="What is the fire rating?",
            expected_answer="Walls must have a 120 min fire resistance rating under NBC.",
            required_facts=["120 min", "fire resistance rating"],
            should_refuse=False,
        )

    def test_score_correctness_high(self) -> None:
        answer = "Under the National Building Code, walls must achieve a 120 min fire resistance rating."
        score = score_correctness(answer, self.test_case, is_refusal=False)
        self.assertGreaterEqual(score.score, 0.80)
        self.assertEqual(len(score.missing_facts), 0)
        self.assertIn("120 min", score.matched_facts)

    def test_score_correctness_missing_facts(self) -> None:
        answer = "Walls must be painted red and installed properly."
        score = score_correctness(answer, self.test_case, is_refusal=False)
        self.assertLess(score.score, 0.50)
        self.assertIn("120 min", score.missing_facts)

    def test_score_correctness_valid_refusal(self) -> None:
        neg_tc = TestCase(
            test_id="tc_neg",
            category="Negative",
            question="What is the copyright penalty?",
            expected_answer="I cannot answer this question.",
            should_refuse=True,
        )
        answer = "I cannot answer this question because the query subject falls outside the scope of the National Building Code."
        score = score_correctness(answer, neg_tc, is_refusal=True)
        self.assertEqual(score.score, 1.0)
        self.assertTrue(score.is_refusal_correct)

    def test_score_correctness_hallucinated_instead_of_refusal(self) -> None:
        neg_tc = TestCase(
            test_id="tc_neg",
            category="Negative",
            question="What is the copyright penalty?",
            expected_answer="I cannot answer this question.",
            should_refuse=True,
        )
        answer = "The copyright penalty under Section 63 is 3 years imprisonment."
        score = score_correctness(answer, neg_tc, is_refusal=False)
        self.assertEqual(score.score, 0.0)
        self.assertFalse(score.is_refusal_correct)


class TestGroundingScorer(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            {
                "id": "c1",
                "text": "Clause 6.1.1.2 Boiler rooms shall be separated by walls having a fire resistance rating of 120 min.",
                "metadata": {"volume": "Volume 1", "page_start": 314},
            }
        ]

    def test_grounding_score_high(self) -> None:
        answer = "Boiler rooms must be separated by walls having a fire resistance rating of 120 min."
        score = score_grounding(answer, self.chunks, is_refusal=False)
        self.assertGreaterEqual(score.score, 0.80)
        self.assertFalse(score.hallucination_detected)

    def test_grounding_score_unsupported_claim(self) -> None:
        answer = "Boiler rooms require automatic halon fire suppression systems with blue painted steel pipes."
        score = score_grounding(answer, self.chunks, is_refusal=False)
        self.assertLess(score.score, 0.60)
        self.assertTrue(score.hallucination_detected)

    def test_grounding_score_for_refusal(self) -> None:
        score = score_grounding("I don't have enough information to answer that.", [], is_refusal=True)
        self.assertEqual(score.score, 1.0)
        self.assertFalse(score.hallucination_detected)


class TestCitationAccuracyVerifier(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = [
            {
                "id": "nbc-v1-p314-001012",
                "text": "Boiler rooms require 120 min fire rated walls.",
                "metadata": {
                    "volume": "Volume 1",
                    "page_start": 314,
                    "clause": "Part 4, Clause 6.1.1.2",
                },
            }
        ]

    def test_verify_valid_citations(self) -> None:
        answer = "Boiler rooms require 120 min fire walls [Volume 1, PDF p. 314] under Part 4, Clause 6.1.1.2."
        result = verify_citation_accuracy(
            answer,
            self.chunks,
            expected_sources=["[Volume 1, PDF p. 314]", "Part 4, Clause 6.1.1.2"],
            is_refusal=False,
        )
        self.assertGreaterEqual(result.citation_score, 0.75)
        self.assertGreaterEqual(result.citation_precision, 0.80)
        self.assertIn("Part 4, Clause 6.1.1.2", result.expected_sources_found)

    def test_verify_missing_citations(self) -> None:
        answer = "Boiler rooms require 120 min fire walls."
        result = verify_citation_accuracy(answer, self.chunks, expected_sources=["[Volume 1, PDF p. 314]"], is_refusal=False)
        self.assertLess(result.citation_score, 0.30)
        self.assertEqual(len(result.extracted_citations), 0)


class TestEndToEndEvaluation(unittest.TestCase):
    def test_evaluate_test_case_in_domain_offline(self) -> None:
        tc = TestCase(
            test_id="test_boiler",
            category="Fire Safety",
            question="What is the minimum fire resistance rating required for walls separating a boiler room from other occupancies?",
            expected_answer="Walls must provide 120 min fire resistance rating under Clause 6.1.1.2.",
            expected_sources=["[Volume 1, PDF p. 314-315]", "Part 4, Clause 6.1.1.2"],
            required_facts=["120 min", "boiler room"],
            should_refuse=False,
        )
        item = evaluate_test_case(tc, offline=True)
        self.assertIsInstance(item, ScoredEvaluationItem)
        self.assertTrue(item.passed)
        self.assertGreaterEqual(item.overall_score, 0.70)
        self.assertGreater(len(item.generated_answer), 20)

    def test_evaluate_test_case_negative_offline(self) -> None:
        tc = TestCase(
            test_id="test_neg",
            category="Negative",
            question="What is the copyright penalty?",
            expected_answer="I cannot answer this question.",
            should_refuse=True,
        )
        item = evaluate_test_case(tc, offline=True)
        self.assertTrue(item.passed)
        self.assertTrue(item.is_refusal)
        self.assertEqual(item.correctness.score, 1.0)

    def test_run_full_evaluation_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            test_cases = load_test_set(DEFAULT_TEST_SET_PATH)[:3]
            items, summary = run_full_evaluation(test_cases, offline=True)

            self.assertEqual(len(items), 3)
            self.assertIsInstance(summary, EvaluationSummary)
            self.assertGreater(summary.mean_overall_score, 0.50)

            artifacts = save_evaluation_artifacts(items, summary, output_dir=out_dir)
            self.assertTrue(artifacts["json"].exists())
            self.assertTrue(artifacts["markdown"].exists())

            # Verify exported JSON is valid
            with open(artifacts["json"], "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("summary", data)
            self.assertEqual(len(data["evaluated_items"]), 3)

            # Verify summary markdown
            with open(artifacts["markdown"], "r", encoding="utf-8") as f:
                md_text = f.read()
            self.assertIn("# RAG Evaluation & Answer Quality Scoring Report", md_text)
            self.assertIn("Executive Quality Benchmarks", md_text)


if __name__ == "__main__":
    unittest.main()
