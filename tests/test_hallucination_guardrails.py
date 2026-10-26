"""Unit tests for hallucination guardrails & refusal handling."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.hallucination_guardrails import (
    DEFAULT_SAFE_REFUSAL,
    GuardrailConfig,
    GuardrailDecision,
    GuardedGenerationResult,
    check_retrieval_guardrail,
    execute_guarded_generation,
    generate_guardrail_markdown_report,
    get_confident_answer_test_case,
    get_refusal_test_cases,
    save_guardrail_artifacts,
)


class TestRetrievalGuardrailChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GuardrailConfig(
            min_similarity_threshold=0.65,
            min_chunks_above_threshold=1,
            min_avg_similarity=0.50,
            enable_topic_overlap_check=True,
        )

    def test_empty_retrieval_refuses(self) -> None:
        decision = check_retrieval_guardrail("What is the fire rating?", [], config=self.config)
        self.assertEqual(decision.decision, "REFUSE")
        self.assertEqual(decision.reason_code, "EMPTY_RETRIEVAL")
        self.assertTrue(decision.is_refusal)
        self.assertIsNotNone(decision.safe_refusal_text)
        self.assertIn("no relevant provisions were found", decision.safe_refusal_text.lower())

    def test_low_max_similarity_refuses(self) -> None:
        chunks = [
            {"id": "c1", "text": "Unrelated topic text", "cosine_similarity": 0.42},
            {"id": "c2", "text": "Another unrelated text", "cosine_similarity": 0.38},
        ]
        decision = check_retrieval_guardrail("Fire door ratings?", chunks, config=self.config)
        self.assertEqual(decision.decision, "REFUSE")
        self.assertEqual(decision.reason_code, "LOW_MAX_SIMILARITY")
        self.assertTrue(decision.is_refusal)
        self.assertIn("falls below the required confidence threshold", decision.safe_refusal_text.lower())

    def test_insufficient_chunks_above_threshold_refuses(self) -> None:
        config_strict = GuardrailConfig(
            min_similarity_threshold=0.70,
            min_chunks_above_threshold=2,
            min_avg_similarity=0.50,
        )
        chunks = [
            {"id": "c1", "text": "Good fire door match", "cosine_similarity": 0.78},
            {"id": "c2", "text": "Weak match", "cosine_similarity": 0.55},
        ]
        decision = check_retrieval_guardrail("Fire door ratings?", chunks, config=config_strict)
        self.assertEqual(decision.decision, "REFUSE")
        self.assertEqual(decision.reason_code, "INSUFFICIENT_SUPPORTING_CHUNKS")

    def test_low_avg_similarity_refuses(self) -> None:
        config_avg = GuardrailConfig(
            min_similarity_threshold=0.65,
            min_chunks_above_threshold=1,
            min_avg_similarity=0.60,
        )
        chunks = [
            {"id": "c1", "text": "Fire door text", "cosine_similarity": 0.70},
            {"id": "c2", "text": "Random text A", "cosine_similarity": 0.30},
            {"id": "c3", "text": "Random text B", "cosine_similarity": 0.35},
        ]
        decision = check_retrieval_guardrail("Fire door ratings?", chunks, config=config_avg)
        self.assertEqual(decision.decision, "REFUSE")
        self.assertEqual(decision.reason_code, "LOW_AVERAGE_SIMILARITY")

    def test_topic_mismatch_refuses(self) -> None:
        # High numerical similarity but zero subject words present
        chunks = [
            {"id": "c1", "text": "Parking and setback requirements for commercial plots.", "cosine_similarity": 0.72},
        ]
        query = "What is the capital gains tax liability?"
        decision = check_retrieval_guardrail(query, chunks, config=self.config)
        self.assertEqual(decision.decision, "REFUSE")
        self.assertEqual(decision.reason_code, "TOPIC_MISMATCH")

    def test_confident_proceed(self) -> None:
        query, chunks = get_confident_answer_test_case()
        decision = check_retrieval_guardrail(query, chunks, config=self.config)
        self.assertEqual(decision.decision, "PROCEED")
        self.assertEqual(decision.reason_code, "OK")
        self.assertFalse(decision.is_refusal)
        self.assertIsNone(decision.safe_refusal_text)
        self.assertGreater(decision.metrics["max_similarity"], 0.80)


class TestGuardedGenerationExecution(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GuardrailConfig(min_similarity_threshold=0.65)
        self.conf_query, self.conf_chunks = get_confident_answer_test_case()

    def test_guarded_generation_preserves_confident_answer(self) -> None:
        res = execute_guarded_generation(
            self.conf_query, self.conf_chunks, config=self.config, offline=True
        )
        self.assertEqual(res.status, "ANSWERED")
        self.assertEqual(res.guardrail_decision.decision, "PROCEED")
        self.assertIn("120", res.answer)
        self.assertIn("[1]", res.citations)

    def test_guarded_generation_mocked_llm_proceed(self) -> None:
        mock_choice = MagicMock()
        mock_choice.message.content = "Walls require a 120 min rating according to [1]."
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_resp

        with patch("src.assistant.api_client.get_chat_client", return_value=mock_client), \
             patch("src.assistant.api_client.get_chat_model", return_value="gemini-2.5-flash"):
            res = execute_guarded_generation(
                self.conf_query, self.conf_chunks, config=self.config, offline=False
            )
            self.assertEqual(res.status, "ANSWERED")
            self.assertEqual(res.model, "gemini-2.5-flash")
            self.assertIn("[1]", res.citations)

    def test_guarded_generation_intercepts_refusal(self) -> None:
        weak_query = "What is the penalty for copyright violation?"
        res = execute_guarded_generation(
            weak_query, [], config=self.config, offline=True
        )
        self.assertEqual(res.status, "REFUSED")
        self.assertEqual(res.guardrail_decision.decision, "REFUSE")
        self.assertEqual(res.guardrail_decision.reason_code, "EMPTY_RETRIEVAL")
        self.assertIn("no relevant provisions were found", res.answer.lower())
        self.assertEqual(len(res.citations), 0)


class TestArtifactsAndReporting(unittest.TestCase):
    def test_save_guardrail_artifacts(self) -> None:
        config = GuardrailConfig()
        conf_query, conf_chunks = get_confident_answer_test_case()
        conf_res = execute_guarded_generation(conf_query, conf_chunks, config=config, offline=True)

        refusals = [
            execute_guarded_generation("Tax rate?", [], config=config, offline=True),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            json_path, md_path = save_guardrail_artifacts(
                conf_res, refusals, config, output_dir=out_dir
            )

            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())

            # Verify JSON structure
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertIn("guardrail_config", data)
            self.assertIn("confident_case", data)
            self.assertIn("refusal_cases", data)
            self.assertEqual(data["confident_case"]["status"], "ANSWERED")
            self.assertEqual(data["refusal_cases"][0]["status"], "REFUSED")

            # Verify Markdown structure
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("# Hallucination Guardrails & Refusal Handling Report", md_text)
            self.assertIn("Confident Answer Case (PROCEED)", md_text)
            self.assertIn("Refusal Cases (REFUSE)", md_text)


if __name__ == "__main__":
    unittest.main()
