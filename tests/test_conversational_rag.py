"""Unit tests for Conversational RAG & Follow-Up Context Engine."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.conversational_rag import (
    ConversationHistory,
    DialogueTurn,
    MultiTurnDialogueResult,
    QueryRewriteResult,
    RetrievalComparison,
    compare_follow_up_retrieval,
    detect_follow_up_signals,
    execute_conversational_turn,
    generate_conversational_rag_markdown,
    get_conversational_fallback_chunks,
    retrieve_chunks_for_query,
    rewrite_follow_up_query,
    run_sample_multi_turn_dialogue,
    save_conversational_rag_artifacts,
)


class TestConversationHistory(unittest.TestCase):
    def setUp(self) -> None:
        self.history = ConversationHistory(max_turns=3)

    def test_add_turn_and_windowing(self) -> None:
        for i in range(1, 5):
            turn = DialogueTurn(
                turn_id=i,
                user_query=f"Question {i}",
                rewritten_query=f"Question {i}",
                assistant_answer=f"Answer {i}",
            )
            self.history.add_turn(turn)

        # max_turns is 3; oldest turn (1) should be dropped
        self.assertEqual(len(self.history.turns), 3)
        self.assertEqual(self.history.turns[0].turn_id, 2)
        self.assertEqual(self.history.turns[-1].turn_id, 4)

    def test_extract_subject_entities(self) -> None:
        turn1 = DialogueTurn(
            turn_id=1,
            user_query="What are the fire resistance rating requirements for walls separating a boiler room?",
            rewritten_query="What are the fire resistance rating requirements for walls separating a boiler room?",
            assistant_answer="Walls must have 120 min rating.",
        )
        self.history.add_turn(turn1)
        entities = self.history.extract_subject_entities()
        self.assertIn("boiler room", entities)
        # Spatial entities should come first
        self.assertEqual(entities[0], "boiler room")

    def test_serialization_and_deserialization(self) -> None:
        turn = DialogueTurn(
            turn_id=1,
            user_query="Boiler room fire wall rating?",
            rewritten_query="Boiler room fire wall rating?",
            assistant_answer="120 min rating under NBC.",
            citations=["[Volume 1, PDF p. 314-315]"],
        )
        self.history.add_turn(turn)
        serialized = self.history.to_dict()
        restored = ConversationHistory.from_dict(serialized)

        self.assertEqual(len(restored.turns), 1)
        self.assertEqual(restored.turns[0].user_query, "Boiler room fire wall rating?")
        self.assertEqual(restored.turns[0].citations, ["[Volume 1, PDF p. 314-315]"])

    def test_format_history_for_rewriter(self) -> None:
        turn = DialogueTurn(
            turn_id=1,
            user_query="What is the fire rating for boiler room walls?",
            rewritten_query="What is the fire rating for boiler room walls?",
            assistant_answer="Requires 2 hours fire resistance.",
        )
        self.history.add_turn(turn)
        formatted = self.history.format_history_for_rewriter(max_turns=2)
        self.assertIn("User: What is the fire rating for boiler room walls?", formatted)
        self.assertIn("Assistant: Requires 2 hours fire resistance.", formatted)


class TestFollowUpDetectionAndRewriting(unittest.TestCase):
    def setUp(self) -> None:
        self.history = ConversationHistory()
        turn1 = DialogueTurn(
            turn_id=1,
            user_query="What is the minimum fire resistance rating required for walls separating a boiler room?",
            rewritten_query="What is the minimum fire resistance rating required for walls separating a boiler room?",
            assistant_answer="According to NBC Part 4 Clause 6.1.1.2, walls must have 120 minutes fire resistance.",
            resolved_entities=["boiler room", "separation walls"],
        )
        self.history.add_turn(turn1)

    def test_standalone_query_not_flagged(self) -> None:
        empty_hist = ConversationHistory()
        res = rewrite_follow_up_query("What is the staircase width in educational buildings?", empty_hist)
        self.assertFalse(res.is_follow_up)
        self.assertEqual(res.rewritten_query, "What is the staircase width in educational buildings?")
        self.assertEqual(res.method, "standalone_passthrough")

    def test_detect_elliptical_follow_up(self) -> None:
        q = "What about the doors in those walls?"
        is_follow_up, markers, _ = detect_follow_up_signals(q, self.history)
        self.assertTrue(is_follow_up)
        self.assertTrue(any(m in ("those", "elliptical_phrase") for m in markers))

    def test_heuristic_rewrite_elliptical_door_query(self) -> None:
        q = "What about the doors in those walls?"
        res = rewrite_follow_up_query(q, self.history, offline=True)
        self.assertTrue(res.is_follow_up)
        self.assertIn("boiler room", res.rewritten_query.lower())
        self.assertIn("doors", res.rewritten_query.lower())
        self.assertIn("National Building Code", res.rewritten_query)
        self.assertEqual(res.method, "heuristic")

    def test_heuristic_rewrite_pronoun_exception_query(self) -> None:
        q = "Are there any exceptions or ventilation openings permitted for them?"
        res = rewrite_follow_up_query(q, self.history, offline=True)
        self.assertTrue(res.is_follow_up)
        self.assertIn("boiler room", res.rewritten_query.lower())
        self.assertIn("exceptions", res.rewritten_query.lower())

    @patch("src.assistant.api_client.get_chat_model", return_value="mock-model")
    @patch("src.assistant.api_client.get_chat_client")
    def test_llm_rewrite_mock(self, mock_client_factory: MagicMock, mock_model: MagicMock) -> None:
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = (
            "What are the opening protection requirements for doors in boiler room separation walls under the NBC?"
        )
        mock_client.chat.completions.create.return_value.choices = [mock_choice]
        mock_client_factory.return_value = mock_client

        res = rewrite_follow_up_query("What about the doors in those walls?", self.history, offline=False)
        self.assertTrue(res.is_follow_up)
        self.assertIn("boiler room", res.rewritten_query.lower())
        self.assertEqual(res.method, "llm")


class TestRetrievalWithRewrittenQuery(unittest.TestCase):
    def test_fallback_chunks_content(self) -> None:
        chunks = get_conversational_fallback_chunks(topic="boiler room door")
        self.assertGreaterEqual(len(chunks), 1)
        self.assertIn("boiler room", chunks[0]["text"].lower())
        self.assertIn("fire doors", chunks[0]["text"].lower())
        self.assertGreaterEqual(chunks[0]["cosine_similarity"], 0.75)

    def test_compare_follow_up_retrieval(self) -> None:
        raw_q = "What about the doors?"
        rewritten_q = (
            "What are the fire resistance rating and opening protection requirements for doors "
            "in boiler room separation walls under the National Building Code of India?"
        )
        cmp_res = compare_follow_up_retrieval(
            turn_id=2,
            raw_query=raw_q,
            rewritten_query=rewritten_q,
            target_entity="boiler room",
            offline=True,
        )
        self.assertEqual(cmp_res.turn_id, 2)
        self.assertGreaterEqual(cmp_res.rewritten_top_similarity, 0.70)
        self.assertTrue(cmp_res.rewritten_contains_entity)
        self.assertIn("boiler room", cmp_res.explanation)


class TestMultiTurnDialogueExecution(unittest.TestCase):
    def test_run_sample_multi_turn_dialogue_offline(self) -> None:
        result = run_sample_multi_turn_dialogue(offline=True)

        self.assertIsInstance(result, MultiTurnDialogueResult)
        self.assertEqual(len(result.turns), 3)

        # Turn 1 should be initial standalone query
        self.assertEqual(result.turns[0].turn_id, 1)
        self.assertFalse(result.turns[0].is_follow_up)
        self.assertIn("boiler room", result.turns[0].user_query.lower())

        # Turn 2 and 3 should be follow-up queries
        self.assertTrue(result.turns[1].is_follow_up)
        self.assertTrue(result.turns[2].is_follow_up)

        # All turns must have answers and citations
        for t in result.turns:
            self.assertTrue(len(t.assistant_answer) > 20)
            self.assertGreater(len(t.citations), 0)
            self.assertGreater(t.retrieval_metrics["top_similarity"], 0.65)

        # Verify comparisons were recorded
        self.assertEqual(len(result.comparisons), 2)
        self.assertGreater(result.summary_metrics["total_turns"], 0)

    def test_generate_markdown_report(self) -> None:
        result = run_sample_multi_turn_dialogue(offline=True)
        md = generate_conversational_rag_markdown(result)
        self.assertIn("# Conversational RAG & Follow-Up Context Report", md)
        self.assertIn("### Turn 1: [Initial Query]", md)
        self.assertIn("### Turn 2: [Follow-Up Turn]", md)
        self.assertIn("Rewritten Standalone Query", md)
        self.assertIn("| Turn | Raw Follow-Up Query | Rewritten Standalone Query |", md)


class TestArtifactExport(unittest.TestCase):
    def test_save_conversational_rag_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            result = run_sample_multi_turn_dialogue(offline=True)
            artifacts = save_conversational_rag_artifacts(result, output_dir=out_dir)

            self.assertTrue(artifacts["json"].exists())
            self.assertTrue(artifacts["markdown"].exists())

            with open(artifacts["json"], "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(len(data["turns"]), 3)
            self.assertEqual(len(data["comparisons"]), 2)

            with open(artifacts["markdown"], "r", encoding="utf-8") as f:
                md_content = f.read()
            self.assertIn("Boiler Room Fire Separation", md_content)


if __name__ == "__main__":
    unittest.main()
