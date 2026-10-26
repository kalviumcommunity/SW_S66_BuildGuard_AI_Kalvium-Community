"""Unit tests for context injection & prompt augmentation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.context_assembly import (
    DEFAULT_GROUNDING_INSTRUCTIONS,
    INSUFFICIENT_CONTEXT_FALLBACK,
    AugmentedPromptResult,
    InjectedChunk,
    TokenBudget,
    assemble_context_with_budget,
    build_augmented_prompt,
    count_tokens,
    format_chunk_with_marker,
    generate_markdown_report,
    get_synthetic_test_chunks,
    save_augmented_prompt,
)


class TestTokenCounting(unittest.TestCase):
    def test_empty_string_returns_zero(self) -> None:
        self.assertEqual(count_tokens(""), 0)

    def test_non_empty_string_returns_positive_count(self) -> None:
        cnt = count_tokens("Building Code of India")
        self.assertGreater(cnt, 0)

    def test_token_count_scales_with_length(self) -> None:
        short_cnt = count_tokens("Fire door rating")
        long_cnt = count_tokens("Fire door rating " * 20)
        self.assertGreater(long_cnt, short_cnt)


class TestChunkFormattingAndMarkers(unittest.TestCase):
    def setUp(self) -> None:
        self.sample_chunk = {
            "id": "nbc-v1-p314-001012",
            "text": "Boiler room walls must have a 120-minute fire resistance rating.",
            "cosine_similarity": 0.8842,
            "rerank_score": 4.0295,
            "metadata": {
                "source": "nationalbuilding01.pdf",
                "volume": 1,
                "page_start": 314,
                "page_end": 315,
                "clause": "6.1.1.2",
                "heading": "Part 4 — Fire and Life Safety",
            },
        }

    def test_numeric_marker_style(self) -> None:
        text, record = format_chunk_with_marker(self.sample_chunk, index=1, marker_style="numeric")
        self.assertEqual(record.marker, "[1]")
        self.assertIn("[1]", text)
        self.assertIn("Doc: nationalbuilding01.pdf", text)
        self.assertIn("Volume: 1", text)
        self.assertIn("Page: 314-315", text)
        self.assertIn("Clause: 6.1.1.2", text)
        self.assertIn("Section: Part 4 — Fire and Life Safety", text)
        self.assertIn("Similarity: 0.8842", text)
        self.assertIn("Re-rank Score: 4.0295", text)
        self.assertIn("Boiler room walls must have a 120-minute fire resistance rating.", text)
        self.assertFalse(record.truncated)
        self.assertGreater(record.token_count, 0)

    def test_verbose_marker_style(self) -> None:
        text, record = format_chunk_with_marker(self.sample_chunk, index=2, marker_style="verbose")
        self.assertEqual(record.marker, "[Source 2]")
        self.assertIn("[Source 2]", text)

    def test_single_page_representation(self) -> None:
        single_page_chunk = dict(self.sample_chunk)
        single_page_chunk["metadata"] = {
            "source": "nationalbuilding01.pdf",
            "page_start": 42,
            "page_end": 42,
        }
        text, record = format_chunk_with_marker(single_page_chunk, index=1)
        self.assertEqual(record.page, "42")
        self.assertIn("Page: 42", text)


class TestContextAssemblyWithBudget(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = get_synthetic_test_chunks()

    def test_empty_chunks_returns_fallback_notice(self) -> None:
        context, injected, used, skipped, trunc = assemble_context_with_budget(
            [], available_context_tokens=1000
        )
        self.assertIn("No relevant context available", context)
        self.assertEqual(len(injected), 0)
        self.assertEqual(used, count_tokens(context))
        self.assertEqual(skipped, 0)
        self.assertEqual(trunc, 0)

    def test_generous_budget_injects_all_chunks(self) -> None:
        context, injected, used, skipped, trunc = assemble_context_with_budget(
            self.chunks, available_context_tokens=5000
        )
        self.assertEqual(len(injected), len(self.chunks))
        self.assertEqual(skipped, 0)
        self.assertGreater(used, 0)
        self.assertLessEqual(used, 5000)
        for i in range(1, len(self.chunks) + 1):
            self.assertIn(f"[{i}]", context)

    def test_tight_budget_skips_overflow_chunks(self) -> None:
        # Give only enough room for roughly 1 chunk
        context, injected, used, skipped, trunc = assemble_context_with_budget(
            self.chunks, available_context_tokens=150, allow_partial_truncation=False
        )
        self.assertLess(len(injected), len(self.chunks))
        self.assertGreater(skipped, 0)
        self.assertLessEqual(used, 150)

    def test_partial_truncation_when_enabled(self) -> None:
        # Give just enough tokens for chunk 1 + partial chunk 2
        chunk1_cost = format_chunk_with_marker(self.chunks[0], 1)[1].token_count
        available = chunk1_cost + 115
        context, injected, used, skipped, trunc = assemble_context_with_budget(
            self.chunks,
            available_context_tokens=available,
            allow_partial_truncation=True,
            min_chunk_tokens=25,
        )
        self.assertEqual(len(injected), 2)
        self.assertEqual(trunc, 1)
        self.assertIn("truncated to fit token budget", context)
        self.assertLessEqual(used, available)


class TestAugmentedPromptBuilder(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks = get_synthetic_test_chunks()
        self.query = "What is the fire rating for boiler room walls?"

    def test_empty_query_raises_error(self) -> None:
        with self.assertRaises(ValueError):
            build_augmented_prompt("", self.chunks)
        with self.assertRaises(ValueError):
            build_augmented_prompt("   ", self.chunks)

    def test_build_augmented_prompt_end_to_end(self) -> None:
        result = build_augmented_prompt(
            query=self.query,
            chunks=self.chunks,
            max_total_tokens=4096,
            reserved_answer_tokens=512,
        )

        self.assertIsInstance(result, AugmentedPromptResult)
        self.assertEqual(result.query, self.query)
        self.assertIn(DEFAULT_GROUNDING_INSTRUCTIONS, result.system_prompt)
        self.assertIn(INSUFFICIENT_CONTEXT_FALLBACK, result.system_prompt)
        self.assertIn(INSUFFICIENT_CONTEXT_FALLBACK, result.user_prompt)

        # Context injection and markers
        self.assertIn("=== RETRIEVED BUILDING CODE CONTEXT ===", result.user_prompt)
        self.assertIn("[1]", result.user_prompt)
        self.assertIn(self.query, result.user_prompt)
        self.assertIn("COMPLIANCE ANSWER:", result.user_prompt)

        # Token budget verification
        tb = result.token_budget
        self.assertEqual(tb.max_total_tokens, 4096)
        self.assertEqual(tb.reserved_answer_tokens, 512)
        self.assertGreater(tb.system_tokens, 0)
        self.assertGreater(tb.query_template_overhead_tokens, 0)
        self.assertEqual(
            tb.available_context_tokens,
            4096 - 512 - tb.system_tokens - tb.query_template_overhead_tokens,
        )
        self.assertLessEqual(tb.used_context_tokens, tb.available_context_tokens)
        self.assertGreaterEqual(tb.remaining_tokens, 0)
        self.assertGreater(tb.utilization_pct, 0.0)
        self.assertLessEqual(tb.utilization_pct, 100.0)

    def test_chat_messages_structure(self) -> None:
        result = build_augmented_prompt(
            query=self.query,
            chunks=self.chunks,
        )
        messages = result.to_chat_messages()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[0]["content"], result.system_prompt)
        self.assertEqual(messages[1]["content"], result.user_prompt)

    def test_insufficient_budget_raises_error(self) -> None:
        # Ridiculously small window that cannot even fit fixed prompts
        with self.assertRaises(ValueError):
            build_augmented_prompt(
                query=self.query,
                chunks=self.chunks,
                max_total_tokens=100,
                reserved_answer_tokens=150,
            )


class TestPersistenceAndReports(unittest.TestCase):
    def test_save_augmented_prompt_and_markdown(self) -> None:
        chunks = get_synthetic_test_chunks()
        result = build_augmented_prompt(
            query="Sample NBC query?",
            chunks=chunks,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            json_path, md_path = save_augmented_prompt(result, output_dir=out_dir, label="test_sample")

            self.assertTrue(json_path.exists())
            self.assertTrue(md_path.exists())

            # Check JSON contents
            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["query"], "Sample NBC query?")
            self.assertIn("token_budget", data)
            self.assertIn("injected_chunks", data)
            self.assertGreater(len(data["injected_chunks"]), 0)

            # Check Markdown contents
            md_text = md_path.read_text(encoding="utf-8")
            self.assertIn("# Augmented Prompt & Context Assembly Report", md_text)
            self.assertIn("Token Budget Allocation", md_text)
            self.assertIn("Injected Sources & Markers", md_text)
            self.assertIn("Grounding Instructions", md_text)


if __name__ == "__main__":
    unittest.main()
