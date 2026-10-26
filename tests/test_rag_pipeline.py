"""Unit tests for the modular RAG pipeline architecture and flow."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.rag_pipeline import (
    AssembleResult,
    EmbedResult,
    GenerateResult,
    PipelineResult,
    RerankResult,
    RetrieveResult,
    _generate_dry_run,
    assemble,
    embed,
    generate,
    rerank,
    retrieve,
    run_pipeline,
    save_pipeline_output,
)


class TestRAGPipelineStages(unittest.TestCase):
    """Test each stage of the RAG pipeline in isolation."""

    # -----------------------------------------------------------------------
    # Stage 1 — EMBED
    # -----------------------------------------------------------------------
    def test_embed_empty_query_raises_error(self) -> None:
        with self.assertRaises(ValueError):
            embed("")
        with self.assertRaises(ValueError):
            embed("   ")

    def test_embed_returns_embed_result(self) -> None:
        mock_vec = [0.1, 0.2, 0.3, 0.4]
        with patch("src.embeddings.retrieve._embed_query", return_value=mock_vec):
            res = embed("What is fire rating?", model_name="dummy-model")
            self.assertIsInstance(res, EmbedResult)
            self.assertEqual(res.query, "What is fire rating?")
            self.assertEqual(res.vector, mock_vec)
            self.assertEqual(res.model_name, "dummy-model")
            self.assertGreaterEqual(res.elapsed_ms, 0.0)

    # -----------------------------------------------------------------------
    # Stage 2 — RETRIEVE
    # -----------------------------------------------------------------------
    def test_retrieve_returns_retrieve_result(self) -> None:
        mock_candidates = [
            {"id": "chunk_1", "text": "NBC fire doors.", "cosine_similarity": 0.88},
            {"id": "chunk_2", "text": "Fire separation walls.", "cosine_similarity": 0.82},
        ]
        with patch(
            "src.embeddings.filtered_retrieval.retrieve_filtered",
            return_value={"results": mock_candidates},
        ):
            res = retrieve(
                "fire doors",
                candidate_k=2,
                metadata_filter={"heading": "Part 4"},
                hybrid=True,
            )
            self.assertIsInstance(res, RetrieveResult)
            self.assertEqual(len(res.candidates), 2)
            self.assertEqual(res.candidate_k, 2)
            self.assertEqual(res.metadata_filter, {"heading": "Part 4"})
            self.assertTrue(res.hybrid)
            self.assertGreaterEqual(res.elapsed_ms, 0.0)

    # -----------------------------------------------------------------------
    # Stage 3 — RERANK
    # -----------------------------------------------------------------------
    def test_rerank_empty_candidates(self) -> None:
        res = rerank("test query", [], final_k=3, method="cross_encoder")
        self.assertIsInstance(res, RerankResult)
        self.assertEqual(res.chunks, [])
        self.assertEqual(res.final_k, 3)
        self.assertEqual(res.elapsed_ms, 0.0)

    def test_rerank_delegates_to_rerank_candidates(self) -> None:
        candidates = [
            {"id": "chunk_1", "text": "Wall thickness.", "cosine_similarity": 0.75},
            {"id": "chunk_2", "text": "Boiler room walls.", "cosine_similarity": 0.70},
        ]
        mock_reranked = [
            {"id": "chunk_2", "text": "Boiler room walls.", "rerank_score": 0.95, "rerank_rank": 1},
            {"id": "chunk_1", "text": "Wall thickness.", "rerank_score": 0.60, "rerank_rank": 2},
        ]
        with patch("src.embeddings.rerank.rerank_candidates", return_value=mock_reranked):
            res = rerank("boiler room wall fire rating", candidates, final_k=2, method="cross_encoder")
            self.assertIsInstance(res, RerankResult)
            self.assertEqual(len(res.chunks), 2)
            self.assertEqual(res.chunks[0]["id"], "chunk_2")
            self.assertGreaterEqual(res.elapsed_ms, 0.0)

    # -----------------------------------------------------------------------
    # Stage 4 — ASSEMBLE
    # -----------------------------------------------------------------------
    def test_assemble_empty_chunks(self) -> None:
        res = assemble([])
        self.assertIsInstance(res, AssembleResult)
        self.assertIn("No relevant chunks retrieved", res.context_block)
        self.assertEqual(res.sources, [])
        self.assertEqual(res.chunk_count, 0)

    def test_assemble_formats_chunk_annotations(self) -> None:
        chunks = [
            {
                "id": "chunk_vol1_p314_c01",
                "text": "Boiler rooms must have 2-hour fire separation.",
                "cosine_similarity": 0.8912,
                "rerank_score": 0.9850,
                "rerank_rank": 1,
                "metadata": {
                    "source": "nationalbuilding01.pdf",
                    "volume": "1",
                    "page": 314,
                    "heading": "Part 4 — Fire and Life Safety",
                },
            },
            {
                "id": "chunk_vol1_p315_c02",
                "text": "Fire doors in boiler rooms must have self-closing devices.",
                "cosine_similarity": 0.8120,
                "rerank_score": 0.8900,
                "rerank_rank": 2,
                "metadata": {
                    "source": "nationalbuilding01.pdf",
                    "volume": "1",
                    "page": 315,
                    "heading": "Part 4 — Fire and Life Safety",
                },
            },
        ]
        res = assemble(chunks)
        self.assertIsInstance(res, AssembleResult)
        self.assertEqual(res.chunk_count, 2)
        self.assertEqual(len(res.sources), 2)

        # Context block content checks
        self.assertIn("=== RETRIEVED BUILDING CODE CONTEXT ===", res.context_block)
        self.assertIn("[Chunk 1 of 2]", res.context_block)
        self.assertIn("Page: 314", res.context_block)
        self.assertIn("Part 4 — Fire and Life Safety", res.context_block)
        self.assertIn("Similarity: 0.8912", res.context_block)
        self.assertIn("Re-rank score: 0.9850", res.context_block)
        self.assertIn("Boiler rooms must have 2-hour fire separation.", res.context_block)

        # Sources list checks
        src = res.sources[0]
        self.assertEqual(src["id"], "chunk_vol1_p314_c01")
        self.assertEqual(src["page"], 314)
        self.assertEqual(src["volume"], "1")
        self.assertEqual(src["rank"], 1)
        self.assertAlmostEqual(src["cosine_similarity"], 0.8912)
        self.assertAlmostEqual(src["rerank_score"], 0.9850)

    def test_assemble_with_page_range_metadata(self) -> None:
        chunks = [
            {
                "id": "nbc-v1-p314-001012",
                "text": "120 min fire resistance rating wall.",
                "cosine_similarity": 0.8264,
                "rerank_score": 4.0295,
                "metadata": {
                    "source": "nationalbuilding01.pdf",
                    "volume": 1,
                    "page_start": 314,
                    "page_end": 315,
                    "heading": "Part 4 — Fire and Life Safety",
                },
            }
        ]
        res = assemble(chunks)
        self.assertIn("Page: 314-315", res.context_block)
        self.assertEqual(res.sources[0]["page"], "314-315")

    # -----------------------------------------------------------------------
    # Stage 5 — GENERATE
    # -----------------------------------------------------------------------
    def test_generate_dry_run(self) -> None:
        sources = [{"page": 314}, {"page": 315}]
        res = _generate_dry_run("What is fire rating?", "context block text", sources)
        self.assertIsInstance(res, GenerateResult)
        self.assertIn("[DRY-RUN]", res.answer)
        self.assertIn("PDF p. 314", res.answer)
        self.assertEqual(res.model, "dry-run")
        self.assertEqual(res.elapsed_ms, 0.0)

    def test_generate_mocked_llm(self) -> None:
        mock_choice = MagicMock()
        mock_choice.message.content = "Walls must have a 2-hour fire rating [Volume 1, PDF p. 314]."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.model_dump_json.return_value = '{"id": "chatcmpl-test"}'

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("src.assistant.api_client.get_chat_client", return_value=mock_client), \
             patch("src.assistant.api_client.get_chat_model", return_value="gemini-2.5-flash"):
            res = generate("What is the boiler room fire rating?", "Retrieved context here...")
            self.assertIsInstance(res, GenerateResult)
            self.assertIn("2-hour fire rating", res.answer)
            self.assertEqual(res.model, "gemini-2.5-flash")
            self.assertGreaterEqual(res.elapsed_ms, 0.0)

            # Check prompt parameters
            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args.kwargs
            self.assertEqual(call_kwargs["temperature"], 0.0)
            self.assertEqual(call_kwargs["max_tokens"], 512)
            self.assertEqual(len(call_kwargs["messages"]), 2)
            self.assertEqual(call_kwargs["messages"][0]["role"], "system")
            self.assertEqual(call_kwargs["messages"][1]["role"], "user")


class TestRAGPipelineEndToEnd(unittest.TestCase):
    """Test full pipeline orchestrator and persistence."""

    def test_run_pipeline_dry_run_end_to_end(self) -> None:
        mock_vec = [0.1] * 384
        mock_candidates = [
            {
                "id": f"chunk_{i}",
                "text": f"Chunk text {i}",
                "cosine_similarity": 0.8 - (i * 0.05),
                "metadata": {"source": "nationalbuilding01.pdf", "page": 100 + i, "heading": "Part 4"},
            }
            for i in range(10)
        ]
        mock_reranked = [
            {
                "id": f"chunk_{i}",
                "text": f"Chunk text {i}",
                "cosine_similarity": 0.8 - (i * 0.05),
                "rerank_score": 0.9 - (i * 0.1),
                "rerank_rank": i + 1,
                "metadata": {"source": "nationalbuilding01.pdf", "page": 100 + i, "heading": "Part 4"},
            }
            for i in range(3)
        ]

        with patch("src.embeddings.retrieve._embed_query", return_value=mock_vec), \
             patch("src.embeddings.filtered_retrieval.retrieve_filtered", return_value={"results": mock_candidates}), \
             patch("src.embeddings.rerank.rerank_candidates", return_value=mock_reranked):

            result = run_pipeline(
                "Boiler room fire rating",
                candidate_k=10,
                final_k=3,
                dry_run=True,
            )

            self.assertIsInstance(result, PipelineResult)
            self.assertEqual(result.query, "Boiler room fire rating")
            self.assertIn("[DRY-RUN]", result.answer)
            self.assertEqual(len(result.sources), 3)
            self.assertIn("=== RETRIEVED BUILDING CODE CONTEXT ===", result.context_block)
            self.assertTrue(result.generated_at)
            self.assertGreater(result.pipeline_elapsed_ms, 0.0)

            # Stage metrics verification
            stages = result.stages
            self.assertIn("embed", stages)
            self.assertIn("retrieve", stages)
            self.assertIn("rerank", stages)
            self.assertIn("assemble", stages)
            self.assertIn("generate", stages)

            self.assertEqual(stages["embed"]["vector_dim"], 384)
            self.assertEqual(stages["retrieve"]["retrieved_count"], 10)
            self.assertEqual(stages["rerank"]["returned_count"], 3)
            self.assertEqual(stages["assemble"]["chunk_count"], 3)
            self.assertEqual(stages["generate"]["model"], "dry-run")

    def test_save_pipeline_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_out = Path(tmpdir)
            with patch("src.rag_pipeline.OUTPUT_DIR", tmp_out):
                res = PipelineResult(
                    query="Test question",
                    answer="Test answer",
                    sources=[{"id": "chunk_1", "page": 42}],
                    context_block="Test context",
                    generated_at="2026-10-26T12:00:00Z",
                    pipeline_elapsed_ms=45.2,
                    stages={"embed": {"elapsed_ms": 10.0}},
                )
                save_pipeline_output(res, label="unittest")

                expected_file = tmp_out / "pipeline_output_unittest.json"
                self.assertTrue(expected_file.exists())

                data = json.loads(expected_file.read_text(encoding="utf-8"))
                self.assertEqual(data["query"], "Test question")
                self.assertEqual(data["answer"], "Test answer")
                self.assertEqual(len(data["sources"]), 1)
                self.assertEqual(data["pipeline_elapsed_ms"], 45.2)


if __name__ == "__main__":
    unittest.main()
