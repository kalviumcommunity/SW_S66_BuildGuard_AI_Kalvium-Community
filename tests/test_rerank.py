"""Unit tests for chunk re-ranking module."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.embeddings.rerank import (
    compute_custom_precision_score,
    generate_markdown_comparison_report,
    rerank_candidates,
    retrieve_and_rerank,
)


class TestChunkRerank(unittest.TestCase):
    def test_compute_custom_precision_score_with_bonus(self) -> None:
        query = 'fire doors "Clause 2.18" requirements'
        text = "Under Clause 2.18, fire doors must maintain integrity and smoke resistance."
        meta = {"heading": "Part 4 — Fire and Life Safety"}
        score = compute_custom_precision_score(
            query=query,
            text=text,
            metadata=meta,
            original_similarity=0.75,
        )
        self.assertGreater(score, 0.70)

    def test_compute_custom_precision_score_unrelated(self) -> None:
        query = "timber permissible stress"
        text = "Concrete aggregate mixture ratios."
        score = compute_custom_precision_score(
            query=query,
            text=text,
            metadata={"heading": "Part 6 Concrete"},
            original_similarity=0.40,
        )
        self.assertLess(score, 0.50)

    def test_rerank_candidates_cross_encoder_mocked(self) -> None:
        candidates = [
            {"rank": 1, "id": "chunk_A", "cosine_similarity": 0.85, "text": "Text A"},
            {"rank": 2, "id": "chunk_B", "cosine_similarity": 0.80, "text": "Text B"},
            {"rank": 3, "id": "chunk_C", "cosine_similarity": 0.75, "text": "Text C"},
        ]

        # Suppose cross-encoder scores chunk_C highest, chunk_A second, chunk_B lowest
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = [1.2, 0.5, 3.8]

        with patch("src.embeddings.rerank._get_cross_encoder", return_value=mock_encoder):
            results = rerank_candidates(
                query="test query",
                candidates=candidates,
                top_k=2,
                method="cross_encoder",
            )

        self.assertEqual(len(results), 2)
        # chunk_C had score 3.8 -> should be rank 1
        self.assertEqual(results[0]["id"], "chunk_C")
        self.assertEqual(results[0]["rerank_rank"], 1)
        self.assertEqual(results[0]["initial_rank"], 3)
        self.assertEqual(results[0]["rank_change"], 2)  # promoted from 3 to 1 (+2)

        # chunk_A had score 1.2 -> should be rank 2
        self.assertEqual(results[1]["id"], "chunk_A")
        self.assertEqual(results[1]["rerank_rank"], 2)
        self.assertEqual(results[1]["initial_rank"], 1)
        self.assertEqual(results[1]["rank_change"], -1)  # demoted from 1 to 2 (-1)

    def test_rerank_candidates_custom_precision(self) -> None:
        candidates = [
            {
                "rank": 1,
                "id": "chunk_weak",
                "cosine_similarity": 0.80,
                "text": "General considerations.",
                "metadata": {},
            },
            {
                "rank": 2,
                "id": "chunk_strong",
                "cosine_similarity": 0.78,
                "text": "Exact clause 4.4.1 dead end corridor maximum travel distance is 6m.",
                "metadata": {"heading": "Part 4"},
            },
        ]

        results = rerank_candidates(
            query="dead end corridor travel distance",
            candidates=candidates,
            top_k=2,
            method="custom_precision",
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "chunk_strong")

    def test_retrieve_and_rerank_validation(self) -> None:
        with self.assertRaises(ValueError):
            retrieve_and_rerank(
                query="test",
                candidate_count=2,
                final_k=5,  # candidate_count < final_k must fail
            )

    def test_generate_markdown_comparison_report(self) -> None:
        sample_data = {
            "candidate_count": 5,
            "final_k": 2,
            "rerank_model": "test-model",
            "benchmarks": [
                {
                    "query_info": {"focus": "test focus"},
                    "result": {
                        "query": "sample query",
                        "candidate_count": 5,
                        "final_k": 2,
                        "metrics": {"promoted_chunks_count": 1},
                        "final_reranked_results": [
                            {
                                "id": "chunk-1",
                                "rerank_rank": 1,
                                "initial_rank": 3,
                                "rank_change": 2,
                                "rerank_score": 4.5,
                                "cosine_similarity": 0.82,
                                "text": "sample text",
                                "metadata": {"heading": "Part 4"},
                            }
                        ],
                        "initial_candidates": [
                            {
                                "id": "chunk-1",
                                "rank": 1,
                                "cosine_similarity": 0.82,
                                "text": "sample text",
                                "metadata": {"heading": "Part 4", "clause": "1.0"},
                            }
                        ],
                    },
                }
            ],
        }

        md = generate_markdown_comparison_report(sample_data)
        self.assertIn("# Chunk Re-Ranking for Precision", md)
        self.assertIn("sample query", md)
        self.assertIn("`chunk-1`", md)


if __name__ == "__main__":
    unittest.main()
