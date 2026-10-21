"""Unit tests for metadata filtering and hybrid search."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.embeddings.filtered_retrieval import (
    build_metadata_filter,
    compute_hybrid_score,
    compute_lexical_score,
    compute_rrf_score,
    search_vector_store,
)


class TestFilteredRetrieval(unittest.TestCase):
    def test_build_metadata_filter_single_condition(self) -> None:
        filt = build_metadata_filter(heading="Part 4 — Fire and Life Safety")
        self.assertEqual(filt, {"heading": "Part 4 — Fire and Life Safety"})

    def test_build_metadata_filter_multiple_conditions(self) -> None:
        filt = build_metadata_filter(
            heading="Part 4 — Fire and Life Safety",
            volume=1,
            modality="text",
        )
        self.assertIsNotNone(filt)
        self.assertIn("$and", filt)
        conditions = filt["$and"]
        self.assertEqual(len(conditions), 3)
        self.assertIn({"heading": "Part 4 — Fire and Life Safety"}, conditions)
        self.assertIn({"volume": 1}, conditions)
        self.assertIn({"modality": "text"}, conditions)

    def test_build_metadata_filter_empty(self) -> None:
        self.assertIsNone(build_metadata_filter())

    def test_build_metadata_filter_extra_dict(self) -> None:
        filt = build_metadata_filter(
            volume=1,
            extra_filter={"source": "nationalbuilding01.pdf"},
        )
        self.assertIsNotNone(filt)
        self.assertIn("$and", filt)
        self.assertIn({"volume": 1}, filt["$and"])
        self.assertIn({"source": "nationalbuilding01.pdf"}, filt["$and"])

    def test_compute_lexical_score_token_overlap(self) -> None:
        text = "Fire doors in exits shall have a minimum fire resistance rating."
        score, matches = compute_lexical_score(text, "fire doors rating")
        self.assertGreater(score, 0.5)

    def test_compute_lexical_score_exact_phrase_bonus(self) -> None:
        text = "According to Clause 2.18 Exit Access Corridor, minimum clear width is required."
        score, matches = compute_lexical_score(
            text,
            "Clause 2.18 Exit Access Corridor",
            exact_phrases=["2.18", "Exit Access Corridor"],
        )
        self.assertIn("2.18", matches)
        self.assertIn("Exit Access Corridor", matches)
        self.assertGreaterEqual(score, 0.9)

    def test_compute_lexical_score_no_match(self) -> None:
        text = "Concrete mixture properties and aggregate specifications."
        score, matches = compute_lexical_score(text, "timber columns and wood joints")
        self.assertEqual(score, 0.0)
        self.assertEqual(matches, [])

    def test_compute_hybrid_score(self) -> None:
        # alpha=0.6: 0.6 * 0.8 + 0.4 * 0.5 = 0.48 + 0.20 = 0.68
        score = compute_hybrid_score(0.8, 0.5, alpha=0.6)
        self.assertAlmostEqual(score, 0.68, places=4)

    def test_compute_rrf_score(self) -> None:
        # rank 1 for vector, rank 2 for lexical, k=60
        # 1/(60+1) + 1/(60+2) = 1/61 + 1/62
        expected = (1.0 / 61.0) + (1.0 / 62.0)
        score = compute_rrf_score(1, 2, k=60)
        self.assertAlmostEqual(score, expected, places=5)

    def test_search_vector_store_with_where_filter(self) -> None:
        mock_col = MagicMock()
        mock_col.count.return_value = 100
        mock_col.query.return_value = {
            "ids": [["chunk-1", "chunk-2"]],
            "documents": [["doc 1", "doc 2"]],
            "metadatas": [[{"heading": "Part 4"}, {"heading": "Part 4"}]],
            "distances": [[0.2, 0.3]],
        }

        where_clause = {"heading": "Part 4"}
        results = search_vector_store(
            query_vector=[0.1, 0.2],
            collection=mock_col,
            top_k=2,
            where=where_clause,
        )

        mock_col.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2]],
            n_results=2,
            include=["documents", "metadatas", "distances"],
            where=where_clause,
        )
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "chunk-1")
        self.assertAlmostEqual(results[0]["cosine_similarity"], 0.8)


if __name__ == "__main__":
    unittest.main()
