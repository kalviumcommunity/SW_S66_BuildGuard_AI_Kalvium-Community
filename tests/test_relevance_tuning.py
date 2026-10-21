"""Unit tests for retrieval relevance tuning module."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.embeddings.relevance_tuning import (
    EXPERIMENT_CONFIGURATIONS,
    TEST_QUERIES,
    RetrievalConfig,
    TestQuery,
    evaluate_query,
    generate_markdown_report,
    select_best_configuration,
)


class TestRelevanceTuning(unittest.TestCase):
    def test_queries_defined_with_ground_truth(self) -> None:
        self.assertGreaterEqual(len(TEST_QUERIES), 5)
        for q in TEST_QUERIES:
            self.assertTrue(q.id)
            self.assertTrue(q.query)
            self.assertTrue(q.expected_chunk_ids)
            self.assertTrue(q.expected_heading)
            self.assertTrue(q.expected_keywords)

    def test_configurations_cover_k_filters_thresholds_hybrid(self) -> None:
        names = {c.name for c in EXPERIMENT_CONFIGURATIONS}
        self.assertIn("baseline_k3", names)
        self.assertIn("baseline_k7", names)
        self.assertIn("vector_threshold_tau75", names)
        self.assertIn("filtered_vector_k5", names)
        self.assertIn("hybrid_filtered_weighted", names)
        self.assertIn("hybrid_filtered_rrf", names)

        # Check parameter differences
        k_values = {c.top_k for c in EXPERIMENT_CONFIGURATIONS}
        self.assertIn(3, k_values)
        self.assertIn(7, k_values)
        self.assertIn(5, k_values)

        has_threshold = any(c.score_threshold is not None for c in EXPERIMENT_CONFIGURATIONS)
        self.assertTrue(has_threshold)

        has_hybrid = any(c.hybrid for c in EXPERIMENT_CONFIGURATIONS)
        self.assertTrue(has_hybrid)

    def test_evaluate_query_metrics_perfect_hit(self) -> None:
        q = TestQuery(
            id="test_q1",
            query="test query",
            category="Test",
            expected_chunk_ids=["chunk_target"],
            expected_heading="Heading A",
            expected_keywords=["test"],
        )
        config = RetrievalConfig(
            name="test_cfg",
            description="test",
            top_k=3,
            use_metadata_filter=False,
        )

        mock_results = {
            "results": [
                {
                    "rank": 1,
                    "id": "chunk_target",
                    "cosine_similarity": 0.88,
                    "score": 0.88,
                    "metadata": {"heading": "Heading A"},
                    "text": "test content",
                },
                {
                    "rank": 2,
                    "id": "chunk_other",
                    "cosine_similarity": 0.70,
                    "score": 0.70,
                    "metadata": {"heading": "Heading B"},
                    "text": "other content",
                },
            ]
        }

        with patch("src.embeddings.relevance_tuning.retrieve_filtered", return_value=mock_results):
            eval_res = evaluate_query(
                test_query=q,
                config=config,
                persist_dir=MagicMock(),
                collection_name="col",
                model_name="model",
                device=None,
            )

        self.assertTrue(eval_res["top_1_hit"])
        self.assertTrue(eval_res["hit_at_k"])
        self.assertEqual(eval_res["first_hit_rank"], 1)
        self.assertEqual(eval_res["reciprocal_rank"], 1.0)
        self.assertEqual(eval_res["domain_precision"], 0.5)

    def test_evaluate_query_metrics_miss(self) -> None:
        q = TestQuery(
            id="test_q2",
            query="test query miss",
            category="Test",
            expected_chunk_ids=["chunk_target"],
            expected_heading="Heading A",
            expected_keywords=["test"],
        )
        config = RetrievalConfig(
            name="test_cfg",
            description="test",
            top_k=2,
            use_metadata_filter=False,
        )

        mock_results = {
            "results": [
                {
                    "rank": 1,
                    "id": "chunk_wrong1",
                    "cosine_similarity": 0.60,
                    "score": 0.60,
                    "metadata": {"heading": "Heading B"},
                    "text": "other",
                },
                {
                    "rank": 2,
                    "id": "chunk_wrong2",
                    "cosine_similarity": 0.55,
                    "score": 0.55,
                    "metadata": {"heading": "Heading C"},
                    "text": "other 2",
                },
            ]
        }

        with patch("src.embeddings.relevance_tuning.retrieve_filtered", return_value=mock_results):
            eval_res = evaluate_query(
                test_query=q,
                config=config,
                persist_dir=MagicMock(),
                collection_name="col",
                model_name="model",
                device=None,
            )

        self.assertFalse(eval_res["top_1_hit"])
        self.assertFalse(eval_res["hit_at_k"])
        self.assertIsNone(eval_res["first_hit_rank"])
        self.assertEqual(eval_res["reciprocal_rank"], 0.0)
        self.assertEqual(eval_res["domain_precision"], 0.0)

    def test_select_best_configuration(self) -> None:
        runs = [
            {
                "config_name": "baseline_k3",
                "config": {"name": "baseline_k3", "top_k": 3, "alpha": None},
                "metrics": {
                    "composite_relevance_score": 60.0,
                    "top_1_hit_rate": 0.50,
                    "top_k_hit_rate": 0.60,
                    "mean_reciprocal_rank": 0.55,
                    "average_domain_precision": 0.70,
                },
            },
            {
                "config_name": "hybrid_filtered_weighted",
                "config": {"name": "hybrid_filtered_weighted", "top_k": 5, "alpha": 0.65},
                "metrics": {
                    "composite_relevance_score": 95.0,
                    "top_1_hit_rate": 1.00,
                    "top_k_hit_rate": 1.00,
                    "mean_reciprocal_rank": 0.95,
                    "average_domain_precision": 1.00,
                },
            },
        ]

        decision = select_best_configuration(runs)
        self.assertEqual(decision["chosen_config_name"], "hybrid_filtered_weighted")
        self.assertGreater(decision["improvements_over_baseline"]["mrr_improvement"], 0)
        self.assertIn("justification", decision)

    def test_generate_markdown_report(self) -> None:
        runs = [
            {
                "config_name": "baseline_k3",
                "config": {
                    "name": "baseline_k3",
                    "top_k": 3,
                    "use_metadata_filter": False,
                    "score_threshold": None,
                    "hybrid": False,
                    "description": "Baseline",
                },
                "metrics": {
                    "composite_relevance_score": 60.0,
                    "top_1_hit_rate": 0.50,
                    "top_k_hit_rate": 0.60,
                    "mean_reciprocal_rank": 0.55,
                    "average_domain_precision": 0.70,
                },
                "query_evaluations": [
                    {"query_id": TEST_QUERIES[0].id, "first_hit_rank": 1}
                ],
            }
        ]
        decision = {
            "chosen_config_name": "baseline_k3",
            "justification": "Test justification text.",
            "ranking_leaderboard": [
                {
                    "rank": 1,
                    "config_name": "baseline_k3",
                    "composite_score": 60.0,
                    "top_1_hit_rate": 0.50,
                    "top_k_hit_rate": 0.60,
                    "mrr": 0.55,
                    "domain_precision": 0.70,
                }
            ],
        }

        md = generate_markdown_report(
            decision=decision,
            all_runs=runs,
            test_queries=[TEST_QUERIES[0]],
        )
        self.assertIn("# Retrieval Relevance Tuning Report", md)
        self.assertIn("Test justification text.", md)
        self.assertIn("`baseline_k3`", md)


if __name__ == "__main__":
    unittest.main()
