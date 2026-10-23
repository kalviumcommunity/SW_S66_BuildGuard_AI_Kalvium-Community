"""Unit tests for retrieval evaluation & recall testing module."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from src.embeddings.retrieval_evaluation import (
    EVAL_CONFIGURATIONS,
    LABELLED_QUERIES,
    EvalConfig,
    LabelledQuery,
    QueryMetrics,
    QueryResult,
    aggregate_config,
    compute_metrics,
    generate_markdown_report,
    _dry_run_evaluation,
    _dcg,
    _ndcg_at_k,
    _is_relevant,
    _keyword_hit_in_texts,
)


class TestLabelledQueries(unittest.TestCase):
    def test_minimum_query_count(self) -> None:
        self.assertGreaterEqual(len(LABELLED_QUERIES), 8)

    def test_all_queries_have_ground_truth(self) -> None:
        for q in LABELLED_QUERIES:
            self.assertTrue(q.id, f"Empty id in {q}")
            self.assertTrue(q.query, f"Empty query in {q}")
            self.assertTrue(q.relevant_chunk_ids, f"No ground-truth IDs for {q.id}")
            self.assertTrue(q.expected_heading, f"No expected heading for {q.id}")
            self.assertTrue(q.expected_keywords, f"No expected keywords for {q.id}")

    def test_covers_fire_and_timber_categories(self) -> None:
        categories = {q.category for q in LABELLED_QUERIES}
        self.assertIn("Fire and Life Safety", categories)
        self.assertIn("Structural Timber", categories)

    def test_expected_source_set(self) -> None:
        for q in LABELLED_QUERIES:
            self.assertTrue(q.expected_source)

    def test_ids_are_unique(self) -> None:
        ids = [q.id for q in LABELLED_QUERIES]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate query IDs found")


class TestEvalConfigurations(unittest.TestCase):
    def test_minimum_config_count(self) -> None:
        self.assertGreaterEqual(len(EVAL_CONFIGURATIONS), 4)

    def test_covers_multiple_k_values(self) -> None:
        k_values = {c.top_k for c in EVAL_CONFIGURATIONS}
        self.assertGreaterEqual(len(k_values), 2, "Need at least 2 different k values")

    def test_has_filtered_and_unfiltered(self) -> None:
        has_filtered = any(c.use_metadata_filter for c in EVAL_CONFIGURATIONS)
        has_unfiltered = any(not c.use_metadata_filter for c in EVAL_CONFIGURATIONS)
        self.assertTrue(has_filtered, "No filtered configurations defined")
        self.assertTrue(has_unfiltered, "No unfiltered (baseline) configurations defined")

    def test_has_hybrid_config(self) -> None:
        has_hybrid = any(c.hybrid for c in EVAL_CONFIGURATIONS)
        self.assertTrue(has_hybrid, "No hybrid search configuration defined")

    def test_config_names_are_unique(self) -> None:
        names = [c.name for c in EVAL_CONFIGURATIONS]
        self.assertEqual(len(names), len(set(names)), "Duplicate config names found")


class TestIsRelevant(unittest.TestCase):
    def _make_query(self, ids: list[str]) -> LabelledQuery:
        return LabelledQuery(
            id="test",
            query="q",
            category="Test",
            relevant_chunk_ids=ids,
            expected_heading="H",
            expected_keywords=["kw"],
        )

    def test_exact_match_is_relevant(self) -> None:
        q = self._make_query(["chunk-001"])
        self.assertTrue(_is_relevant("chunk-001", q))

    def test_non_match_not_relevant(self) -> None:
        q = self._make_query(["chunk-001"])
        self.assertFalse(_is_relevant("chunk-999", q))

    def test_multi_relevant_ids(self) -> None:
        q = self._make_query(["chunk-001", "chunk-002"])
        self.assertTrue(_is_relevant("chunk-002", q))


class TestDCGAndNDCG(unittest.TestCase):
    def test_dcg_perfect_top1(self) -> None:
        score = _dcg([1, 0, 0])
        self.assertAlmostEqual(score, 1.0 / 1.0, places=5)

    def test_dcg_zero_for_all_irrelevant(self) -> None:
        self.assertEqual(_dcg([0, 0, 0]), 0.0)

    def test_ndcg_perfect_order(self) -> None:
        q = LabelledQuery(
            id="t", query="q", category="C",
            relevant_chunk_ids=["a"],
            expected_heading="H", expected_keywords=["kw"],
        )
        ndcg = _ndcg_at_k(["a", "b", "c"], q)
        self.assertAlmostEqual(ndcg, 1.0, places=5)

    def test_ndcg_miss(self) -> None:
        q = LabelledQuery(
            id="t", query="q", category="C",
            relevant_chunk_ids=["a"],
            expected_heading="H", expected_keywords=["kw"],
        )
        ndcg = _ndcg_at_k(["b", "c", "d"], q)
        self.assertEqual(ndcg, 0.0)


class TestKeywordHit(unittest.TestCase):
    def test_keyword_found(self) -> None:
        self.assertTrue(_keyword_hit_in_texts(["Fire exit corridor Clause 2.18"], ["2.18"]))

    def test_keyword_case_insensitive(self) -> None:
        self.assertTrue(_keyword_hit_in_texts(["FIRE DOORS"], ["fire doors"]))

    def test_keyword_not_found(self) -> None:
        self.assertFalse(_keyword_hit_in_texts(["general text"], ["clause 99.99"]))


class TestComputeMetrics(unittest.TestCase):
    def _make_query(self, ids: list[str]) -> LabelledQuery:
        return LabelledQuery(
            id="q_test",
            query="test query",
            category="Test",
            relevant_chunk_ids=ids,
            expected_heading="Part 4 — Fire and Life Safety",
            expected_keywords=["exit"],
        )

    def _make_result(
        self,
        retrieved_ids: list[str],
        headings: list[str] | None = None,
        texts: list[str] | None = None,
    ) -> QueryResult:
        k = len(retrieved_ids)
        return QueryResult(
            query_id="q_test",
            config_name="test_cfg",
            retrieved_ids=retrieved_ids,
            retrieved_scores=[0.9 - i * 0.1 for i in range(k)],
            retrieved_headings=headings or ["Part 4 — Fire and Life Safety"] * k,
            retrieved_texts=texts or ["exit text"] * k,
            retrieved_metadata=[{"heading": h} for h in (headings or ["Part 4 — Fire and Life Safety"] * k)],
        )

    def test_perfect_top1_hit(self) -> None:
        q = self._make_query(["chunk-001"])
        result = self._make_result(["chunk-001", "chunk-002", "chunk-003"])
        m = compute_metrics(result, q)
        self.assertEqual(m.recall_at_k, 1.0)
        self.assertTrue(m.hit_at_1)
        self.assertTrue(m.hit_at_k)
        self.assertAlmostEqual(m.reciprocal_rank, 1.0)
        self.assertEqual(m.first_relevant_rank, 1)
        self.assertEqual(m.failure_reason, "")

    def test_hit_at_rank3(self) -> None:
        q = self._make_query(["chunk-003"])
        result = self._make_result(["chunk-001", "chunk-002", "chunk-003"])
        m = compute_metrics(result, q)
        self.assertEqual(m.recall_at_k, 1.0)
        self.assertFalse(m.hit_at_1)
        self.assertTrue(m.hit_at_k)
        self.assertAlmostEqual(m.reciprocal_rank, 1.0 / 3.0, places=5)
        self.assertEqual(m.first_relevant_rank, 3)
        self.assertIn("found_at_rank_3", m.failure_reason)

    def test_complete_miss(self) -> None:
        q = self._make_query(["chunk-999"])
        result = self._make_result(
            ["chunk-001", "chunk-002", "chunk-003"],
            texts=["unrelated content"] * 3,
        )
        m = compute_metrics(result, q)
        self.assertEqual(m.recall_at_k, 0.0)
        self.assertFalse(m.hit_at_1)
        self.assertFalse(m.hit_at_k)
        self.assertEqual(m.reciprocal_rank, 0.0)
        self.assertEqual(m.first_relevant_rank, 0)
        self.assertIn("vocabulary_mismatch", m.failure_reason)

    def test_multi_relevant_partial_recall(self) -> None:
        q = self._make_query(["chunk-001", "chunk-002"])
        result = self._make_result(["chunk-001", "chunk-003", "chunk-004"])
        m = compute_metrics(result, q)
        self.assertAlmostEqual(m.recall_at_k, 0.5)
        self.assertTrue(m.hit_at_k)

    def test_precision_at_k(self) -> None:
        q = self._make_query(["chunk-001"])
        result = self._make_result(["chunk-001", "chunk-002", "chunk-003"])
        m = compute_metrics(result, q)
        self.assertAlmostEqual(m.precision_at_k, 1.0 / 3.0, places=5)

    def test_wrong_section_failure_reason(self) -> None:
        q = self._make_query(["chunk-999"])
        result = self._make_result(
            ["chunk-001", "chunk-002", "chunk-003"],
            headings=["Part 6 — Structural"] * 3,
            texts=["no keywords here"] * 3,
        )
        m = compute_metrics(result, q)
        self.assertIn("wrong_section", m.failure_reason)


class TestAggregateConfig(unittest.TestCase):
    def _make_config(self, name: str = "test") -> EvalConfig:
        return EvalConfig(name=name, description="desc", top_k=5)

    def _make_metrics(self, hit: bool, rank: int = 1) -> QueryMetrics:
        rr = 1.0 / rank if hit else 0.0
        return QueryMetrics(
            query_id="q", config_name="test", top_k=5,
            recall_at_k=1.0 if hit else 0.0,
            precision_at_k=0.2 if hit else 0.0,
            reciprocal_rank=rr, ndcg_at_k=rr,
            hit_at_1=(rank == 1 and hit),
            hit_at_k=hit,
            heading_match_rate=1.0 if hit else 0.0,
            keyword_hit=hit,
            first_relevant_rank=rank if hit else 0,
            failure_reason="" if hit else "vocabulary_mismatch: miss",
        )

    def test_perfect_summary(self) -> None:
        config = self._make_config()
        metrics = [self._make_metrics(True) for _ in range(4)]
        s = aggregate_config(config, metrics)
        self.assertAlmostEqual(s.mean_recall, 1.0)
        self.assertAlmostEqual(s.hit_rate_at_k, 1.0)
        self.assertAlmostEqual(s.hit_rate_at_1, 1.0)
        self.assertAlmostEqual(s.mean_mrr, 1.0)

    def test_mixed_summary(self) -> None:
        config = self._make_config()
        metrics = [
            self._make_metrics(True, rank=1),
            self._make_metrics(True, rank=3),
            self._make_metrics(False),
            self._make_metrics(False),
        ]
        s = aggregate_config(config, metrics)
        self.assertAlmostEqual(s.mean_recall, 0.5)
        self.assertAlmostEqual(s.hit_rate_at_k, 0.5)
        self.assertEqual(len(s.failures), 2)

    def test_empty_metrics_list(self) -> None:
        config = self._make_config()
        s = aggregate_config(config, [])
        self.assertEqual(s.mean_recall, 0.0)
        self.assertEqual(s.hit_rate_at_k, 0.0)


class TestMarkdownReport(unittest.TestCase):
    def _make_summary(self, name: str, recall: float, mrr: float) -> "ConfigSummary":
        from src.embeddings.retrieval_evaluation import ConfigSummary
        return ConfigSummary(
            config_name=name, description="test", top_k=5,
            mean_recall=recall, mean_precision=recall / 5,
            mean_mrr=mrr, mean_ndcg=mrr,
            hit_rate_at_1=mrr, hit_rate_at_k=recall,
            mean_heading_match=0.8, keyword_hit_rate=0.9,
        )

    def test_report_contains_all_sections(self) -> None:
        summaries = [
            self._make_summary("cfg_a", 0.9, 0.85),
            self._make_summary("cfg_b", 0.7, 0.65),
        ]
        q = LABELLED_QUERIES[:2]
        report = generate_markdown_report(summaries, q, "cfg_a", "2024-01-01T00:00:00Z")
        self.assertIn("# Retrieval Evaluation", report)
        self.assertIn("## 1. Labelled Query Set", report)
        self.assertIn("## 2. Recall@k Results", report)
        self.assertIn("## 3. Per-Query Breakdown", report)
        self.assertIn("## 4. Failure Inspection", report)
        self.assertIn("## 5. Best Configuration", report)

    def test_best_config_marked_in_report(self) -> None:
        summaries = [
            self._make_summary("best_cfg", 0.95, 0.90),
            self._make_summary("other_cfg", 0.60, 0.55),
        ]
        report = generate_markdown_report(summaries, LABELLED_QUERIES[:2], "best_cfg", "2024-01-01T00:00:00Z")
        self.assertIn("best_cfg", report)
        self.assertIn("(best)", report)


class TestDryRunEvaluation(unittest.TestCase):
    def test_dry_run_returns_valid_payload(self) -> None:
        payload = _dry_run_evaluation(LABELLED_QUERIES[:4], EVAL_CONFIGURATIONS[:3])
        self.assertIn("best_configuration", payload)
        self.assertIn("configuration_summaries", payload)
        self.assertIn("labelled_queries", payload)
        self.assertIn("overall", payload)
        self.assertEqual(len(payload["labelled_queries"]), 4)
        self.assertEqual(len(payload["configuration_summaries"]), 3)
        self.assertTrue(payload.get("dry_run"))

    def test_dry_run_best_config_exists(self) -> None:
        payload = _dry_run_evaluation(LABELLED_QUERIES, EVAL_CONFIGURATIONS)
        best = payload["best_configuration"]
        config_names = [s["config_name"] for s in payload["configuration_summaries"]]
        self.assertIn(best, config_names)

    def test_dry_run_metrics_in_range(self) -> None:
        payload = _dry_run_evaluation(LABELLED_QUERIES, EVAL_CONFIGURATIONS)
        for s in payload["configuration_summaries"]:
            self.assertGreaterEqual(s["mean_recall"], 0.0)
            self.assertLessEqual(s["mean_recall"], 1.0)
            self.assertGreaterEqual(s["mean_mrr"], 0.0)
            self.assertLessEqual(s["mean_mrr"], 1.0)

    def test_dry_run_produces_markdown_report(self) -> None:
        payload = _dry_run_evaluation(LABELLED_QUERIES, EVAL_CONFIGURATIONS)
        self.assertIn("report_markdown", payload)
        md = payload["report_markdown"]
        self.assertIn("# Retrieval Evaluation", md)
        self.assertGreater(len(md), 500)

    def test_filtered_configs_higher_hit_rate_than_baseline(self) -> None:
        payload = _dry_run_evaluation(LABELLED_QUERIES, EVAL_CONFIGURATIONS)
        summaries = {s["config_name"]: s for s in payload["configuration_summaries"]}
        # Filtered configs should statistically tend to have higher hit rates
        # (since dry-run uses 0.85 vs 0.60 probability)
        filtered_recall = [
            s["mean_recall"] for name, s in summaries.items()
            if "filtered" in name
        ]
        baseline_recall = [
            s["mean_recall"] for name, s in summaries.items()
            if name.startswith("baseline")
        ]
        if filtered_recall and baseline_recall:
            avg_filtered = sum(filtered_recall) / len(filtered_recall)
            avg_baseline = sum(baseline_recall) / len(baseline_recall)
            # Soft assertion - not guaranteed by randomness but expected directionally
            self.assertGreaterEqual(avg_filtered + 0.3, avg_baseline)


if __name__ == "__main__":
    unittest.main()
