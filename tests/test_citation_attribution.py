"""Unit tests for Source Citation and Attribution module.

Validates:
1. In-text citation parsing and tag mapping ([1], [2]).
2. Metadata extraction (source_document, volume, page range, clause, section_heading, chunk_id).
3. Automated claim verification against source chunks.
4. Detection of invalid or fabricated citations.
5. No-source fallback mechanism with zero citations.
6. Artifact generation and saving.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from src.citation_attribution import (
    CitationVerificationReport,
    CitedAnswerResult,
    ClaimAttribution,
    SourceMetadata,
    build_citations_map,
    generate_cited_answer,
    save_citation_artifacts,
    verify_citations,
    verify_claim_against_source,
)


class TestCitationAttribution(unittest.TestCase):
    """Test suite for Citation and Attribution verification."""

    def setUp(self):
        self.sample_chunks = [
            {
                "id": "chunk_doc1_001",
                "text": "For Type 1 fire-resistive construction, exterior load-bearing walls must achieve a 2-hour fire-resistance rating (120 minutes) and automatic sprinklers are mandatory throughout.",
                "similarity": 0.88,
                "score": 0.88,
                "metadata": {
                    "source_doc": "NBC_2016_Vol1.pdf",
                    "source": "NBC_2016_Vol1.pdf",
                    "volume": "Volume 1",
                    "page_start": 142,
                    "page_end": 143,
                    "clause": "Clause 3.4.1",
                    "heading": "Fire Resistance Requirements",
                    "chunk_index": 12,
                },
            },
            {
                "id": "chunk_doc2_005",
                "text": "The minimum stair width for commercial high-occupancy office buildings is 1.5 meters, with maximum handrail height of 1000 mm.",
                "similarity": 0.81,
                "score": 0.81,
                "metadata": {
                    "source_doc": "NBC_2016_Vol2.pdf",
                    "source": "NBC_2016_Vol2.pdf",
                    "volume": "Volume 2",
                    "page_start": 314,
                    "page_end": 315,
                    "clause": "Clause 4.2.8",
                    "heading": "Means of Egress and Staircases",
                    "chunk_index": 55,
                },
            },
        ]

    def test_build_citations_map(self):
        """Task 1 & 2: Citations map accurately extracts source tags and metadata."""
        c_map = build_citations_map(self.sample_chunks)
        self.assertEqual(len(c_map), 2)
        self.assertIn("[1]", c_map)
        self.assertIn("[2]", c_map)

        meta1 = c_map["[1]"]
        self.assertEqual(meta1.chunk_id, "chunk_doc1_001")
        self.assertEqual(meta1.source_document, "NBC_2016_Vol1.pdf")
        self.assertEqual(meta1.volume, "Volume 1")
        self.assertEqual(meta1.page, "142-143")
        self.assertEqual(meta1.clause, "Clause 3.4.1")
        self.assertEqual(meta1.section_heading, "Fire Resistance Requirements")
        self.assertAlmostEqual(meta1.similarity_score, 0.88)

        meta2 = c_map["[2]"]
        self.assertEqual(meta2.chunk_id, "chunk_doc2_005")
        self.assertEqual(meta2.source_document, "NBC_2016_Vol2.pdf")
        self.assertEqual(meta2.page, "314-315")
        self.assertEqual(meta2.clause, "Clause 4.2.8")

    def test_verify_claim_against_source_supported(self):
        """Task 3: Supported claim returns verified true."""
        source_meta = SourceMetadata(
            marker_id=1,
            marker_tag="[1]",
            chunk_id="c1",
            source_document="NBC_2016.pdf",
            volume="Vol 1",
            page="142",
            clause="Clause 3.4.1",
            section_heading="Fire Safety",
            similarity_score=0.88,
            full_text="Exterior load-bearing walls must achieve a 2-hour fire-resistance rating (120 minutes) and automatic sprinklers are mandatory.",
            snippet="Exterior load-bearing walls must achieve a 2-hour...",
        )

        claim_text = "Exterior load-bearing walls must achieve a 2-hour fire-resistance rating."
        is_supported, snippet, confidence = verify_claim_against_source(claim_text, source_meta)

        self.assertTrue(is_supported)
        self.assertGreater(confidence, 0.5)
        self.assertTrue(len(snippet) > 0)

    def test_verify_claim_against_source_unsupported(self):
        """Task 3: Claim not supported by the source returns false."""
        source_meta = SourceMetadata(
            marker_id=1,
            marker_tag="[1]",
            chunk_id="c1",
            source_document="NBC_2016.pdf",
            volume="Vol 1",
            page="142",
            clause="Clause 3.4.1",
            section_heading="Fire Safety",
            similarity_score=0.88,
            full_text="Exterior load-bearing walls must achieve a 2-hour fire-resistance rating and sprinklers are mandatory.",
            snippet="Exterior load-bearing walls...",
        )

        claim_text = "The minimum stair width for office buildings is 3.5 meters with titanium handrails."
        is_supported, snippet, confidence = verify_claim_against_source(claim_text, source_meta)

        self.assertFalse(is_supported)
        self.assertLess(confidence, 0.4)

    def test_verify_citations_fabricated_tag(self):
        """Task 4: Detects fabricated citation tags that do not exist in retrieved sources."""
        answer = "Office stair width is 1.5 meters [2], while solar panels are mandatory under municipal law [99]."

        report = verify_citations(answer, self.sample_chunks)

        self.assertFalse(report.is_fully_verified)
        self.assertIn("[99]", report.fabricated_citations)
        fabricated_attributions = [a for a in report.claim_attributions if "[99]" in a.citations]
        self.assertTrue(len(fabricated_attributions) > 0)
        self.assertEqual(fabricated_attributions[0].verification_status, "FABRICATED_CITATION")

    def test_generate_cited_answer_confident_grounded(self):
        """Task 1, 2, 3: Generates cited answer with verified sources and full audit report."""
        query = "What is the fire resistance rating for walls separating a boiler room?"
        result = generate_cited_answer(
            query=query,
            chunks=self.sample_chunks,
            min_similarity_threshold=0.5,
            offline=True,
        )

        self.assertFalse(result.is_fallback)
        self.assertEqual(result.mode, "cited_answer")
        self.assertGreater(len(result.citations_map), 0)
        self.assertIn("[1]", result.answer)
        self.assertIsNotNone(result.verification_report)
        self.assertTrue(result.verification_report.is_fully_verified)
        self.assertEqual(len(result.verification_report.fabricated_citations), 0)

    def test_generate_cited_answer_no_source_fallback(self):
        """Task 4: Answers with insufficient supporting sources return fallback with zero citations."""
        query = "What are the rules for space elevator launchpads in Mumbai?"
        # Case A: empty chunks
        result_empty = generate_cited_answer(
            query=query,
            chunks=[],
            offline=True,
        )
        self.assertTrue(result_empty.is_fallback)
        self.assertEqual(result_empty.mode, "no_source_fallback")
        self.assertEqual(len(result_empty.citations_map), 0)
        self.assertIn("don't have enough information", result_empty.answer.lower())
        self.assertNotIn("[1]", result_empty.answer)

        # Case B: all chunks below threshold
        low_sim_chunks = [
            {
                "id": "weak_chunk_1",
                "text": "Gardening techniques in urban environments.",
                "similarity": 0.25,
                "score": 0.25,
                "metadata": {"source_doc": "Garden.pdf"},
            }
        ]
        result_weak = generate_cited_answer(
            query=query,
            chunks=low_sim_chunks,
            min_similarity_threshold=0.55,
            offline=True,
        )
        self.assertTrue(result_weak.is_fallback)
        self.assertEqual(result_weak.mode, "no_source_fallback")
        self.assertEqual(len(result_weak.citations_map), 0)
        self.assertIn("information", result_weak.answer.lower())

    def test_save_citation_artifacts(self):
        """Task 5: Artifact saving persists JSON and Markdown verification reports."""
        c_map = build_citations_map(self.sample_chunks)
        cited_res = CitedAnswerResult(
            query="Sample query",
            answer="Sample answer [1].",
            citations_map=c_map,
            verification_report=CitationVerificationReport(
                total_claims=1,
                verified_claims=1,
                unsupported_claims=0,
                fabricated_citations=[],
                overall_accuracy=1.0,
                is_fully_verified=True,
                summary="All claims verified.",
                claim_attributions=[],
            ),
            mode="cited_answer",
            is_fallback=False,
            supporting_chunks=self.sample_chunks,
        )
        fallback_res = CitedAnswerResult(
            query="Fallback query",
            answer="Fallback answer",
            citations_map={},
            verification_report=None,
            mode="no_source_fallback",
            is_fallback=True,
            supporting_chunks=[],
        )

        temp_dir = tempfile.mkdtemp()
        try:
            json_path, md_path = save_citation_artifacts(
                cited_result=cited_res,
                fallback_result=fallback_res,
                output_dir=Path(temp_dir),
            )
            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(md_path))

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("sample_cited_answer", data)
            self.assertEqual(data["sample_cited_answer"]["query"], "Sample query")

            with open(md_path, "r", encoding="utf-8") as f:
                md_content = f.read()
            self.assertIn("Source Citation & Attribution Audit Report", md_content)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
