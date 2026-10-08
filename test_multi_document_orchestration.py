"""Offline regression tests for corpus-wide document investigation."""

from pathlib import Path
from tempfile import TemporaryDirectory
import os
import unittest
from unittest.mock import patch

from app.graph import (
    apply_evidence_budget,
    bounded_exhaustive_search,
    extract_evidence,
    verify_evidence,
)


def _spec(*requirements):
    return {
        "answer_strategy": "aggregate_evidence" if len(requirements) > 1 else "direct",
        "subject": "company",
        "event": None,
        "information_needed": "; ".join(requirements),
        "search_concepts": ["company", "answer", *requirements],
        "evidence_requirements": list(requirements),
        "question_type": "factual_extraction",
    }


class MultiDocumentOrchestrationTests(unittest.TestCase):
    def test_relevant_evidence_survives_unrelated_selected_sources(self):
        evidence = [
            {"source_document": "unrelated.pdf", "page": page, "excerpt": "General background text."}
            for page in range(1, 70)
        ] + [{"source_document": "answer.pdf", "page": 5, "excerpt": "The company headquarters is Pune."}]
        bounded = apply_evidence_budget("Where is the company headquarters?", evidence, max_items=8, max_chars=4000, max_per_source=8)
        self.assertIn("answer.pdf", {item["source_document"] for item in bounded})

    def test_different_requirements_are_verified_from_different_documents(self):
        result = verify_evidence({
            "question": "What is the headquarters and founding year?",
            "investigation_spec": _spec("headquarters", "founding year"),
            "candidate_evidence": [
                {"source_document": "company_profile.md", "page": 1, "excerpt": "The company headquarters is Pune."},
                {"source_document": "company_history.md", "page": 4, "excerpt": "The company was founded in 2012."},
            ],
            "selected_documents": ["company_profile.md", "company_history.md"],
        })
        self.assertEqual({item["source_document"] for item in result["verified_evidence"]}, {"company_profile.md", "company_history.md"})
        self.assertEqual(set(result["documents_with_verified_evidence"]), {"company_profile.md", "company_history.md"})

    def test_same_page_number_keeps_document_identity(self):
        bounded = apply_evidence_budget("What is the answer?", [
            {"source_document": "a.pdf", "page": 5, "excerpt": "The answer in A is alpha."},
            {"source_document": "b.pdf", "page": 5, "excerpt": "The answer in B is beta."},
        ], max_items=4, max_chars=1000, max_per_source=4)
        self.assertEqual({(item["source_document"], item["page"]) for item in bounded}, {("a.pdf", 5), ("b.pdf", 5)})

    def test_round_robin_exhaustive_search_gives_each_document_a_turn(self):
        with TemporaryDirectory() as folder:
            d1 = Path(folder) / "large_unrelated.md"
            d2 = Path(folder) / "small_answer.md"
            d1.write_text("\n".join(f"## Section {i}\nUnrelated material {i}." for i in range(1, 6)), encoding="utf-8")
            d2.write_text("## Answer\nThe headquarters is Pune.", encoding="utf-8")
            maps = {
                d1.name: {"sections": [{"label": f"Section {i}", "line_start": (i - 1) * 2 + 1, "line_end": i * 2} for i in range(1, 6)]},
                d2.name: {"sections": [{"label": "Answer", "line_start": 1, "line_end": 2}]},
            }
            state = {
                "question": "Where is the company headquarters?",
                "investigation_spec": _spec("headquarters"),
                "selected_documents": [d1.name, d2.name],
                "available_documents": [{"name": d1.name, "path": str(d1), "pages": None}, {"name": d2.name, "path": str(d2), "pages": None}],
                "document_maps": maps,
                "inspected_documents": [],
                "search_paths_attempted": [],
                "investigation_trace": [],
            }
            with patch.dict(os.environ, {"MAX_EXHAUSTIVE_PAGES": "2", "EXHAUSTIVE_BATCH_PAGES": "1"}, clear=False):
                result = bounded_exhaustive_search(state)
                extracted = extract_evidence({**state, **result})

        self.assertEqual({region["source"] for region in result["exhaustion_certificate"]["regions_inspected"]}, {d1.name, d2.name})
        self.assertIn(d2.name, {item["source_document"] for item in extracted["candidate_evidence"]})
        self.assertFalse(result["exhaustion_certificate"]["coverage_complete"])

    def test_provenance_survives_similar_claims(self):
        result = verify_evidence({
            "question": "What is the revenue?",
            "investigation_spec": _spec("revenue"),
            "candidate_evidence": [
                {"source_document": "2023.md", "page": 5, "excerpt": "The company revenue was 10 million."},
                {"source_document": "2024.md", "page": 5, "excerpt": "The company revenue was 12 million."},
            ],
        })
        self.assertEqual({(item["source_document"], item["page"]) for item in result["verified_evidence"]}, {("2023.md", 5), ("2024.md", 5)})


if __name__ == "__main__":
    unittest.main()
