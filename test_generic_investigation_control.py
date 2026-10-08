"""Deterministic checks for the generic bounded-investigation control flow."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.graph import assess_evidence_sufficiency, bounded_exhaustive_search, route_after_sufficiency


class GenericInvestigationControlTests(unittest.TestCase):
    def test_no_verified_evidence_is_not_absence_before_exhaustion(self):
        state = {
            "question": "What is the retention period?",
            "investigation_spec": {"answer_strategy": "direct"},
            "verified_evidence": [],
            "evidence_coverage": {},
            "contradictions": [],
            "retry_count": 2,
            "exhaustive_search_attempted": False,
            "investigation_trace": [],
        }
        result = assess_evidence_sufficiency(state)
        self.assertFalse(result["insufficient_evidence"])
        self.assertEqual(route_after_sufficiency({**state, **result}), "bounded_exhaustive_search")

    def test_bounded_exhaustion_certifies_absence_after_all_sections(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "unseen_policy.md"
            path.write_text(
                "# Retention policy\n\nRecords are retained for seven years.\n\n"
                "## Exceptions\n\nLegal holds suspend deletion.\n",
                encoding="utf-8",
            )
            state = {
                "question": "What is the retention period?",
                "investigation_spec": {
                    "answer_strategy": "direct",
                    "search_concepts": ["retention period"],
                    "evidence_requirements": ["retention period"],
                },
                "selected_documents": [path.name],
                "available_documents": [{"name": path.name, "path": str(path), "pages": None}],
                "document_maps": {path.name: {"sections": [
                    {"label": "Retention policy", "line_start": 1, "line_end": 3},
                    {"label": "Exceptions", "line_start": 5, "line_end": 5},
                ]}},
                "inspected_documents": [],
                "search_paths_attempted": ["structural_map", "targeted_window"],
                "investigation_trace": [],
            }
            result = bounded_exhaustive_search(state)

        self.assertTrue(result["exhaustion_certificate"]["complete"])
        self.assertEqual(result["exhaustion_certificate"]["status"], "EXHAUSTIVE_SEARCH_COMPLETE")
        self.assertEqual(len(result["inspected_documents"]), 2)

        final_state = {
            **state,
            **result,
            "verified_evidence": [],
            "evidence_coverage": {},
            "contradictions": [],
        }
        assessed = assess_evidence_sufficiency(final_state)
        self.assertTrue(assessed["insufficient_evidence"])
        self.assertEqual(assessed["exhaustion_certificate"]["status"], "EXHAUSTED_ABSENT")


if __name__ == "__main__":
    unittest.main()
