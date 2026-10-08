"""Tests for bounded multi-evidence synthesis and coverage."""

import unittest

from app.graph import (
    _fallback_investigation_spec,
    assess_evidence_sufficiency,
    verify_evidence,
)


QUESTION = "What evidence suggests that the person has experience taking projects from requirements to deployment?"


class SynthesisQATests(unittest.TestCase):
    def test_synthesis_can_be_satisfied_collectively(self):
        spec = _fallback_investigation_spec(QUESTION)
        result = verify_evidence({
            "question": QUESTION,
            "investigation_spec": spec,
            "candidate_evidence": [
                {"source_document": "profile.md", "excerpt": "Requirements discovery and workflow mapping for each project.", "page": 1},
                {"source_document": "experience.md", "excerpt": "Implemented Python and Flask systems with API integration.", "page": 2},
                {"source_document": "delivery.md", "excerpt": "Testing, deployment and iteration for systems delivered to clients.", "page": 3},
            ],
        })
        self.assertEqual(spec["question_type"], "synthesis")
        self.assertEqual(len(result["verified_evidence"]), 3)
        self.assertTrue(all(item["satisfied"] for item in result["evidence_coverage"].values()))
        state = {**result, "investigation_spec": spec}
        self.assertFalse(assess_evidence_sufficiency(state)["insufficient_evidence"])

    def test_synthesis_missing_requirement_allows_partial_answer(self):
        spec = _fallback_investigation_spec(QUESTION)
        result = verify_evidence({
            "question": QUESTION,
            "investigation_spec": spec,
            "candidate_evidence": [
                {"source_document": "profile.md", "excerpt": "Requirements discovery and workflow mapping for each project.", "page": 1},
            ],
        })
        state = {**result, "investigation_spec": spec, "exhaustion_certificate": {"complete": True}}
        assessed = assess_evidence_sufficiency(state)
        self.assertFalse(assessed["insufficient_evidence"])
        self.assertTrue(any("requirements" in gap.lower() or "requirement" in gap.lower() for gap in assessed["investigation_gaps"]))

    def test_supporting_evidence_keeps_provenance(self):
        spec = _fallback_investigation_spec(QUESTION)
        result = verify_evidence({
            "question": QUESTION,
            "investigation_spec": spec,
            "candidate_evidence": [{
                "source_document": "project_report.pdf",
                "excerpt": "A system was delivered to a paying client after live execution.",
                "page": 12,
            }],
        })
        self.assertTrue(result["verified_evidence"])
        self.assertEqual(result["verified_evidence"][0]["page"], 12)


if __name__ == "__main__":
    unittest.main()
