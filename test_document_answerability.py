"""Tests for document relevance versus document answerability."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.graph import (
    assess_evidence_sufficiency,
    inspect_selected_documents,
    revise_investigation,
    route_after_sufficiency,
    verify_evidence,
)


class DocumentAnswerabilityTests(unittest.TestCase):
    def test_zero_evidence_creates_gap_and_routes_to_bounded_revision(self):
        state = {
            "question": "Can the employee take parental leave?",
            "investigation_spec": {
                "question_type": "policy_lookup",
                "answer_strategy": "direct",
                "subject": "employee",
                "event": "parental leave",
                "information_needed": "eligibility requirements",
                "evidence_requirements": ["eligibility requirements"],
            },
            "candidate_evidence": [],
            "document_assessments": {
                "leave_policy.md": {
                    "document_relevance": "high",
                    "document_answerability": "unknown",
                }
            },
            "retry_count": 0,
        }
        state.update(verify_evidence(state))
        state.update(assess_evidence_sufficiency(state))

        # A zero-result first pass is not an absence certificate.  The graph
        # must keep investigating and only finalize insufficiency after the
        # bounded exhaustive path.
        self.assertFalse(state["insufficient_evidence"])
        self.assertTrue(state["investigation_gaps"])
        self.assertEqual(route_after_sufficiency(state), "revise_investigation")
        revised = revise_investigation(state)
        self.assertIn("Investigation gap:", revised["investigation_query"])
        self.assertGreater(revised["inspection_window_size"], 1)

    def test_inspected_document_without_match_is_not_declared_unanswerable(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "policy.md"
            path.write_text("## General policy\n\nThe office is open on weekdays.", encoding="utf-8")
            result = inspect_selected_documents({
                "question": "What is the parental leave eligibility rule?",
                "available_documents": [{
                    "name": "policy.md", "path": str(path), "suffix": ".md",
                }],
                "selected_documents": ["policy.md"],
                "investigation_spec": {
                    "information_needed": "parental leave eligibility",
                    "search_concepts": ["parental leave", "eligibility"],
                },
                "document_assessments": {
                    "policy.md": {"document_relevance": "high"}
                },
            })

        assessment = result["document_assessments"]["policy.md"]
        self.assertTrue(assessment["investigated"])
        self.assertEqual(assessment["document_answerability"], "unknown")
        self.assertTrue(assessment["remaining_gaps"])

    def test_verified_evidence_marks_source_as_likely_answerable(self):
        state = {
            "question": "Who is the COO?",
            "investigation_spec": {
                "question_type": "attribute_lookup",
                "answer_strategy": "direct",
                "subject": "organization",
                "information_needed": "name of the COO",
                "target_attribute": "COO",
                "evidence_requirements": ["COO identity"],
            },
            "candidate_evidence": [{
                "source_document": "company.md",
                "page": 2,
                "excerpt": "The COO is John Smith.",
            }],
            "document_assessments": {
                "company.md": {"document_relevance": "high", "investigated": True}
            },
        }
        result = verify_evidence(state)
        assessment = result["document_assessments"]["company.md"]

        self.assertEqual(assessment["document_answerability"], "likely")
        self.assertTrue(assessment["evidence_found"])


if __name__ == "__main__":
    unittest.main()
