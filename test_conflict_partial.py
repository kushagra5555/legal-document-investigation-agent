import unittest

from app.graph import (
    _deterministic_audit,
    _requirement_ledger,
    assess_evidence_sufficiency,
    route_after_audit,
    verify_evidence,
)
from web_app import render_answer_text


class ConflictAndPartialAnswerTests(unittest.TestCase):
    def test_explanation_spans_are_complementary_not_conflicting(self):
        state = {
            "question": "Why are standards important and how do standards support development?",
            "investigation_spec": {
                "subject": "standards",
                "event": "",
                "information_needed": "important development",
                "question_type": "reasoning",
                "answer_strategy": "explain_from_evidence",
                "evidence_requirements": [
                    "why standards are important",
                    "how standards support development",
                ],
            },
            "requirement_ledger": [
                {"requirement_id": "R1", "what": "why standards are important", "expected_form": "reason"},
                {"requirement_id": "R2", "what": "how standards support development", "expected_form": "reason"},
            ],
            "candidate_evidence": [
                {"source_document": "report.pdf", "page": 10, "excerpt": "Standards reduce uncertainty and help firms participate in markets."},
                {"source_document": "report.pdf", "page": 35, "excerpt": "Quality infrastructure turns standards into reliable measurements and supports economic development."},
            ],
            "available_documents": [],
            "selected_documents": ["report.pdf"],
            "investigation_history": [{"source": "report.pdf", "page": 10}],
        }
        result = verify_evidence(state)
        self.assertEqual(result["contradictions"], [])
        self.assertFalse(result["insufficient_evidence"])
        self.assertTrue(all(item["skipped"] for item in result["investigation_trace"][-1]["conflict_flags"]))

    def test_date_conflict_is_a_solver_conflict_claim_not_refusal(self):
        state = {
            "question": "When was the company founded?",
            "investigation_spec": {
                "subject": "company", "event": "", "information_needed": "founded",
                "question_type": "factual_extraction", "answer_strategy": "direct",
                "evidence_requirements": ["founding date"],
            },
            "requirement_ledger": [{"requirement_id": "R1", "what": "founding date", "expected_form": "date", "entities": ["company"]}],
            "candidate_evidence": [
                {"source_document": "a.pdf", "page": 1, "excerpt": "The company was founded in 1998."},
                {"source_document": "b.pdf", "page": 1, "excerpt": "The company was founded in 2012."},
            ],
            "available_documents": [], "selected_documents": ["a.pdf", "b.pdf"],
            "investigation_history": [{"source": "a.pdf", "page": 1}],
        }
        result = verify_evidence(state)
        self.assertEqual(result["contradictions"][0]["evidence_ids"], ["E1", "E2"])
        audit = _deterministic_audit({
            **state,
            **result,
            "solver_answer": "The sources state 1998 and 2012.",
            "solver_claims": [{"claim_id": "C1", "text": "The sources state 1998 and 2012.", "type": "conflict", "evidence_ids": ["E1", "E2"]}],
        }, result["verified_evidence"])
        self.assertTrue(audit["approved"])

    def test_partial_support_is_not_blanket_insufficient(self):
        result = assess_evidence_sufficiency({
            "evidence_coverage": {"supported part": {"satisfied": True}, "missing part": {"satisfied": False}},
            "verified_evidence": [{"source_document": "report.pdf", "page": 3, "excerpt": "Supported."}],
            "exhaustion_certificate": {"complete": True, "coverage_complete": True},
            "investigation_gaps": ["Requirement 'missing part' was not verified; searched 3 page/region(s) using strategies: heading_index."],
            "requirement_ledger": [],
            "insufficient_evidence": False,
        })
        self.assertFalse(result["insufficient_evidence"])

    def test_insufficient_evidence_requires_no_verified_evidence(self):
        result = assess_evidence_sufficiency({
            "evidence_coverage": {"part": {"satisfied": True}},
            "verified_evidence": [{"source_document": "report.pdf", "page": 3, "excerpt": "Evidence."}],
            "exhaustion_certificate": {"complete": True, "coverage_complete": True},
            "investigation_gaps": [], "requirement_ledger": [],
        })
        self.assertFalse(result["insufficient_evidence"])

    def test_identical_evidence_does_not_start_another_retry(self):
        evidence = [{"source_document": "report.pdf", "page": 3, "excerpt": "Evidence."}]
        signature = "report.pdf|3|Evidence."
        self.assertEqual(route_after_audit({
            "audit_status": "rejected", "approved": False, "retry_count": 1,
            "verified_evidence": evidence, "last_retry_evidence_signature": signature,
        }), "finalize")

    def test_answer_markup_is_escaped_then_rendered(self):
        rendered = render_answer_text("<script>alert(1)</script>\n**Important**\n1. First\n- Second")
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn("<script>", rendered)
        self.assertIn("<strong>Important</strong>", rendered)
        self.assertIn("<ol>", rendered)
        self.assertIn("<ul>", rendered)
        self.assertNotIn("**", rendered)

    def test_how_connect_is_reasoning_requirement(self):
        ledger = _requirement_ledger(
            "How does standards connect to economic development?",
            {"evidence_requirements": ["how standards connect to economic development"]},
        )
        self.assertEqual(ledger[0]["expected_form"], "reason")


if __name__ == "__main__":
    unittest.main()
