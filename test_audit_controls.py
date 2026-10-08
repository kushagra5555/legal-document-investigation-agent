import unittest
from types import SimpleNamespace

from app.graph import (
    _claim_verdict_rows,
    _deterministic_audit,
    _response_was_truncated,
)


class AuditControlTests(unittest.TestCase):
    def test_rejected_result_has_no_fixed_confidence(self):
        state = {
            "solver_answer": "The answer is unsupported.",
            "solver_claims": [{
                "claim_id": "C1",
                "claim": "The answer is unsupported.",
                "type": "factual",
                "evidence_ids": [],
            }],
            "requirement_ledger": [{"requirement_id": "R1", "what": "answer", "expected_form": "fact"}],
        }
        result = _deterministic_audit(state, [])
        self.assertFalse(result["approved"])
        self.assertEqual(result["confidence"], 0.0)

    def test_near_verbatim_claim_is_entailed_without_llm(self):
        evidence = [{
            "source_document": "report.pdf",
            "page": 35,
            "excerpt": "Standards support economic development by reducing uncertainty.",
        }]
        state = {
            "solver_claims": [{
                "claim_id": "C1",
                "claim": "Standards support economic development by reducing uncertainty.",
                "type": "factual",
                "evidence_ids": ["E1"],
            }],
            "requirement_ledger": [{"requirement_id": "R1", "what": "support", "expected_form": "fact"}],
        }
        rows = _claim_verdict_rows(state, evidence)
        self.assertEqual(rows[0]["verdict"], "ENTAILED")
        self.assertEqual(rows[0]["cited_quotes"][0]["quote"], evidence[0]["excerpt"])

    def test_max_tokens_response_is_marked_truncated(self):
        response = SimpleNamespace(response_metadata={"finish_reason": "MAX_TOKENS"})
        self.assertTrue(_response_was_truncated(response))

    def test_unproven_refusal_is_not_approved(self):
        state = {
            "solver_answer": "Insufficient evidence in the selected sources.",
            "solver_claims": [],
            "exhaustion_certificate": {"complete": False},
            "coverage_complete": False,
        }
        result = _deterministic_audit(state, [])
        self.assertFalse(result["approved"])

    def test_completed_refusal_can_be_audited_without_confidence_claim(self):
        state = {
            "solver_answer": "Insufficient evidence in the selected sources.",
            "solver_claims": [],
            "exhaustion_certificate": {"complete": True},
            "coverage_complete": True,
        }
        result = _deterministic_audit(state, [])
        self.assertTrue(result["approved"])
        self.assertIsNone(result["confidence"])


if __name__ == "__main__":
    unittest.main()
