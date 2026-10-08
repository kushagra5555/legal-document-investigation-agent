import json
import unittest
from unittest.mock import patch

from app import graph


class FakeResponse:
    def __init__(self, content):
        self.content = content


def _state():
    evidence = [{
        "source_document": "article.pdf",
        "page": 1,
        "excerpt": "HON'BLE THE CHIEF JUSTICE DY CHANDRACHUD",
    }]
    return {
        "question": "What is the name of the Chief Justice mentioned on the first page?",
        "investigation_spec": {
            "question_type": "attribute_lookup",
            "target_attribute": "name",
            "direct_identification_required": True,
            "target_entity": "Chief Justice",
            "evidence_requirements": ["the name of the Chief Justice"],
        },
        "verified_evidence": evidence,
        "evidence": evidence,
        "budgeted_evidence": evidence,
        "evidence_coverage": {"the name of the Chief Justice": {"satisfied": True}},
        "requirement_ledger": [],
        "evidence_verification": [],
        "contradictions": [],
        "investigation_trace": [{"stage": "targeted_inspection"}],
        "solver_claims": [],
        "solver_unanswered_requirements": [],
        "retry_count": 0,
    }


class RegressionAuditTests(unittest.TestCase):
    def test_single_document_solver_repair_preserves_complete_answer(self):
        responses = iter([
            FakeResponse('```json\n{"answer_text":"Based on the transcript, the Chief Justice is Chief Justice D.Y'),
            FakeResponse(json.dumps({
                "answer_text": "The Chief Justice mentioned is Chief Justice D.Y. Chandrachud.",
                "claims": [{"claim_id": "C1", "text": "The Chief Justice is D.Y. Chandrachud.", "evidence_ids": ["E1"], "type": "quoted"}],
                "unanswered_requirements": [],
            })),
        ])

        with patch.object(graph, "build_llm", return_value=object()), patch.object(
            graph, "invoke_with_resilience", side_effect=lambda *args, **kwargs: (next(responses), {"status": "ok"})
        ):
            result = graph.solve_question(_state())

        self.assertEqual(result["solver_answer"], "The Chief Justice mentioned is Chief Justice D.Y. Chandrachud.")
        trace = result["investigation_trace"][-1]
        self.assertIn("D.Y. Chandrachud", trace["raw_solver_response"])
        self.assertGreater(trace["raw_solver_response_length"], 0)

    def test_raw_solver_output_is_not_shortened_by_parser(self):
        raw = json.dumps({"answer_text": "A complete answer with the full entity name.", "claims": []})
        answer, claims, unanswered = graph._parse_solver_payload(raw)
        self.assertEqual(answer, "A complete answer with the full entity name.")
        self.assertEqual(claims, [])
        self.assertEqual(unanswered, [])

    def test_evidence_budget_preserves_provenance_and_does_not_corrupt_text(self):
        evidence = [
            {"source_document": "a.pdf", "page": 4, "excerpt": "A complete quoted passage."},
            {"source_document": "b.pdf", "page": 4, "excerpt": "Another complete quoted passage."},
        ]
        packed = graph.apply_evidence_budget("What is the answer?", evidence, max_items=2, max_chars=500)
        self.assertEqual({item["source_document"] for item in packed}, {"a.pdf", "b.pdf"})
        self.assertEqual(packed[0]["excerpt"], "A complete quoted passage.")

    def test_auditor_receives_same_bounded_evidence_as_solver(self):
        captured = []
        auditor_json = json.dumps({"approved": True, "confidence": 1, "issues": [], "reason": "Supported", "investigation_action": ""})

        def fake_invoke(llm, messages, **kwargs):
            captured.append(messages[-1].content)
            return FakeResponse(auditor_json), {"status": "ok"}

        state = _state()
        state["solver_answer"] = "The Chief Justice mentioned is Chief Justice D.Y. Chandrachud."
        state["solver_claims"] = [{"claim_id": "C1", "claim": "The Chief Justice is D.Y. Chandrachud.", "evidence_ids": ["E1"]}]
        with patch.object(graph, "build_llm", return_value=object()), patch.object(graph, "invoke_with_resilience", side_effect=fake_invoke):
            result = graph.audit_answer(state)
        self.assertTrue(result["approved"])
        self.assertIn("HON'BLE THE CHIEF JUSTICE DY CHANDRACHUD", captured[0])

    def test_rejected_retry_cannot_overwrite_approved_candidate(self):
        state = _state()
        state.update({
            "solver_answer": "The Chief Justice is D.Y. Chandrachud.",
            "solver_claims": [{"claim_id": "C1", "claim": "The Chief Justice is D.Y. Chandrachud.", "evidence_ids": ["E1"]}],
            "best_solver_answer": "The Chief Justice is D.Y. Chandrachud.",
            "best_solver_claims": [{"claim_id": "C1", "claim": "The Chief Justice is D.Y. Chandrachud.", "evidence_ids": ["E1"]}],
            "best_audit_status": "approved",
            "best_confidence": 1.0,
            "audit_status": "rejected",
            "approved": False,
        })
        result = graph.finalize_answer(state)
        self.assertTrue(result["approved"])
        self.assertEqual(result["final_answer"], "The Chief Justice is D.Y. Chandrachud.")

    def test_state_update_does_not_reset_unrelated_fields(self):
        state = _state()
        state["investigation_spec"] = {"answer_strategy": "direct"}
        update = graph.revise_investigation(state)
        merged = {**state, **update}
        self.assertEqual(merged["verified_evidence"], state["verified_evidence"])
        self.assertEqual(merged["investigation_spec"]["answer_strategy"], "direct")


if __name__ == "__main__":
    unittest.main()
