import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ["LLM_MAX_RETRIES"] = "1"
os.environ["LLM_BACKOFF_BASE_SECONDS"] = "0"
os.environ["LLM_REQUEST_TIMEOUT_SECONDS"] = "0.2"

from app import graph
from app.llm import LLMCallFailure, invoke_with_resilience


class FakeModel:
    def __init__(self, responses=None, error=None):
        self.responses = list(responses or [])
        self.error = error
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.error:
            raise self.error
        value = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        return SimpleNamespace(content=value)


def base_state():
    evidence = [{"source_document": "example.pdf", "page": 4, "excerpt": "The CEO is Bob."}]
    return {
        "question": "Who is the CEO?",
        "evidence": evidence,
        "verified_evidence": evidence,
        "budgeted_evidence": evidence,
        "evidence_verification": [{"source_document": "example.pdf", "page": 4, "relevant": True}],
        "evidence_coverage": {},
        "requirement_ledger": [],
        "investigation_trace": [{"stage": "evidence_verification"}],
        "solver_answer": "Bob is the CEO. [example.pdf]",
        "solver_claims": [{"claim_id": "C1", "claim": "Bob is the CEO.", "evidence_ids": ["E1"]}],
        "solver_unanswered_requirements": [],
        "retry_count": 0,
    }


class LLMFailureHandlingTests(unittest.TestCase):
    def test_wrapper_retries_rate_limit_and_records_status(self):
        model = FakeModel(error=RuntimeError("rate limit"))
        model.error.status_code = 429
        with self.assertRaises(LLMCallFailure) as caught:
            invoke_with_resilience(model, [], node="auditor")
        self.assertEqual(caught.exception.status, "rate_limited")
        self.assertEqual(model.calls, 2)
        self.assertEqual(caught.exception.http_status, 429)

    def test_wrapper_classifies_empty_response_as_blocked(self):
        model = FakeModel(responses=["", ""])
        with self.assertRaises(LLMCallFailure) as caught:
            invoke_with_resilience(model, [], node="solver")
        self.assertEqual(caught.exception.status, "blocked")
        self.assertEqual(model.calls, 2)

    def test_solver_transport_failure_is_service_error_not_raw_dump(self):
        model = FakeModel(error=TimeoutError("timed out"))
        state = base_state()
        with patch.object(graph, "build_llm", return_value=model):
            result = graph.solve_question(state)
        self.assertTrue(result["solver_failed"])
        self.assertIn("model service failed", result["solver_answer"].lower())
        self.assertNotIn("selected source contains this relevant information", result["solver_answer"].lower())
        self.assertEqual(result["service_error"]["status"], "timeout")

    def test_auditor_transport_failure_falls_back_without_rejection_retry(self):
        model = FakeModel(error=TimeoutError("timed out"))
        state = base_state()
        with patch.object(graph, "build_llm", return_value=model):
            result = graph.audit_answer(state)
        self.assertEqual(result["audit_status"], "fallback_validated")
        self.assertTrue(result["approved"])
        self.assertTrue(result["audit_unavailable"])
        self.assertEqual(graph.route_after_audit({**state, **result}), "finalize")
        self.assertNotEqual(result.get("confidence"), 0.0)

    def test_invalid_json_uses_deterministic_fallback(self):
        model = FakeModel(responses=["not json", "still not json"])
        state = base_state()
        with patch.object(graph, "build_llm", return_value=model):
            result = graph.audit_answer(state)
        self.assertEqual(result["audit_status"], "fallback_validated")
        self.assertTrue(result["approved"])
        self.assertEqual(result["service_error"]["status"], "invalid_json")

    def test_insufficient_evidence_is_not_reported_after_solver_transport_failure(self):
        model = FakeModel(error=TimeoutError("timed out"))
        state = base_state()
        with patch.object(graph, "build_llm", return_value=model):
            solved = graph.solve_question(state)
            audited = graph.audit_answer({**state, **solved})
            final = graph.finalize_answer({**state, **solved, **audited, "insufficient_evidence": True})
        self.assertNotIn("insufficient evidence in the provided documents", final["final_answer"].lower())
        self.assertEqual(final["audit_status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
