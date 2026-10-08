"""Offline tests for auditor validation and bounded retry behavior."""

import unittest
from unittest.mock import patch

from app.graph import audit_answer, solve_question


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)

    def invoke(self, messages):
        return FakeResponse(next(self.responses))


BASE_STATE = {
    "question": "What is the notice period?",
    "evidence": [{"source_document": "contract.md", "excerpt": "Notice is 30 days.", "page": None}],
    "solver_answer": "The notice period is 30 days [contract.md].",
}


class AuditorReliabilityTests(unittest.TestCase):
    def test_malformed_first_response_is_retried(self) -> None:
        llm = FakeLLM([
            "Here is the answer: approved!",
            '{"approved": true, "confidence": 0.9, "issues": [], "reason": "Supported."}',
        ])
        with patch("app.graph.build_llm", return_value=llm):
            result = audit_answer(BASE_STATE)
        self.assertTrue(result["approved"])
        self.assertEqual(result["audit_attempts"], 2)

    def test_two_invalid_responses_return_safe_warning(self) -> None:
        llm = FakeLLM(["not json", "still not json"])
        with patch("app.graph.build_llm", return_value=llm):
            result = audit_answer(BASE_STATE)
        self.assertFalse(result["approved"])
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(result["audit_attempts"], 2)
        self.assertIn("validation failed", result["audit_issues"][0])

    def test_auditor_cannot_approve_generic_insufficient_answer_when_evidence_exists(self) -> None:
        state = dict(BASE_STATE)
        state["solver_answer"] = "Insufficient evidence"
        llm = FakeLLM([
            '{"approved": true, "confidence": 0.9, "issues": [], "reason": "Supported."}',
        ])
        with patch("app.graph.build_llm", return_value=llm):
            result = audit_answer(state)
        self.assertFalse(result["approved"])
        self.assertEqual(result["investigation_action"], "verify_specific_claim")

    def test_solver_repairs_a_generic_insufficient_answer_when_evidence_exists(self) -> None:
        llm = FakeLLM([
            "Insufficient evidence",
            "The notice period is 30 days [contract.md].",
        ])
        with patch("app.graph.build_llm", return_value=llm):
            result = solve_question(BASE_STATE)
        self.assertEqual(result["solver_answer"], "The notice period is 30 days [contract.md].")

    def test_solver_uses_source_backed_fallback_after_two_generic_answers(self) -> None:
        llm = FakeLLM(["Insufficient evidence", "Insufficient evidence"])
        with patch("app.graph.build_llm", return_value=llm):
            result = solve_question(BASE_STATE)
        self.assertIn("Notice is 30 days", result["solver_answer"])
        self.assertIn("[contract.md]", result["solver_answer"])


if __name__ == "__main__":
    unittest.main()
