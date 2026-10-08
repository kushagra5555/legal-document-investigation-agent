"""Tests for the global evidence/context budget."""

import json
import unittest
from unittest.mock import patch

from app.graph import apply_evidence_budget, audit_answer, budget_evidence, solve_question


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class CaptureLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def invoke(self, messages):
        self.prompts.append(str(messages[-1].content))
        if "SOLVER CLAIMS ONLY:" in self.prompts[-1]:
            return FakeResponse(json.dumps({
                "approved": True,
                "confidence": 0.9,
                "issues": [],
                "reason": "Supported.",
            }))
        return FakeResponse("A bounded answer [contract.md].")


class ContextBudgetTests(unittest.TestCase):
    def make_evidence(self) -> list[dict]:
        return [
            {
                "source_document": "contract.md" if index % 2 == 0 else "policy.md",
                "excerpt": (
                    "The notice period is 30 days and this is highly relevant. "
                    if index < 4 else "Unrelated background text. "
                ) * 20,
                "page": index + 1,
            }
            for index in range(20)
        ]

    def test_budget_ranks_relevant_items_and_preserves_provenance(self):
        result = apply_evidence_budget(
            "What is the notice period?",
            self.make_evidence(),
            max_items=3,
            max_chars=180,
            max_per_source=2,
        )
        self.assertLessEqual(len(result), 3)
        self.assertLessEqual(sum(len(item["excerpt"]) for item in result), 180)
        self.assertTrue(all("source_document" in item and "page" in item for item in result))
        self.assertTrue(any("notice period" in item["excerpt"].lower() for item in result))

    def test_solver_and_auditor_receive_the_same_bounded_context(self):
        evidence = self.make_evidence()
        state = {
            "question": "What is the notice period?",
            "evidence": evidence,
        }
        state["budgeted_evidence"] = apply_evidence_budget(
            state["question"], evidence, max_items=5, max_chars=200, max_per_source=2
        )
        llm = CaptureLLM()
        with patch("app.graph.build_llm", return_value=llm):
            solve_question(state)
            state["solver_answer"] = "A bounded answer [contract.md]."
            audit_answer(state)

        solver_prompt, auditor_prompt = llm.prompts
        self.assertLessEqual(solver_prompt.count("SOURCE:"), 24)
        self.assertLessEqual(auditor_prompt.count("SOURCE:"), 24)
        self.assertLessEqual(
            sum(len(item["excerpt"]) for item in state["budgeted_evidence"]),
            200,
        )
        self.assertNotIn("Unrelated background text. " * 20, solver_prompt)
        self.assertNotIn("Unrelated background text. " * 20, auditor_prompt)


if __name__ == "__main__":
    unittest.main()
