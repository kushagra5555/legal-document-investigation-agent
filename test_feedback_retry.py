"""Tests for auditor-feedback-driven re-investigation."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from app.graph import GRAPH_CONFIG, MAX_RETRIES, build_graph


class FakeResponse:
    def __init__(self, content: str):
        self.content = content


class SequencedLLM:
    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def invoke(self, messages):
        prompt = str(messages[-1].content) if isinstance(messages, list) else str(messages)
        self.prompts.append(prompt)
        return FakeResponse(next(self.responses))


def audit_json(approved: bool, reason: str) -> str:
    return json.dumps({
        "approved": approved,
        "confidence": 0.95 if approved else 0.25,
        "issues": [] if approved else [reason],
        "reason": reason,
    })


class FeedbackRetryGraphTests(unittest.TestCase):
    def run_graph(self, llm_responses: list[str]):
        llm = SequencedLLM(llm_responses)
        inspect_calls: list[dict] = []

        inspect_tool = Mock()

        def inspect(payload):
            inspect_calls.append(payload)
            window_size = payload["window_size"]
            return {
                "source": "contract.md",
                "document_type": "markdown",
                "content": f"employee resignation notice period evidence-round-{window_size}",
                "blocks": [{
                    "source": "contract.md",
                    "page": None,
                    "content": f"employee resignation notice period evidence-round-{window_size}",
                }],
                "pages": [],
                "metadata": {"window_size": window_size},
                "status": "ok",
            }

        inspect_tool.invoke.side_effect = inspect
        extract_tool = Mock()
        extract_tool.invoke.side_effect = lambda payload: {
            "evidence": [{
                "source": "contract.md",
                "text": payload["inspected_document"]["content"],
                "page": None,
            }],
            "status": "ok",
        }

        with TemporaryDirectory() as folder:
            Path(folder, "contract.md").write_text("# Contract\n\nNotice is 30 days.", encoding="utf-8")
            with patch("app.graph.build_llm", return_value=llm), \
                 patch("app.graph.inspect_document_window_tool", inspect_tool), \
                 patch("app.graph.extract_relevant_section_tool", extract_tool):
                result = build_graph().invoke({
                    "question": "What is the notice period for employee resignation?",
                    "documents_dir": folder,
                    "document_scope": ["contract.md"],
                }, config=GRAPH_CONFIG)
        return result, llm, inspect_calls

    def test_approved_answer_does_not_retry(self):
        result, _, inspect_calls = self.run_graph([
            "Plan the investigation.",
            "The notice period is 30 days [contract.md].",
            audit_json(True, "The answer is supported."),
        ])
        self.assertTrue(result["approved"])
        self.assertEqual(result["retry_count"], 0)
        self.assertEqual(len(inspect_calls), 1)

    def test_rejection_revises_investigation_and_produces_new_evidence(self):
        result, llm, inspect_calls = self.run_graph([
            "Plan the investigation.",
            "The notice period is 30 days [contract.md].",
            audit_json(False, "Evidence is insufficient; inspect additional context."),
            "The revised evidence confirms the notice period is 30 days [contract.md].",
            audit_json(True, "The revised answer is supported."),
        ])
        self.assertTrue(result["approved"])
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual([call["window_size"] for call in inspect_calls], [1, 2])
        solver_prompts = [
            prompt for prompt in llm.prompts
            if "EVIDENCE COVERAGE:" in prompt and "SOLVER CLAIMS ONLY:" not in prompt
        ]
        self.assertGreaterEqual(len(solver_prompts), 2)
        self.assertIn("evidence-round-1", solver_prompts[0])
        self.assertIn("evidence-round-2", solver_prompts[1])
        self.assertNotEqual(solver_prompts[0], solver_prompts[1])

    def test_repeated_rejection_stops_at_maximum(self):
        result, _, inspect_calls = self.run_graph([
            "Plan the investigation.",
            "First answer.",
            audit_json(False, "Evidence is insufficient."),
            "Second answer.",
            audit_json(False, "The answer remains unsupported."),
            "Third answer.",
            audit_json(False, "The answer is still not supported."),
        ])
        self.assertFalse(result["approved"])
        self.assertEqual(result["retry_count"], MAX_RETRIES)
        self.assertEqual(len(inspect_calls), MAX_RETRIES + 1)
        self.assertIn("AUDIT WARNING", result["final_answer"])


if __name__ == "__main__":
    unittest.main()
