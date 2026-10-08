"""Regression tests for tolerant planning and short identifier inspection."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.graph import _normalize_investigation_spec, verify_evidence
from app.tools import extract_relevant_section, inspect_document_window


def inspected(text: str) -> dict:
    return {
        "source": "controlled.md",
        "status": "ok",
        "blocks": [{"line": 1, "content": text}],
    }


class PlannerAndShortTokenTests(unittest.TestCase):
    def test_malformed_information_list_preserves_useful_plan(self):
        result = _normalize_investigation_spec({
            "intent": "attribute_lookup",
            "subject": "organization",
            "information_needed": ["name of the current Chief Executive Officer", "role holder"],
            "question_type": "attribute_lookup",
            "answer_strategy": "Locate the direct answer from the evidence",
            "target_attribute": "CEO",
            "search_concepts": ["CEO", "Chief Executive Officer"],
            "constraints": ["direct statement"],
            "evidence_requirements": ["name the role holder"],
        }, "Who is the CEO?")

        self.assertIsNotNone(result)
        self.assertEqual(result["target_attribute"], "CEO")
        self.assertIn("Chief Executive Officer", result["information_needed"])
        self.assertEqual(result["answer_strategy"], "direct")

    def test_malformed_answer_strategy_is_salvaged_without_dropping_plan(self):
        result = _normalize_investigation_spec({
            "intent": "compare_entities_from_evidence",
            "subject": "two policies",
            "information_needed": "comparison dimensions",
            "question_type": "comparison",
            "answer_strategy": ["compare the evidence and explain the difference"],
            "search_concepts": ["policy A", "policy B"],
        }, "Compare policy A and policy B.")

        self.assertIsNotNone(result)
        self.assertEqual(result["answer_strategy"], "compare_evidence")
        self.assertEqual(result["question_type"], "comparison")

    def test_one_malformed_field_does_not_discard_other_planner_fields(self):
        result = _normalize_investigation_spec({
            "intent": "attribute_lookup",
            "subject": "organization",
            "information_needed": {"primary": "current CEO", "format": "name"},
            "question_type": "attribute_lookup",
            "answer_strategy": "direct answer from verified evidence",
            "target_attribute": "CEO",
            "search_concepts": ["CEO", "leadership"],
            "constraints": "must cite the source",
        }, "Who is the CEO?")

        self.assertEqual(result["intent"], "attribute_lookup")
        self.assertEqual(result["target_attribute"], "CEO")
        self.assertIn("current CEO", result["information_needed"])
        self.assertEqual(result["constraints"], ["must cite the source"])

    def test_ceo_short_acronym_reaches_inspection_and_evidence(self):
        result = inspect_document_window("documents/controlled_ceo_bob.pdf", "Who is the CEO?")
        self.assertEqual(result["metadata"]["matched_block_count"], 1)
        evidence = extract_relevant_section(
            "Who is the CEO?", result,
            investigation_spec={
                "subject": "organization",
                "information_needed": "name of the CEO",
                "target_attribute": "CEO",
                "search_concepts": ["CEO", "Chief Executive Officer"],
            },
        )
        self.assertEqual(len(evidence["evidence"]), 1)
        self.assertIn("Bob", evidence["evidence"][0]["text"])

    def test_direct_attribute_statement_is_verified_without_repeated_subject_name(self):
        result = verify_evidence({
            "question": "Who is the CEO?",
            "investigation_spec": {
                "question_type": "attribute_lookup",
                "subject": "corporation or organization",
                "information_needed": "name of the Chief Executive Officer (CEO)",
                "target_attribute": "CEO",
            },
            "candidate_evidence": [{
                "source_document": "controlled_ceo_bob.pdf",
                "page": 1,
                "excerpt": "The CEO is Bob.",
            }],
        })
        self.assertEqual(len(result["verified_evidence"]), 1)

    def test_cfo_and_cto_short_role_terms_are_supported(self):
        with TemporaryDirectory() as folder:
            for role, name in (("CFO", "Carol"), ("CTO", "Taylor")):
                path = Path(folder) / f"{role}.md"
                path.write_text(f"The {role} is {name}.", encoding="utf-8")
                result = inspect_document_window(str(path), f"Who is the {role}?")
                self.assertEqual(result["metadata"]["matched_block_count"], 1, role)

    def test_ai_short_acronym_is_supported(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "ai.md"
            path.write_text("AI means artificial intelligence.", encoding="utf-8")
            result = inspect_document_window(
                str(path), "What does AI mean in this document?"
            )
            self.assertEqual(result["metadata"]["matched_block_count"], 1)

    def test_normal_factual_question_still_matches(self):
        result = inspect_document_window(
            "documents/controlled_company_2012.pdf",
            "When was the company founded?",
        )
        self.assertIn("2012", result["content"])

    def test_synthesis_evidence_is_not_reduced_to_one_short_token(self):
        result = extract_relevant_section(
            "Trace the facts, legal issues, arguments, precedents, reasoning, and conclusion.",
            {
                "source": "judgment.md",
                "status": "ok",
                "blocks": [
                    {"page": 2, "content": "Facts: the dispute began after the permit was cancelled."},
                    {"page": 4, "content": "Legal issues: the Court considered the statute."},
                    {"page": 6, "content": "Arguments: the petitioner relied on precedent."},
                    {"page": 8, "content": "Reasoning and conclusion: the Court dismissed the petition."},
                ],
            },
            investigation_spec={
                "subject": "case",
                "information_needed": "facts legal issues arguments precedents reasoning conclusion",
                "search_concepts": ["facts", "legal issues", "arguments", "precedents", "reasoning", "conclusion"],
            },
        )
        self.assertEqual({item["page"] for item in result["evidence"]}, {2, 4, 6, 8})


if __name__ == "__main__":
    unittest.main()
