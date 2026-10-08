import json
import unittest
from unittest.mock import patch

from app import graph


class _Planner:
    def __init__(self, payload):
        self.payload = payload

    def invoke(self, messages):
        return type("Response", (), {"content": json.dumps(self.payload)})()


class PlannerGeneralityTests(unittest.TestCase):
    def plan(self, question, payload):
        with patch.object(graph, "build_llm", return_value=_Planner(payload)):
            return graph.analyze_question({"question": question})

    def test_over_specific_team_lead_planner_falls_back_to_relationship_intent(self):
        result = self.plan(
            "Who led the team that prepared the report?",
            {
                "intent": "attribute lookup",
                "subject": "team",
                "event": None,
                "information_needed": "Official administrative credits identifying the team leader",
                "constraints": [],
                "search_concepts": ["official credits"],
                "evidence_requirements": ["Official credits designating the team lead"],
                "question_type": "attribute_lookup",
                "answer_strategy": "direct",
            },
        )
        spec = result["investigation_spec"]
        self.assertEqual(spec["planner_validation"]["status"], "OVER_SPECIFIC")
        self.assertTrue(spec["planner_validation"]["fallback_used"])
        self.assertIn("led", spec["information_needed"].lower())
        self.assertIn("led by", spec["planner_variants"])
        self.assertNotIn("official credits", spec["information_needed"].lower())
        self.assertIn("prepared", spec["original_question_terms"])

    def test_project_manager_is_not_given_an_invented_document_context(self):
        result = graph._coerce_general_spec(
            graph._fallback_investigation_spec("Who is the project manager?"),
            "Who is the project manager?",
        )
        spec = result
        self.assertEqual(spec["target_attribute"], "project manager")
        self.assertIn("project manager", " ".join(spec["planner_variants"]))
        self.assertFalse(spec["planner_validation"].get("fallback_used"))

    def test_budget_question_keeps_entity_and_value_type(self):
        spec = graph._coerce_general_spec(
            graph._fallback_investigation_spec("What was the budget for Project Orion?"),
            "What was the budget for Project Orion?",
        )
        self.assertEqual(spec["target_attribute"], "budget")
        self.assertEqual(spec["expected_answer_type"], "number_or_amount")
        self.assertTrue(any("budget" in value.lower() for value in spec["planner_variants"]))

    def test_author_risk_question_preserves_relationship_and_topic(self):
        spec = graph._coerce_general_spec(
            graph._fallback_investigation_spec("What did the author say about the risks?"),
            "What did the author say about the risks?",
        )
        self.assertEqual(spec["requested_relationship"], "said_or_discussed")
        self.assertIn("risks", spec["original_question_terms"])


if __name__ == "__main__":
    unittest.main()
