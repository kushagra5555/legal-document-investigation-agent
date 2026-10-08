import unittest

from app.graph import (
    _claim_support,
    _coerce_general_spec,
    _fallback_investigation_spec,
    _requirement_ledger,
    verify_evidence,
)


class SemanticTimeRoleTests(unittest.TestCase):
    def test_assigned_limit_beats_subject_self_estimate(self):
        question = "What time limit was subject B given?"
        spec = {
            "question_type": "attribute_lookup",
            "subject": "Subject B",
            "event": "time limit",
            "information_needed": "time limit given to Subject B",
            "evidence_requirements": ["time limit given to Subject B"],
            "speakers": ["Subject B"],
            "answer_strategy": "direct",
        }
        result = verify_evidence({
            "question": question,
            "investigation_spec": spec,
            "requirement_ledger": _requirement_ledger(question, spec),
            "candidate_evidence": [
                {
                    "source_document": "test.pdf",
                    "page": 4,
                    "excerpt": "JUSTICE A: You are given 10 minutes.",
                    "speaker": "JUSTICE A",
                    "related_speakers": ["Subject B"],
                },
                {
                    "source_document": "test.pdf",
                    "page": 4,
                    "excerpt": "SUBJECT B: I can finish in 10 or 15 minutes.",
                    "speaker": "Subject B",
                    "related_speakers": ["JUSTICE A"],
                },
            ],
        })

        verified = result["verified_evidence"]
        self.assertEqual(len(verified), 1)
        self.assertIn("given 10 minutes", verified[0]["excerpt"])
        self.assertEqual(verified[0]["temporal_role"], "assigned")
        self.assertEqual(result["requirement_ledger"][0]["status"], "SUPPORTED")

    def test_claim_splitter_does_not_create_initial_fragment(self):
        claims = _claim_support(
            "K.M. Nataraj appeared for the Union [test.pdf]. He was given 10 minutes [test.pdf].",
            [
                {"source_document": "test.pdf", "excerpt": "For the Union.", "page": 4},
                {"source_document": "test.pdf", "excerpt": "Timed you to 10 minutes.", "page": 4},
            ],
        )
        self.assertEqual(len(claims), 2)
        self.assertFalse(any(claim["claim"] == "K.M." for claim in claims))
        self.assertTrue(all(claim["evidence_ids"] for claim in claims))

    def test_planner_fallback_preserves_compound_question(self):
        question = (
            "What were the four key areas that Chief Justice DY Chandrachud mentioned, "
            "and which two areas did Mahesh Jethmalani say he was not addressing?"
        )
        spec = _coerce_general_spec(_fallback_investigation_spec(question), question)
        self.assertEqual(len(spec["evidence_requirements"]), 2)
        self.assertIn("four key areas", spec["evidence_requirements"][0])
        self.assertIn("which two areas", spec["evidence_requirements"][1])
        self.assertIn("Chief Justice DY Chandrachud", spec["speakers"])
        self.assertIn("Mahesh Jethmalani", spec["speakers"])


if __name__ == "__main__":
    unittest.main()
