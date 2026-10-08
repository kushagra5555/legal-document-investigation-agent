import unittest

from app.graph import _requirement_ledger, verify_evidence


class LegalReasoningVerificationTests(unittest.TestCase):
    def test_speaker_attributed_provision_reasoning_is_verified(self):
        question = (
            "What did V. Giri argue about the rights claimed in Madhav Rao Scindia, "
            "and why did he say Article 363 was not attracted?"
        )
        spec = {
            "question_type": "event_extraction",
            "subject": "rights claimed in Madhav Rao Scindia",
            "event": "legal argument and consequence",
            "information_needed": "rights claimed and why Article 363 was not attracted",
            "evidence_requirements": [
                "Direct quotes or paraphrased legal arguments from V. Giri's opinion",
                "Explicit reasoning linking the status of the rights to the jurisdictional bar",
            ],
            "speakers": ["V. Giri"],
            "answer_strategy": "explain_from_evidence",
        }
        result = verify_evidence({
            "question": question,
            "investigation_spec": spec,
            "requirement_ledger": _requirement_ledger(question, spec),
            "candidate_evidence": [
                {
                    "source_document": "case.pdf",
                    "page": 22,
                    "speaker": "V. GIRI",
                    "related_speakers": ["CHIEF JUSTICE DY CHANDRACHUD"],
                    "excerpt": (
                        "V. GIRI: The rights claimed by the petitioners originated "
                        "in Articles 291 and 362, not from the Covenant."
                    ),
                },
                {
                    "source_document": "case.pdf",
                    "page": 22,
                    "speaker": "V. GIRI",
                    "related_speakers": ["CHIEF JUSTICE DY CHANDRACHUD"],
                    "excerpt": (
                        "V. GIRI: Therefore Article 363 was not attracted because "
                        "the rights were under the Constitution itself."
                    ),
                },
            ],
        })

        self.assertEqual(len(result["verified_evidence"]), 2)
        self.assertTrue(all(item["speaker"] == "V. GIRI" for item in result["verified_evidence"]))
        self.assertTrue(all(item["status"] == "SUPPORTED" for item in result["requirement_ledger"]))
        self.assertTrue(all(item["verified_evidence_ids"] for item in result["requirement_ledger"]))


if __name__ == "__main__":
    unittest.main()
