"""Adversarial tests for false-relevance prevention."""

import unittest

from app.graph import verify_evidence


SPEC = {
    "intent": "retrieve_policy_requirement",
    "subject": "employee",
    "event": "voluntary employee resignation",
    "information_needed": "required notice period",
    "constraints": ["must describe employee-initiated resignation"],
    "search_concepts": ["employee resignation", "voluntary resignation", "resignation notice"],
    "evidence_requirements": ["must state the applicable notice period"],
}


def state_for(*passages: tuple[str, str]) -> dict:
    evidence = [
        {"source_document": source, "excerpt": text, "page": index + 1}
        for index, (source, text) in enumerate(passages)
    ]
    return {"question": "What is the notice period for employee resignation?", "investigation_spec": SPEC, "candidate_evidence": evidence}


class RetrievalQualityTests(unittest.TestCase):
    def test_employer_termination_is_not_employee_resignation_evidence(self):
        result = verify_evidence(state_for(
            ("employer.md", "Employer termination requires 90 days notice."),
            ("employee.md", "Employees who voluntarily resign must provide 30 days notice."),
        ))
        self.assertEqual(len(result["verified_evidence"]), 1)
        self.assertEqual(result["verified_evidence"][0]["source_document"], "employee.md")
        rejected = result["evidence_verification"][0]
        self.assertFalse(rejected["relevant"])
        self.assertFalse(rejected["event_match"])

    def test_probation_and_contractor_passages_are_rejected(self):
        result = verify_evidence(state_for(
            ("probation.md", "Probationary employees may be terminated with 7 days notice."),
            ("contractor.md", "Contractors must provide 15 days notice."),
            ("employee.md", "Employees who voluntarily resign must provide 30 days notice."),
        ))
        self.assertEqual([item["source_document"] for item in result["verified_evidence"]], ["employee.md"])

    def test_paraphrased_resignation_evidence_is_accepted(self):
        result = verify_evidence(state_for(
            ("policy.md", "An employee wishing to leave the organization must provide one calendar month of written notice."),
        ))
        self.assertTrue(result["verified_evidence"])
        self.assertEqual(result["verified_evidence"][0]["source_document"], "policy.md")

    def test_irrelevant_corpus_produces_insufficient_evidence(self):
        result = verify_evidence(state_for(
            ("handbook.md", "Employees are expected to follow company policies."),
        ))
        self.assertTrue(result["insufficient_evidence"])
        self.assertEqual(result["verified_evidence"], [])

    def test_conflicting_verified_sources_are_reported(self):
        result = verify_evidence(state_for(
            ("policy_a.md", "Employees who voluntarily resign must provide 30 days notice."),
            ("policy_b.md", "Employees who voluntarily resign must provide 60 days notice."),
        ))
        self.assertEqual(len(result["verified_evidence"]), 2)
        self.assertEqual(len(result["contradictions"]), 1)
        self.assertEqual(set(result["contradictions"][0]["values"]), {"30 days", "60 days"})

    def test_verified_evidence_keeps_provenance(self):
        result = verify_evidence({
            "question": "What is the notice period for employee resignation?",
            "investigation_spec": SPEC,
            "candidate_evidence": [{
                "source_document": "policy.pdf",
                "excerpt": "Employees who voluntarily resign must provide 30 days notice.",
                "page": 12,
            }],
        })
        item = result["verified_evidence"][0]
        self.assertEqual(item["source_document"], "policy.pdf")
        self.assertEqual(item["page"], 12)


if __name__ == "__main__":
    unittest.main()
