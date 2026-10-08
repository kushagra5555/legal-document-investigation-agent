"""Regression tests for non-legal factual document questions."""

import unittest

from app.graph import _fallback_investigation_spec, assess_evidence_sufficiency, finalize_answer, verify_evidence
from app.tools import classify_document_role, extract_relevant_section, inspect_document


CV = """John Smith
Software Engineer

Email: john@example.com
Phone: 9876543210

Education:
B.Tech in Computer Science
ABC University

Skills:
Python
React
SQL

Experience:
Software Engineer at XYZ Technologies
"""


def evidence_for(question: str):
    spec = _fallback_investigation_spec(question)
    inspected = {
        "source": "cv.md",
        "status": "ok",
        "blocks": [
            {"source": "cv.md", "line": 1, "content": CV, "section": ""},
        ],
    }
    extracted = extract_relevant_section(question, inspected, investigation_spec=spec)
    candidate = [
        {
            "source_document": item["source"],
            "excerpt": item["text"],
            "page": item.get("page"),
            "line": item.get("line"),
            "section": item.get("section"),
        }
        for item in extracted["evidence"]
    ]
    return spec, verify_evidence({
        "question": question,
        "investigation_spec": spec,
        "candidate_evidence": candidate,
    })


class GeneralDocumentQATests(unittest.TestCase):
    def assert_answerable(self, question: str, expected: str):
        spec, result = evidence_for(question)
        self.assertIsNone(spec["event"])
        self.assertTrue(result["verified_evidence"])
        self.assertFalse(result["insufficient_evidence"])
        self.assertTrue(any(expected.lower() in item["excerpt"].lower() for item in result["verified_evidence"]))

    def test_name(self):
        self.assert_answerable("What is the name of the person?", "John Smith")

    def test_email(self):
        self.assert_answerable("What is the person's email?", "john@example.com")

    def test_phone(self):
        self.assert_answerable("What is the person's phone number?", "9876543210")

    def test_degree(self):
        self.assert_answerable("What degree does the person have?", "B.Tech in Computer Science")

    def test_university(self):
        self.assert_answerable("Which university did the person attend?", "ABC University")

    def test_skills(self):
        self.assert_answerable("What skills are listed?", "Python")

    def test_employer(self):
        self.assert_answerable("Where does the person work?", "XYZ Technologies")

    def test_unknown_date_of_birth_is_insufficient(self):
        spec = _fallback_investigation_spec("What is the person's date of birth?")
        result = verify_evidence({
            "question": "What is the person's date of birth?",
            "investigation_spec": spec,
            "candidate_evidence": [{
                "source_document": "cv.md",
                "excerpt": "Software Engineer with experience in Python.",
                "page": None,
            }],
        })
        self.assertTrue(result["insufficient_evidence"])

    def test_exact_name_lookup_returns_clear_not_found_answer(self):
        question = "ADVOCATE GENERAL DC RAINA does this name was present"
        spec = _fallback_investigation_spec(question)
        result = finalize_answer({
            "investigation_spec": spec,
            "verified_evidence": [],
            "insufficient_evidence": True,
        })
        self.assertIn("not found in the searchable text", result["final_answer"])

    def test_legal_case_timeline_uses_legal_investigation_profile(self):
        spec = _fallback_investigation_spec(
            "Trace the development of the case from the initial dispute to the final conclusion. "
            "Identify the legal issues, arguments of the parties, precedents, and court reasoning."
        )
        self.assertEqual(spec["intent"], "analyze_legal_case_record")
        self.assertIn("arguments or submissions of the parties", spec["evidence_requirements"])

    def test_hearing_transcript_clearly_reports_missing_final_judgment(self):
        question = "Identify the court's central reasoning for its final conclusion."
        spec = _fallback_investigation_spec(question)
        state = {
            "question": question,
            "investigation_spec": spec,
            "inspected_documents": [{"metadata": {"record_kind": "hearing_transcript"}}],
            "verified_evidence": [{"source_document": "hearing.pdf", "excerpt": "Counsel made submissions."}],
            "evidence_coverage": {},
            "contradictions": [],
        }
        state.update(assess_evidence_sufficiency(state))
        result = finalize_answer(state)
        self.assertIn("hearing transcript, not a final judgment", result["final_answer"])

    def test_non_pdf_role_is_unknown_without_full_document_read(self):
        result = classify_document_role(__file__)
        self.assertEqual(result["record_role"], "unknown")


if __name__ == "__main__":
    unittest.main()
