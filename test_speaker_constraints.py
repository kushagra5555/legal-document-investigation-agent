import tempfile
import unittest
from pathlib import Path

from app.graph import (
    _named_speakers,
    _speaker_constraint_diagnostic,
    finalize_answer,
    verify_evidence,
)
from app.tools import _speaker_structure, _speaker_turns, inspect_document_window


class SpeakerConstraintTests(unittest.TestCase):
    def test_report_title_is_not_a_speaker(self):
        question = "What are the three critical functions of standards identified in the World Development Report 2025?"
        self.assertNotIn("World Development Report", _named_speakers(question))

    def test_transcript_speaker_table_is_meaningful(self):
        text = """JUSTICE A. KAPOOR: What is your submission?\nCOUNSEL R. SHAH: For the Union.\nJUSTICE A. KAPOOR: Continue.\nCOUNSEL R. SHAH: I will."""
        turns = _speaker_turns(text)
        meaningful, names = _speaker_structure(turns)
        self.assertTrue(meaningful)
        self.assertIn("JUSTICE A. KAPOOR", names)
        self.assertIn("COUNSEL R. SHAH", names)

    def test_repeated_report_labels_are_not_a_transcript(self):
        turns = _speaker_turns(
            "ISO XXX: This is a long standards and catalogue description that continues "
            "as ordinary report body text for many words and includes definitions, "
            "examples, references, classifications, measurement details, explanatory "
            "material, implementation notes, historical context, and cross-sector "
            "comparisons rather than a conversational response from a person.\n"
            "ISO XXXX: This is another long report passage with no conversational turn "
            "and it continues with standards, quality, compatibility, implementation, "
            "measurement, reporting, classification, definitions, examples, historical "
            "context, and explanatory material for the document.\n"
            "ISO XXX: More long report body text describing a classification system and "
            "its application across jurisdictions, sectors, organizations, and markets, "
            "including definitions, examples, measurements, implementation notes, and "
            "comparative explanatory material.\n"
            "ISO XXXX: More long report body text describing a classification system and "
            "its application across jurisdictions, sectors, organizations, and markets, "
            "including definitions, examples, measurements, implementation notes, and "
            "comparative explanatory material."
        )
        meaningful, _ = _speaker_structure(turns)
        self.assertFalse(meaningful)

    def test_real_speaker_validates_and_unknown_speaker_is_dropped(self):
        state = {
            "investigation_spec": {"speakers": ["A. Kapoor"]},
            "document_maps": {
                "hearing.pdf": {
                    "has_speaker_turns": True,
                    "speaker_names": ["JUSTICE A. KAPOOR", "COUNSEL R. SHAH"],
                },
                "report.pdf": {"has_speaker_turns": False, "speaker_names": []},
            },
        }
        valid = _speaker_constraint_diagnostic(state, "hearing.pdf")
        dropped = _speaker_constraint_diagnostic(state, "report.pdf")
        self.assertTrue(valid["validated"])
        self.assertFalse(valid["dropped"])
        self.assertTrue(dropped["dropped"])
        self.assertIn("no meaningful speaker-turn table", dropped["reason"])

    def test_zero_matching_turns_are_dropped_not_fatal(self):
        state = {
            "investigation_spec": {"speakers": ["Alice Example"]},
            "document_maps": {
                "hearing.pdf": {
                    "has_speaker_turns": True,
                    "speaker_names": ["JUSTICE A. KAPOOR", "COUNSEL R. SHAH"],
                }
            },
        }
        result = _speaker_constraint_diagnostic(state, "hearing.pdf")
        self.assertTrue(result["dropped"])
        self.assertEqual(result["reason"], "speaker_constraint_dropped: no matching turns")

    def test_non_transcript_evidence_cannot_be_wrong_context_by_speaker(self):
        state = {
            "question": "What are the functions of standards?",
            "investigation_spec": {
                "subject": "World Development Report",
                "event": None,
                "information_needed": "functions of standards",
                "question_type": "list_extraction",
                "answer_strategy": "prose",
                "speakers": ["World Development Report"],
                "evidence_requirements": ["functions of standards"],
            },
            "candidate_evidence": [{
                "source_document": "report.pdf",
                "page": 35,
                "excerpt": "Measurement, Compatibility, and Quality are three functions of standards.",
                "has_speaker_turns": False,
                "validated_speaker_constraints": [],
            }],
            "requirement_ledger": [{"what": "functions of standards", "expected_form": "free_text"}],
            "selected_documents": ["report.pdf"],
            "document_investigation": {},
            "document_assessments": {},
            "investigation_history": [{"source": "report.pdf", "page": 35}],
        }
        result = verify_evidence(state)
        verdict = result["evidence_verification"][0]
        self.assertNotIn("WRONG_CONTEXT", verdict["reason"])

    def test_fallback_sweep_passes_body_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.txt"
            path.write_text("DOI: 10.1234/example\nMeasurement and quality are discussed here.", encoding="utf-8")
            result = inspect_document_window(
                str(path), "term-not-present", window_size=1, include_body=True
            )
        body = " ".join(block["content"] for block in result["blocks"])
        self.assertIn("Measurement and quality", body)
        self.assertNotIn("DOI:", body)

    def test_budget_exhaustion_is_not_insufficient_evidence_claim(self):
        result = finalize_answer({
            "question": "What is stated?",
            "investigation_spec": {},
            "exhaustive_search_attempted": True,
            "exhaustion_certificate": {
                "coverage_complete": False,
                "pages_processed": 240,
                "paths_attempted": ["structural_map", "bounded_exhaustive_batches"],
            },
            "verified_evidence": [],
            "insufficient_evidence": True,
            "retry_count": 2,
        })
        self.assertEqual(result["audit_status"], "investigation_budget_exhausted")
        self.assertIn("Investigation stopped at its limit", result["final_answer"])
        self.assertNotIn("Insufficient evidence", result["final_answer"])


if __name__ == "__main__":
    unittest.main()
