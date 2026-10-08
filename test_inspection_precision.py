"""Precision-first tests for targeted document inspection.

These tests deliberately use small synthetic records.  They describe the
expected investigation behaviour before the production ranking logic changes:
identifying evidence must beat repeated conversational mentions.
"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.graph import assess_evidence_sufficiency
from app.tools import extract_relevant_section, inspect_document_window


CHIEF_JUSTICE_HEADER = "HON'BLE THE CHIEF JUSTICE DY CHANDRACHUD"


def write_transcript(path: Path, repeated_mentions: int = 12) -> None:
    pages = [
        "\n".join(
            [
                "CHIEF JUSTICE'S COURT",
                CHIEF_JUSTICE_HEADER,
                "TRANSCRIPT OF HEARING",
            ]
        )
    ]
    pages.extend(
        f"Page {number}: Counsel addressed the Chief Justice during the hearing."
        for number in range(2, repeated_mentions + 2)
    )
    path.write_text("\n\n".join(pages), encoding="utf-8")


class InspectionPrecisionTests(unittest.TestCase):
    def test_exact_entity_attribute_lookup_finds_identifying_phrase(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "court.md"
            write_transcript(path)

            result = inspect_document_window(
                str(path),
                "Who is the Chief Justice?",
                exact_phrase=CHIEF_JUSTICE_HEADER,
            )

            self.assertEqual(result["metadata"]["matched_block_count"], 1)
            self.assertIn(CHIEF_JUSTICE_HEADER, result["content"])

    def test_document_header_title_evidence_beats_dialogue_mentions(self) -> None:
        """The identifying court header must outrank repeated hearing dialogue."""
        with TemporaryDirectory() as folder:
            path = Path(folder) / "court.md"
            write_transcript(path, repeated_mentions=30)

            result = inspect_document_window(str(path), "Who is the Chief Justice?")

            self.assertLessEqual(result["metadata"]["returned_block_count"], 5)
            self.assertIn(CHIEF_JUSTICE_HEADER, result["blocks"][0]["content"])

    def test_heading_evidence_keeps_the_adjacent_identifying_content(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "case.md"
            path.write_text(
                "# Supreme Court roster\n\n"
                "Chief Justice: DY Chandrachud\n\n"
                "## Hearing discussion\n\n"
                "The Chief Justice heard submissions from counsel.",
                encoding="utf-8",
            )

            result = inspect_document_window(str(path), "Who is the Chief Justice?", window_size=1)

            self.assertIn("# Supreme Court roster", result["content"])
            self.assertIn("Chief Justice: DY Chandrachud", result["content"])

    def test_exact_phrase_with_nearby_entity_is_retained_as_evidence(self) -> None:
        inspected = {
            "source": "court.md",
            "status": "ok",
            "blocks": [{
                "page": 1,
                "content": f"CHIEF JUSTICE'S COURT\n{CHIEF_JUSTICE_HEADER}",
            }],
        }
        result = extract_relevant_section(
            "Who is the Chief Justice?",
            inspected,
            investigated_query="Chief Justice name",
            investigation_spec={
                "question_type": "attribute_lookup",
                "subject": "Supreme Court",
                "information_needed": "name of the Chief Justice",
                "target_attribute": "Chief Justice",
                "search_concepts": ["Chief Justice", "DY Chandrachud"],
            },
        )

        self.assertEqual(len(result["evidence"]), 1)
        self.assertIn("DY CHANDRACHUD", result["evidence"][0]["text"])
        self.assertEqual(result["evidence"][0]["page"], 1)

    def test_broad_repeated_mentions_do_not_receive_equal_priority(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "court.md"
            write_transcript(path, repeated_mentions=100)

            result = inspect_document_window(str(path), "Who is the Chief Justice?")

            # A hundred dialogue mentions must not make a hundred equal candidates.
            self.assertLessEqual(result["metadata"]["returned_block_count"], 5)
            self.assertLessEqual(result["metadata"]["matched_block_count"], 5)

    def test_multi_evidence_synthesis_keeps_separate_supported_stages(self) -> None:
        inspected = {
            "source": "judgment.md",
            "status": "ok",
            "blocks": [
                {"page": 2, "content": "FACTUAL BACKGROUND: The initial dispute began after the permit was cancelled."},
                {"page": 4, "content": "LEGAL ISSUE: The Court considered whether the cancellation followed the statute."},
                {"page": 6, "content": "PARTIES' ARGUMENTS: The petitioner alleged unfairness; the respondent relied on precedent."},
                {"page": 8, "content": "COURT REASONING AND CONCLUSION: Applying the precedent, the Court dismissed the petition."},
            ],
        }
        result = extract_relevant_section(
            "Trace the factual events, legal issues, party arguments, precedents, and final reasoning.",
            inspected,
            investigation_spec={
                "subject": "case",
                "information_needed": "factual events legal issues party arguments precedents reasoning conclusion",
                "search_concepts": ["factual", "legal", "arguments", "precedent", "reasoning", "conclusion"],
            },
        )

        self.assertEqual({item["page"] for item in result["evidence"]}, {2, 4, 6, 8})

    def test_unsupported_final_conclusion_is_detected_for_hearing_only_record(self) -> None:
        result = assess_evidence_sufficiency({
            "question": "What was the court's final conclusion?",
            "investigation_spec": {"intent": "analyze_legal_case_record"},
            "inspected_documents": [{"metadata": {"record_kind": "hearing_transcript"}}],
            "verified_evidence": [{"source_document": "hearing.pdf", "page": 3, "excerpt": "Counsel made submissions."}],
            "contradictions": [],
            "exhaustion_certificate": {"complete": True},
        })

        self.assertTrue(result["insufficient_evidence"])
        self.assertEqual(result["investigation_action"], "hearing_transcript_missing_final_judgment")


if __name__ == "__main__":
    unittest.main()
