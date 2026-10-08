"""Tests for question-aware, provenance-preserving extraction."""

import unittest

from app.tools import extract_relevant_section


class ExtractRelevantSectionTests(unittest.TestCase):
    def test_extracts_matching_markdown_passages(self) -> None:
        result = extract_relevant_section(
            "What are the termination conditions?",
            {
                "source": "contract.md",
                "status": "ok",
                "content": "# Contract\n\nTermination requires 30 days notice.\n\nThe supplier provides reports.",
                "pages": [],
            },
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(result["evidence"][0]["source"], "contract.md")

    def test_preserves_pdf_page(self) -> None:
        result = extract_relevant_section(
            "What is the notice period?",
            {
                "source": "contract.pdf",
                "status": "ok",
                "content": "",
                "pages": [{"source": "contract.pdf", "page": 7, "content": "Notice period is 30 days."}],
            },
        )
        self.assertEqual(result["evidence"][0]["page"], 7)

    def test_does_not_return_unrelated_passage(self) -> None:
        result = extract_relevant_section(
            "What are the termination conditions?",
            {
                "source": "contract.md",
                "status": "ok",
                "content": "The supplier provides reports.",
                "pages": [],
            },
        )
        self.assertEqual(result["evidence"], [])

    def test_termination_query_excludes_weak_single_match(self) -> None:
        result = extract_relevant_section(
            "What are the termination conditions across these contracts?",
            {
                "source": "contract.md",
                "status": "ok",
                "content": (
                    "# Contract 01\n\n## Termination\n\n"
                    "Either party may terminate for material breach after written notice.\n\n"
                    "## Obligations\n\nThe supplier must return information at termination."
                ),
                "pages": [],
            },
        )
        self.assertEqual(len(result["evidence"]), 1)
        self.assertIn("material breach", result["evidence"][0]["text"])

    def test_termination_query_uses_matching_section(self) -> None:
        result = extract_relevant_section(
            "What are the termination conditions?",
            {
                "source": "contract_02.md",
                "status": "ok",
                "content": (
                    "## Termination\n\nTermination for convenience requires 60 days' notice.\n\n"
                    "## Obligations\n\nThe supplier must provide transition assistance for 30 days after termination."
                ),
                "pages": [],
            },
        )
        self.assertEqual(len(result["evidence"]), 1)
        self.assertIn("convenience", result["evidence"][0]["text"])


if __name__ == "__main__":
    unittest.main()
