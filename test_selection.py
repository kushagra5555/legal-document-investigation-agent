"""Tests for metadata-based document-selection fallback."""

import unittest

from app.graph import _metadata_fallback_selection


class MetadataSelectionTests(unittest.TestCase):
    def test_selects_matching_metadata_when_names_are_ambiguous(self) -> None:
        catalog = [
            {
                "name": "agreement_01.pdf",
                "title": "Commercial Lease Agreement",
                "description": "Commercial property lease agreement",
                "suffix": ".pdf",
                "entities": ["ABC Ltd"],
            },
            {
                "name": "agreement_02.pdf",
                "title": "Employee Handbook",
                "description": "Workplace policies and benefits",
                "suffix": ".pdf",
                "entities": [],
            },
        ]
        selected = _metadata_fallback_selection("What are the commercial lease terms?", catalog)
        self.assertEqual(selected, ["agreement_01.pdf"])

    def test_falls_back_to_known_corpus_when_metadata_has_no_match(self) -> None:
        catalog = [
            {"name": "agreement_01.pdf", "title": "Agreement", "description": "", "suffix": ".pdf", "entities": []},
            {"name": "agreement_02.pdf", "title": "Agreement", "description": "", "suffix": ".pdf", "entities": []},
        ]
        selected = _metadata_fallback_selection("What is the insolvency rule?", catalog)
        self.assertEqual(selected, ["agreement_01.pdf", "agreement_02.pdf"])


if __name__ == "__main__":
    unittest.main()
