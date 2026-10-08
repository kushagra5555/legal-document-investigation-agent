"""Tests that discovery can use a caller-selected corpus folder."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.graph import discover_documents


class ConfigurableCorpusTests(unittest.TestCase):
    def test_discovery_uses_documents_dir_from_state(self) -> None:
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "expenses.csv").write_text("category,limit\nTravel,500\n", encoding="utf-8")
            (root / "notes.txt").write_text("Travel policy", encoding="utf-8")
            result = discover_documents({"documents_dir": str(root)})
            names = [document["name"] for document in result["available_documents"]]
            self.assertEqual(names, ["expenses.csv", "notes.txt"])

    def test_missing_corpus_is_safe(self) -> None:
        result = discover_documents({"documents_dir": "does-not-exist"})
        self.assertEqual(result["available_documents"], [])


if __name__ == "__main__":
    unittest.main()
