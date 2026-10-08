"""Tests for cached structural document maps."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.tools import build_document_map


class DocumentMapTests(unittest.TestCase):
    def test_markdown_map_keeps_sections_without_returning_full_content(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "policy.md"
            path.write_text(
                "# Leave policy\n\nEligibility requires twelve months.\n\n"
                "## Exceptions\n\nExceptions require approval.",
                encoding="utf-8",
            )
            result = build_document_map(str(path))

        self.assertEqual(result["status"], "ok")
        self.assertEqual([item["label"] for item in result["sections"]], ["Leave policy", "Exceptions"])
        self.assertLessEqual(len(result["sections"][0]["description"]), 280)

    def test_map_cache_refreshes_when_document_changes(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "notes.md"
            path.write_text("# First section\n\nFirst.", encoding="utf-8")
            first = build_document_map(str(path))
            path.write_text("# Second section\n\nSecond.", encoding="utf-8")
            second = build_document_map(str(path))

        self.assertEqual(first["sections"][0]["label"], "First section")
        self.assertEqual(second["sections"][0]["label"], "Second section")

    def test_xlsx_map_uses_sheet_and_row_ranges(self):
        from openpyxl import Workbook

        with TemporaryDirectory() as folder:
            path = Path(folder) / "employees.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Employees"
            sheet.append(["Name", "Status"])
            sheet.append(["Employee X", "Active"])
            workbook.save(path)
            workbook.close()
            result = build_document_map(str(path))

        self.assertEqual(result["sections"][0]["sheet"], "Employees")
        self.assertEqual(result["sections"][0]["row_end"], 2)


if __name__ == "__main__":
    unittest.main()
