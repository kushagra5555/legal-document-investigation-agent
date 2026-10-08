"""Domain-independent demonstration using spreadsheet and DOCX content."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from app.tools import extract_relevant_section, inspect_document_window


def write_handbook_docx(path: Path) -> None:
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body>'
        '<w:p><w:r><w:t>Equipment Handbook</w:t></w:r></w:p>'
        '<w:p><w:r><w:t>Battery replacement requires an approved service kit.</w:t></w:r></w:p>'
        '<w:p><w:r><w:t>Contact support before opening the device.</w:t></w:r></w:p>'
        '</w:body></w:document>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)


class NonLegalFormatTests(unittest.TestCase):
    def test_employee_expense_spreadsheet(self) -> None:
        from openpyxl import Workbook

        with TemporaryDirectory() as folder:
            path = Path(folder) / "expenses.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Expenses"
            sheet.append(["Category", "Limit"])
            sheet.append(["Travel", 500])
            sheet.append(["Meals", 100])
            workbook.save(path)

            inspected = inspect_document_window(str(path), "What is the travel limit?", window_size=1)
            extracted = extract_relevant_section("What is the travel limit?", inspected)
            self.assertTrue(extracted["evidence"])
            self.assertTrue(any("Travel" in item["text"] for item in extracted["evidence"]))
            self.assertTrue(all(item["source"] == "expenses.xlsx" for item in extracted["evidence"]))

    def test_product_handbook_docx(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "handbook.docx"
            write_handbook_docx(path)

            inspected = inspect_document_window(str(path), "How do I replace the battery?", window_size=1)
            extracted = extract_relevant_section("How do I replace the battery?", inspected)
            self.assertTrue(extracted["evidence"])
            self.assertIn("Battery replacement", extracted["evidence"][0]["text"])
            self.assertEqual(extracted["evidence"][0]["source"], "handbook.docx")


if __name__ == "__main__":
    unittest.main()
