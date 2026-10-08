"""Tests for the selected-document inspection tool."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import zipfile

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.tools import inspect_document


def write_text_pdf(path: Path) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=200, height=200)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font_ref})}
    )
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 100 Td (Termination notice) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as output:
        writer.write(output)


def write_simple_docx(path: Path) -> None:
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body><w:p><w:r><w:t>Termination notice is 30 days.</w:t></w:r></w:p></w:body>'
        '</w:document>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", document_xml)


def write_simple_xlsx(path: Path) -> None:
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.active["A1"] = "Topic"
    workbook.active["B1"] = "Termination notice"
    workbook.active["A2"] = "Days"
    workbook.active["B2"] = 30
    workbook.save(path)


class InspectDocumentTests(unittest.TestCase):
    def test_valid_markdown(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "contract.md"
            path.write_text("# Contract\n\nTermination requires notice.", encoding="utf-8")
            result = inspect_document(str(path))
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["source"], "contract.md")
            self.assertIn("Termination", result["content"])
            self.assertEqual(result["pages"], [])

    def test_valid_pdf_preserves_page(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "policy.pdf"
            write_text_pdf(path)
            result = inspect_document(str(path))
            self.assertEqual(result["status"], "ok")
            self.assertIn("Termination notice", result["content"])
            self.assertEqual(result["pages"][0]["source"], "policy.pdf")
            self.assertEqual(result["pages"][0]["page"], 1)

    def test_valid_docx(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "policy.docx"
            write_simple_docx(path)
            result = inspect_document(str(path))
            self.assertEqual(result["status"], "ok")
            self.assertIn("Termination notice", result["content"])
            self.assertEqual(result["blocks"][0]["paragraph"], 1)

    def test_valid_xlsx_preserves_sheet_and_row(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "policy.xlsx"
            write_simple_xlsx(path)
            result = inspect_document(str(path))
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["blocks"][0]["sheet"], "Sheet")
            self.assertEqual(result["blocks"][0]["row"], 1)

    def test_missing_document(self) -> None:
        result = inspect_document("missing.md")
        self.assertEqual(result["error_type"], "missing_file")
        self.assertEqual(result["content"], "")

    def test_unsupported_document_type(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "data.xls"
            path.write_text("not supported", encoding="utf-8")
            result = inspect_document(str(path))
            self.assertEqual(result["error_type"], "unsupported_type")

    def test_empty_document(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "empty.md"
            path.write_text("", encoding="utf-8")
            result = inspect_document(str(path))
            self.assertEqual(result["error_type"], "empty_document")

    def test_unreadable_pdf_returns_structured_error(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "broken.pdf"
            path.write_bytes(b"not a PDF")
            result = inspect_document(str(path))
            self.assertEqual(result["status"], "error")
            self.assertIn(result["error_type"], {"extraction_error", "read_error"})


if __name__ == "__main__":
    unittest.main()
