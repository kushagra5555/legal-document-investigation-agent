"""Large-document checks for bounded page, row, and section windows."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from app.tools import inspect_document_window


def write_large_pdf(path: Path, matching_page: int, page_count: int = 100) -> None:
    writer = PdfWriter()
    for page_number in range(1, page_count + 1):
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
        text = "TARGET termination notice" if page_number == matching_page else f"Unrelated page {page_number}"
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as output:
        writer.write(output)


class WindowingTests(unittest.TestCase):
    def test_large_markdown_returns_bounded_section_window(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "large.md"
            sections = [
                f"## Section {index}\n\n"
                + ("This section contains the TARGET termination notice."
                   if index == 500 else f"Unrelated section {index}.")
                for index in range(1, 1001)
            ]
            path.write_text("\n\n".join(sections), encoding="utf-8")
            result = inspect_document_window(str(path), "termination notice", window_size=1)
            self.assertLessEqual(result["metadata"]["returned_block_count"], 5)
            self.assertIn("TARGET", result["content"])
            self.assertNotIn("Unrelated section 1.", result["content"])

    def test_large_xlsx_returns_bounded_row_window(self) -> None:
        from openpyxl import Workbook

        with TemporaryDirectory() as folder:
            path = Path(folder) / "large.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            for row in range(1, 1001):
                sheet.cell(row, 1, "TARGET termination notice" if row == 500 else f"Row {row}")
            workbook.save(path)
            result = inspect_document_window(str(path), "termination notice", window_size=2)
            self.assertLessEqual(result["metadata"]["returned_block_count"], 5)
            self.assertIn("TARGET", result["content"])
            self.assertEqual(result["blocks"][0]["sheet"], "Sheet")

    def test_large_pdf_returns_bounded_page_window(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "large.pdf"
            write_large_pdf(path, matching_page=50)
            result = inspect_document_window(str(path), "termination notice", window_size=1)
            self.assertLessEqual(result["metadata"]["returned_block_count"], 3)
            self.assertIn("TARGET", result["content"])
            pages = {page["page"] for page in result["pages"]}
            self.assertEqual(pages, {49, 50, 51})

    def test_exact_phrase_does_not_match_common_individual_words(self) -> None:
        with TemporaryDirectory() as folder:
            path = Path(folder) / "names.md"
            path.write_text(
                "General discussion only.\n\nAn advocate spoke at the event.",
                encoding="utf-8",
            )
            result = inspect_document_window(
                str(path),
                "ADVOCATE GENERAL DC RAINA",
                exact_phrase="ADVOCATE GENERAL DC RAINA",
            )
            self.assertEqual(result["metadata"]["matched_block_count"], 0)
            self.assertEqual(result["blocks"], [])


if __name__ == "__main__":
    unittest.main()
