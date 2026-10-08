"""Document inspection tools used after candidate selection."""

from __future__ import annotations

from pathlib import Path
import os
import re
import sqlite3
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

from langchain_core.tools import StructuredTool


SUPPORTED_TYPES = {".md", ".txt", ".csv", ".tsv", ".pdf", ".docx", ".xlsx"}
STOP_WORDS = {
    "what", "which", "when", "where", "who", "does", "the", "are", "was",
    "were", "this", "that", "these", "those", "about", "across", "from",
    "with", "into", "have", "has", "how", "can", "may", "any", "all",
}
GENERIC_DOCUMENT_WORDS = {
    "contract", "contracts", "document", "documents", "agreement", "agreements",
    "question", "questions", "condition", "conditions", "across",
}
_DOCUMENT_MAP_CACHE: dict[str, tuple[int, int, dict[str, Any]]] = {}
_FTS_CACHE: dict[str, tuple[int, int, sqlite3.Connection]] = {}


def _speaker_structure(turns: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Return whether a document has a meaningful transcript-like speaker table."""

    minimum_distinct = max(1, int(os.getenv("SPEAKER_MIN_DISTINCT", "2")))
    minimum_short_turns = max(1, int(os.getenv("SPEAKER_MIN_SHORT_TURNS", "2")))
    short_turn_words = max(1, int(os.getenv("SPEAKER_SHORT_TURN_WORDS", "20")))
    metadata_labels = {"DOI", "ISBN", "ISSN", "URL", "HTTP", "WWW", "PAGE", "DATE", "TITLE", "AUTHOR", "COPYRIGHT"}
    counts: dict[str, int] = {}
    meaningful_turns = []
    for turn in turns:
        speaker = " ".join(str(turn.get("speaker", "")).split()).strip()
        body = str(turn.get("text", "")).strip()
        if speaker and speaker.upper() not in metadata_labels and body:
            counts[speaker] = counts.get(speaker, 0) + 1
            meaningful_turns.append(body)
    names = sorted(counts)
    minimum_turns = max(minimum_distinct, minimum_distinct + 1)
    short_turns = sum(
        len(re.findall(r"\b\w+\b", body)) <= short_turn_words
        for body in meaningful_turns
    )
    # Repeated uppercase ``LABEL:`` lines also occur in report metadata and
    # standards/catalogue text.  A transcript-like table must contain a
    # small-turn signal as well as multiple repeated speakers; otherwise the
    # label is treated as ordinary document structure, not a speaker.
    meaningful = (
        len(names) >= minimum_distinct
        and len(meaningful_turns) >= minimum_turns
        and sum(count >= 2 for count in counts.values()) >= minimum_distinct
        and short_turns >= minimum_short_turns
    )
    return meaningful, names


def _body_text(value: str) -> str:
    """Remove common boilerplate lines while preserving report body text."""

    kept = []
    for line in str(value).splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if not stripped:
            continue
        if re.search(r"\b(?:doi|isbn|issn)\s*[:\-]", lowered):
            continue
        if "copyright" in lowered or lowered.startswith("www.") or lowered.startswith("http"):
            continue
        kept.append(stripped)
    return "\n".join(kept)


def _speaker_turns(text: str, page: int | None = None) -> list[dict[str, Any]]:
    """Extract generic NAME: transcript turns, including short replies."""

    # Parse line-by-line so a short reply such as ``For the Union.`` is not
    # swallowed into the previous speaker's turn.  The marker is deliberately
    # structural (uppercase speaker label followed by a colon), rather than a
    # hardcoded list of names or roles.
    marker = re.compile(r"^\s*(?:\d+\s+)?(?P<speaker>[A-Z][A-Z .'-]{1,}):\s*(?P<body>.*)$")
    turns = []
    current: dict[str, Any] | None = None
    for line in text.splitlines(keepends=True):
        match = marker.match(line.rstrip("\r\n"))
        if match:
            if current and current["text"].strip():
                current["text"] = " ".join(current["text"].split())
                turns.append(current)
            current = {
                "speaker": " ".join(match.group("speaker").split()),
                "text": match.group("body"),
                "page": page,
                "start": 0,
                "end": 0,
            }
        elif current is not None:
            current["text"] += " " + line.strip()
    if current and current["text"].strip():
        current["text"] = " ".join(current["text"].split())
        turns.append(current)
    return turns


def _entity_hints(text: str) -> list[str]:
    candidates = re.findall(r"\b[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){1,4}\b", text)
    return list(dict.fromkeys(" ".join(value.split()) for value in candidates))[:100]


def build_local_fts(path: str) -> dict[str, Any]:
    """Build a local SQLite FTS5 index over document units; never an LLM context."""

    document_path = Path(path)
    if not document_path.exists():
        return {"status": "error", "error": "Document does not exist.", "rows": 0}
    stat = document_path.stat()
    key = str(document_path.resolve())
    cached = _FTS_CACHE.get(key)
    if cached and cached[:2] == (stat.st_mtime_ns, stat.st_size):
        connection = cached[2]
    else:
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE VIRTUAL TABLE units USING fts5(source, location, text)")
        structural = build_document_map(str(document_path))
        for unit in structural.get("text_units", []):
            connection.execute("INSERT INTO units(source, location, text) VALUES (?, ?, ?)", (document_path.name, str(unit.get("location", "")), str(unit.get("text", ""))))
        connection.commit()
        _FTS_CACHE[key] = (stat.st_mtime_ns, stat.st_size, connection)
    return {"status": "ok", "rows": connection.execute("SELECT count(*) FROM units").fetchone()[0]}


def search_local_fts(path: str, query: str, limit: int = 20) -> list[dict[str, Any]]:
    build_local_fts(path)
    connection = _FTS_CACHE[str(Path(path).resolve())][2]
    terms = [term for term in re.findall(r"[A-Za-z0-9_]+", query) if len(term) > 1]
    if not terms:
        return []
    match_query = " OR ".join(f'"{term.replace(chr(34), "")}"' for term in terms[:20])
    rows = connection.execute("SELECT source, location, text, bm25(units) AS score FROM units WHERE units MATCH ? ORDER BY score LIMIT ?", (match_query, limit)).fetchall()
    return [{"source": row[0], "location": row[1], "text": row[2], "score": row[3], "path": "local_fts"} for row in rows]


def _meaningful_terms(text: str) -> set[str]:
    """Keep normal words plus meaningful short acronyms and identifiers."""
    terms: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9]*", text):
        lowered = token.lower()
        short_identifier = len(token) <= 3 and (
            token.isupper() or any(character.isdigit() for character in token)
        )
        if (
            (len(token) > 3 or short_identifier)
            and lowered not in STOP_WORDS
            and lowered not in GENERIC_DOCUMENT_WORDS
        ):
            terms.add(lowered)
    return terms


def _error_result(source: str, document_type: str, error_type: str, message: str) -> dict[str, Any]:
    return {
        "source": source,
        "document_type": document_type,
        "content": "",
        "pages": [],
        "blocks": [],
        "metadata": {},
        "status": "error",
        "error_type": error_type,
        "error": message,
    }


def list_documents(directory: str) -> dict[str, Any]:
    """Build a small metadata catalog without opening document contents."""

    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        return {"status": "error", "error": "Document directory does not exist.", "documents": []}
    documents = []
    for path in sorted(directory_path.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_TYPES:
            continue
        title = path.stem.replace("_", " ").title()
        date_match = re.search(r"\b20\d{2}-\d{2}-\d{2}\b", path.name)
        description = f"{path.suffix.lower()[1:].upper()} document: {title}."
        if path.suffix.lower() == ".md":
            try:
                first_heading = re.search(
                    r"^#\s+(.+)$", path.read_text(encoding="utf-8"), re.MULTILINE
                )
                if first_heading:
                    title = first_heading.group(1).strip()
                description = f"Document containing sections: {title}."
            except (OSError, UnicodeError):
                pass
        documents.append({
            "name": path.name,
            "path": str(path),
            "suffix": path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "title": title,
            "date": date_match.group(0) if date_match else None,
            "pages": None,
            "entities": [],
            "description": description,
        })
    return {"status": "ok", "error": None, "documents": documents}


def build_document_map(path: str) -> dict[str, Any]:
    """Build/cache a compact structural map without returning document content."""

    document_path = Path(path)
    source = document_path.name
    if not document_path.exists() or not document_path.is_file():
        return {"source": source, "status": "error", "sections": [], "page_count": None}
    stat = document_path.stat()
    cache_key = str(document_path.resolve())
    cached = _DOCUMENT_MAP_CACHE.get(cache_key)
    if cached and cached[:2] == (stat.st_mtime_ns, stat.st_size):
        return dict(cached[2])

    suffix = document_path.suffix.lower()
    sections: list[dict[str, Any]] = []
    page_count: int | None = None
    map_warnings: list[str] = []
    try:
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(document_path))
            page_count = len(reader.pages)
            text_units = []
            speaker_turns = []
            entities = []
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    raw_text = page.extract_text() or ""
                except Exception as exc:
                    map_warnings.append(f"page_text_extraction_failed:{page_number}:{type(exc).__name__}")
                    continue
                text = " ".join(raw_text.split())
                if not text:
                    continue
                text_units.append({"location": f"page:{page_number}", "page": page_number, "text": text})
                speaker_turns.extend(_speaker_turns(raw_text, page_number))
                entities.extend(_entity_hints(raw_text))
                lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
                headings = [line[:160] for line in lines if (
                    len(line) <= 160 and (
                        re.match(r"^(?:\d+(?:\.\d+)*[.)]?\s+|[IVXLC]+[.)]?\s+)", line, re.I)
                        or (line.upper() == line and len(line.split()) <= 12)
                    )
                )][:3]
                sections.append({
                    "label": headings[0] if headings else f"Page {page_number}",
                    "page_start": page_number,
                    "page_end": page_number,
                    "description": text[:280],
                    "headings": headings,
                })
            structural_text_units = text_units
        elif suffix in {".md", ".txt", ".csv", ".tsv"}:
            text = document_path.read_text(encoding="utf-8")
            structural_text_units = [{"location": f"line:{number}", "line": number, "text": line} for number, line in enumerate(text.splitlines(), start=1) if line.strip()]
            speaker_turns = _speaker_turns(text)
            entities = _entity_hints(text)
            lines = text.splitlines()
            current: dict[str, Any] | None = None
            for line_number, line in enumerate(lines, start=1):
                stripped = line.strip()
                heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
                if heading:
                    if current:
                        current["line_end"] = line_number - 1
                        sections.append(current)
                    current = {
                        "label": heading.group(2).strip()[:160],
                        "line_start": line_number,
                        "line_end": line_number,
                        "description": "",
                        "headings": [heading.group(2).strip()[:160]],
                    }
                elif stripped and current and len(current["description"]) < 280:
                    current["description"] = f"{current['description']} {stripped}".strip()[:280]
            if current:
                current["line_end"] = len(lines)
                sections.append(current)
            if not sections and text.strip():
                sections = [{"label": "Document body", "line_start": 1, "line_end": len(lines), "description": " ".join(text.split())[:280], "headings": []}]
        elif suffix == ".xlsx":
            from openpyxl import load_workbook

            workbook = load_workbook(str(document_path), read_only=True, data_only=True)
            structural_text_units = []
            speaker_turns = []
            entities = []
            for sheet in workbook.worksheets:
                headers = [str(cell.value or "") for cell in next(sheet.iter_rows(min_row=1, max_row=1), [])]
                entities.extend(value for value in headers if value)
                structural_text_units.append({"location": f"sheet:{sheet.title}", "sheet": sheet.title, "text": " ".join(headers)})
                sections.append({
                    "label": sheet.title,
                    "sheet": sheet.title,
                    "row_start": 1,
                    "row_end": sheet.max_row,
                    "description": f"Worksheet {sheet.title} with {sheet.max_row} rows and {sheet.max_column} columns.",
                    "headings": [],
                })
            workbook.close()
        elif suffix == ".docx":
            namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            with zipfile.ZipFile(document_path) as archive, archive.open("word/document.xml") as xml_file:
                paragraphs: list[str] = []
                for _, element in ET.iterparse(xml_file, events=("end",)):
                    if element.tag == f"{namespace}p":
                        text = " ".join(node.text or "" for node in element.iter(f"{namespace}t")).strip()
                        if text:
                            paragraphs.append(text)
                        element.clear()
            structural_text_units = [{"location": f"paragraph:{number}", "paragraph": number, "text": value} for number, value in enumerate(paragraphs, start=1)]
            speaker_turns = _speaker_turns("\n".join(paragraphs))
            entities = _entity_hints("\n".join(paragraphs))
            for start in range(0, len(paragraphs), 12):
                group = paragraphs[start:start + 12]
                sections.append({
                    "label": f"Paragraphs {start + 1}-{start + len(group)}",
                    "paragraph_start": start + 1,
                    "paragraph_end": start + len(group),
                    "description": " ".join(group)[:280],
                    "headings": [item[:160] for item in group if len(item) < 120][:2],
                })
        has_speaker_turns, speaker_names = _speaker_structure(speaker_turns)
        if not has_speaker_turns:
            # Candidate NAME: labels in a report are not a speaker table.
            # Do not expose them as real speakers to constraint validation.
            speaker_names = []
        result = {
            "source": source,
            "status": "ok",
            "document_type": suffix.lstrip("."),
            "page_count": page_count,
            "section_count": len(sections),
            "sections": sections[:500],
            "text_units": structural_text_units[:2000],
            "speaker_turns": speaker_turns[:5000],
            "speaker_names": speaker_names,
            "speaker_turn_count": len(speaker_turns),
            "has_speaker_turns": has_speaker_turns,
            "entity_index": list(dict.fromkeys(entities))[:500],
            "map_warnings": map_warnings[:50],
            "index_features": [feature for feature in ["page_text", "headings", "entity_hints", "speaker_turns" if has_speaker_turns else "", "spreadsheet_headers" if suffix == ".xlsx" else "", "local_fts5"] if feature],
        }
    except Exception:
        result = {"source": source, "status": "error", "document_type": suffix.lstrip("."), "page_count": page_count, "section_count": 0, "sections": []}
    _DOCUMENT_MAP_CACHE[cache_key] = (stat.st_mtime_ns, stat.st_size, result)
    return dict(result)


def classify_document_role(path: str) -> dict[str, Any]:
    """Read only document boundary pages to identify its role in a case record."""

    document_path = Path(path)
    source = document_path.name
    if document_path.suffix.lower() != ".pdf":
        return {"source": source, "record_role": "unknown", "page_count": None, "role_evidence": ""}
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(document_path))
        page_count = len(reader.pages)
        indices = sorted({0, 1, max(0, page_count - 2), max(0, page_count - 1)})
        boundary_text = "\n".join(
            reader.pages[index].extract_text() or "" for index in indices if index < page_count
        ).lower()
        if "transcript of hearing" in boundary_text or "end of day’s proceedings" in boundary_text or "end of day's proceedings" in boundary_text:
            role = "hearing_transcript"
        elif any(marker in boundary_text for marker in ("final judgment", "judgment", "judgement", "final order", "operative portion")):
            role = "judgment_or_order"
        elif any(marker in boundary_text for marker in ("written submissions", "submissions on behalf", "arguments on behalf")):
            role = "party_submissions"
        elif any(marker in boundary_text for marker in ("petition", "affidavit", "counter affidavit", "pleading")):
            role = "pleading"
        else:
            role = "unknown"
        return {"source": source, "record_role": role, "page_count": page_count, "role_evidence": boundary_text[:400]}
    except Exception:
        return {"source": source, "record_role": "unknown", "page_count": None, "role_evidence": ""}


def inspect_document(path: str) -> dict[str, Any]:
    """Open one selected Markdown/PDF document and return provenance-rich content."""

    document_path = Path(path)
    source = document_path.name
    suffix = document_path.suffix.lower()

    if not document_path.exists():
        return _error_result(source, suffix.lstrip("."), "missing_file", "Document does not exist.")
    if not document_path.is_file():
        return _error_result(source, suffix.lstrip("."), "not_a_file", "Document path is not a file.")
    if suffix not in SUPPORTED_TYPES:
        return _error_result(source, suffix.lstrip(".") or "unknown", "unsupported_type", "File type is not supported.")

    try:
        if suffix in {".md", ".txt", ".csv", ".tsv"}:
            content = document_path.read_text(encoding="utf-8")
            if not content.strip():
                return _error_result(source, suffix.lstrip("."), "empty_document", "Document contains no text.")
            if suffix == ".md":
                blocks = []
                current_heading = ""
                current_heading_level = 0
                for part in content.split("\n\n"):
                    part = part.strip()
                    if not part:
                        continue
                    heading_match = re.fullmatch(r"(#{1,6})\s+(.+)", part)
                    if heading_match:
                        current_heading_level = len(heading_match.group(1))
                        current_heading = heading_match.group(2).strip()
                        blocks.append({
                            "source": source, "content": part, "heading": True,
                            "section": current_heading, "section_level": current_heading_level,
                        })
                    else:
                        blocks.append({
                            "source": source, "content": part,
                            "section": current_heading, "section_level": current_heading_level,
                        })
            else:
                blocks = [
                    {"source": source, "line": line_number, "content": line.strip()}
                    for line_number, line in enumerate(content.splitlines(), start=1)
                    if line.strip()
                ]
            return {
                "source": source,
                "document_type": "markdown" if suffix == ".md" else suffix.lstrip("."),
                "content": content,
                "pages": [],
                "blocks": blocks,
                "metadata": {"size_bytes": document_path.stat().st_size},
                "status": "ok",
                "error_type": None,
                "error": None,
            }

        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(document_path))
            pages: list[dict[str, Any]] = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                pages.append({"source": source, "page": page_number, "content": text})
            content = "\n\n".join(page["content"] for page in pages).strip()
            if not content:
                return _error_result(source, "pdf", "empty_document", "Document contains no extractable text.")
            return {
                "source": source,
                "document_type": "pdf",
                "content": content,
                "pages": pages,
                "blocks": pages,
                "metadata": {"size_bytes": document_path.stat().st_size, "page_count": len(pages)},
                "status": "ok",
                "error_type": None,
                "error": None,
            }

        if suffix == ".docx":
            blocks = []
            word_namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            with zipfile.ZipFile(document_path) as archive:
                root = ET.fromstring(archive.read("word/document.xml"))
            for index, paragraph in enumerate(root.iter(f"{word_namespace}p"), start=1):
                text = "".join(
                    node.text or "" for node in paragraph.iter(f"{word_namespace}t")
                ).strip()
                if text:
                    blocks.append({"source": source, "paragraph": index, "content": text})
            content = "\n\n".join(block["content"] for block in blocks).strip()
            if not content:
                return _error_result(source, "docx", "empty_document", "Document contains no extractable text.")
            return {
                "source": source, "document_type": "docx", "content": content,
                "pages": [], "blocks": blocks,
                "metadata": {"size_bytes": document_path.stat().st_size},
                "status": "ok", "error_type": None, "error": None,
            }

        if suffix == ".xlsx":
            from openpyxl import load_workbook
            from openpyxl.utils import get_column_letter

            workbook = load_workbook(str(document_path), read_only=True, data_only=True)
            blocks = []
            sheet_names = list(workbook.sheetnames)
            for worksheet in workbook.worksheets:
                for row_number, row in enumerate(worksheet.iter_rows(values_only=True), start=1):
                    cells = [
                        f"{get_column_letter(column_number)}{row_number}: {value}"
                        for column_number, value in enumerate(row, start=1)
                        if value is not None and str(value).strip()
                    ]
                    if cells:
                        blocks.append({
                            "source": source, "sheet": worksheet.title,
                            "row": row_number, "content": " | ".join(cells),
                        })
            workbook.close()
            content = "\n\n".join(block["content"] for block in blocks).strip()
            if not content:
                return _error_result(source, "xlsx", "empty_document", "Workbook contains no non-empty cells.")
            return {
                "source": source, "document_type": "xlsx", "content": content,
                "pages": [], "blocks": blocks,
                "metadata": {"size_bytes": document_path.stat().st_size, "sheet_names": sheet_names},
                "status": "ok", "error_type": None, "error": None,
            }

        return _error_result(source, suffix.lstrip("."), "unsupported_type", "File type is not supported.")
    except ImportError:
        return _error_result(source, suffix.lstrip("."), "missing_dependency", f"Support for {suffix} requires an optional package.")
    except (OSError, UnicodeError) as exc:
        return _error_result(source, suffix.lstrip("."), "read_error", f"Could not read document: {type(exc).__name__}.")
    except Exception:
        return _error_result(source, suffix.lstrip("."), "extraction_error", "Document text extraction failed.")


def _window_terms(query: str) -> set[str]:
    terms = _meaningful_terms(query)
    return terms | {term.rstrip("s") for term in terms if len(term) > 4}


def _select_streamed_windows(
    blocks,
    terms: set[str],
    window_size: int,
    exact_phrase: str = "",
    include_unmatched: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Keep a bounded set of the strongest matches and nearby context.

    Repeated conversational mentions must not turn into an unbounded evidence
    set. Ranking is deterministic and local: exact phrases, headings, title-like
    language, and entity-rich uppercase text outrank ordinary dialogue.
    """

    max_candidates = 5
    all_blocks: list[dict[str, Any]] = []
    recent: list[tuple[int, dict[str, Any]]] = []
    active_windows: list[dict[str, Any]] = []

    def score_block(block: dict[str, Any], content: str) -> int:
        score = sum(2 for term in terms if term in content)
        if block.get("heading") or block.get("section_level"):
            score += 3
        if re.search(r"\b(?:hon['’]?ble|title|roster|profile|appointed|office)\b", content):
            score += 4
        if re.search(r"\b(?:chief justice|chief executive|chief operating officer)\b", content):
            score += 3
        words = re.findall(r"\b[A-Z][A-Z'’.-]{1,}\b", str(block.get("content", "")))
        if len(words) >= 2:
            score += 2
        if exact_phrase and exact_phrase in content:
            score += 100
        return score

    for index, block in enumerate(blocks):
        all_blocks.append(block)
        content = str(block.get("content", "")).lower()
        for window in active_windows:
            if index <= window["end"]:
                window["blocks"][index] = block
        header_query = bool({
            "name", "full", "candidate", "profile", "personal", "email",
            "phone", "mobile", "telephone", "contact",
        } & terms)
        matched = (
            exact_phrase in content
            if exact_phrase
            else (header_query and index == 0)
            or (bool(terms) and any(term in content for term in terms))
        )
        if matched:
            active_windows.append({
                "index": index,
                "score": score_block(block, content),
                "end": index + window_size,
                "blocks": {previous_index: previous_block for previous_index, previous_block in recent},
            })
            active_windows[-1]["blocks"][index] = block
            active_windows.sort(key=lambda window: (window["score"], -window["index"]), reverse=True)
            active_windows = active_windows[:max_candidates]
        recent.append((index, block))
        recent = recent[-window_size:] if window_size else []

    selected: dict[int, dict[str, Any]] = {}
    for window in sorted(active_windows, key=lambda item: (item["score"], -item["index"]), reverse=True):
        for index, block in window["blocks"].items():
            selected[index] = block
    # The output bound is independent of the number of repeated matches and
    # protects downstream extraction even when a context window overlaps.
    if include_unmatched:
        return all_blocks, len(active_windows)
    bounded = [
        block for _, block in sorted(
            ((index, block) for index, block in selected.items()),
            key=lambda pair: (score_block(pair[1], str(pair[1].get("content", "")).lower()), -pair[0]),
            reverse=True,
        )
    ][:max_candidates]
    return bounded, len(active_windows)


def inspect_document_window(
    path: str,
    query: str,
    window_size: int = 1,
    exact_phrase: str = "",
    include_body: bool = False,
    page_start: int | None = None,
    page_end: int | None = None,
    line_start: int | None = None,
    line_end: int | None = None,
    paragraph_start: int | None = None,
    paragraph_end: int | None = None,
    sheet: str = "",
    row_start: int | None = None,
    row_end: int | None = None,
) -> dict[str, Any]:
    """Stream one selected document and return only matching nearby blocks."""

    document_path = Path(path)
    source = document_path.name
    suffix = document_path.suffix.lower()
    window_size = max(0, int(window_size))
    if not document_path.exists():
        return _error_result(source, suffix.lstrip("."), "missing_file", "Document does not exist.")
    if not document_path.is_file():
        return _error_result(source, suffix.lstrip("."), "not_a_file", "Document path is not a file.")
    if suffix not in SUPPORTED_TYPES:
        return _error_result(source, suffix.lstrip(".") or "unknown", "unsupported_type", "File type is not supported.")

    terms = _window_terms(query)
    exact_phrase = " ".join(exact_phrase.lower().split())
    try:
        metadata: dict[str, Any] = {
            "windowed": True,
            "streamed": True,
            "source_path": str(document_path),
        }
        pages: list[dict[str, Any]] = []
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(str(document_path))
            metadata["page_count"] = len(reader.pages)
            blocks = (
                {"source": source, "path": str(document_path), "page": page_number, "content": page.extract_text() or ""}
                for page_number, page in enumerate(reader.pages, start=1)
                if (page_start is None or page_number >= page_start)
                and (page_end is None or page_number <= page_end)
            )
            selected, match_count = _select_streamed_windows(blocks, terms, window_size, exact_phrase, include_body)
            pages = selected
            document_type = "pdf"
        elif suffix == ".xlsx":
            from openpyxl import load_workbook
            from openpyxl.utils import get_column_letter

            workbook = load_workbook(str(document_path), read_only=True, data_only=True)
            metadata["sheet_names"] = list(workbook.sheetnames)

            def rows():
                for worksheet in workbook.worksheets:
                    for row_number, row in enumerate(worksheet.iter_rows(values_only=True), start=1):
                        cells = [
                            f"{get_column_letter(column_number)}{row_number}: {value}"
                            for column_number, value in enumerate(row, start=1)
                            if value is not None and str(value).strip()
                        ]
                        if cells and (not sheet or worksheet.title == sheet) and (row_start is None or row_number >= row_start) and (row_end is None or row_number <= row_end):
                            yield {
                                "source": source, "path": str(document_path), "sheet": worksheet.title,
                                "row": row_number, "content": " | ".join(cells),
                            }

            selected, match_count = _select_streamed_windows(rows(), terms, window_size, exact_phrase, include_body)
            workbook.close()
            document_type = "xlsx"
        elif suffix == ".docx":
            word_namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

            def paragraphs():
                with zipfile.ZipFile(document_path) as archive:
                    with archive.open("word/document.xml") as xml_file:
                        index = 0
                        for _, element in ET.iterparse(xml_file, events=("end",)):
                            if element.tag != f"{word_namespace}p":
                                continue
                            index += 1
                            text = "".join(
                                node.text or "" for node in element.iter(f"{word_namespace}t")
                            ).strip()
                            element.clear()
                            if text and (paragraph_start is None or index >= paragraph_start) and (paragraph_end is None or index <= paragraph_end):
                                yield {
                                    "source": source,
                                    "path": str(document_path),
                                    "paragraph": index,
                                    "content": text,
                                }

            selected, match_count = _select_streamed_windows(paragraphs(), terms, window_size, exact_phrase, include_body)
            document_type = "docx"
        else:
            document_type = "markdown" if suffix == ".md" else suffix.lstrip(".")

            def lines():
                current_heading = ""
                current_heading_level = 0
                with document_path.open("r", encoding="utf-8") as input_file:
                    for line_number, line in enumerate(input_file, start=1):
                        text = line.strip()
                        if not text:
                            continue
                        heading_match = re.fullmatch(r"(#{1,6})\s+(.+)", text)
                        if heading_match:
                            current_heading_level = len(heading_match.group(1))
                            current_heading = heading_match.group(2).strip()
                            if line_start is None or line_number >= line_start:
                                yield {
                                "source": source, "path": str(document_path), "line": line_number, "content": text,
                                "heading": True, "section": current_heading,
                                "section_level": current_heading_level,
                                }
                        else:
                            if (line_start is None or line_number >= line_start) and (line_end is None or line_number <= line_end):
                                yield {
                                "source": source, "path": str(document_path), "line": line_number, "content": text,
                                "section": current_heading, "section_level": current_heading_level,
                                }

            selected, match_count = _select_streamed_windows(lines(), terms, window_size, exact_phrase, include_body)

        if include_body:
            cleaned = []
            for block in selected:
                body = _body_text(str(block.get("content", "")))
                if body:
                    cleaned.append({**block, "content": body})
            selected = cleaned
        content = "\n\n".join(block["content"] for block in selected).strip()
        if suffix == ".pdf":
            all_turns = []
            for block in selected:
                turns = _speaker_turns(str(block.get("content", "")), block.get("page"))
                if turns:
                    block["speaker_turns"] = turns
                    all_turns.extend(turns)
            metadata["speaker_turns"] = all_turns[:5000]
            has_turns, speaker_names = _speaker_structure(all_turns)
            metadata["has_speaker_turns"] = has_turns
            metadata["speaker_names"] = speaker_names
        metadata.update({"matched_block_count": match_count, "returned_block_count": len(selected)})
        if include_body:
            body_chars = sum(len(_body_text(str(block.get("content", "")))) for block in selected)
            raw_chars = sum(len(str(block.get("content", ""))) for block in selected)
            metadata["body_text_chars"] = body_chars
            metadata["metadata_only_chars"] = max(0, raw_chars - body_chars)
            metadata["sweep_read_mostly_metadata"] = bool(raw_chars and body_chars / raw_chars < 0.20)
        return {
            "source": source, "document_type": document_type, "content": content,
            "pages": pages, "blocks": selected, "metadata": metadata,
            "status": "ok", "error_type": None, "error": None,
        }
    except ImportError:
        return _error_result(source, suffix.lstrip("."), "missing_dependency", f"Support for {suffix} requires an optional package.")
    except (OSError, UnicodeError, zipfile.BadZipFile):
        return _error_result(source, suffix.lstrip("."), "read_error", "Could not stream the selected document.")
    except Exception:
        return _error_result(source, suffix.lstrip("."), "extraction_error", "Windowed document extraction failed.")


def extract_relevant_section(
    question: str,
    inspected_document: dict[str, Any],
    investigated_query: str = "",
    investigation_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return question-matching passages from one already-inspected document."""

    question_type_hint = str((investigation_spec or {}).get("question_type", ""))

    def bounded_excerpt(text: str, matched_terms: set[str], limit: int = 900) -> str:
        """Keep a useful local context when a parser returns one huge page block."""

        text = " ".join(text.split())
        if len(text) <= limit:
            return text
        lower = text.lower()
        if question_type_hint == "list_extraction" and any(
            marker in lower for marker in ("four issues", "three issues", "five issues", "the issues")
        ):
            limit = max(limit, 1800)
            list_positions = [
                lower.find(marker) for marker in ("four issues", "three issues", "five issues", "the issues")
                if lower.find(marker) >= 0
            ]
            match_positions = list_positions or [lower.find(term) for term in matched_terms if lower.find(term) >= 0]
        else:
            match_positions = [lower.find(term) for term in matched_terms if lower.find(term) >= 0]
        anchor = min(match_positions) if match_positions else 0
        start = max(0, anchor - limit // 3)
        end = min(len(text), start + limit)
        if end - start < limit:
            start = max(0, end - limit)
        prefix = "… " if start > 0 else ""
        suffix = " …" if end < len(text) else ""
        return prefix + text[start:end].strip() + suffix

    source = str(inspected_document.get("source", "unknown"))
    spec = investigation_spec or {}
    document_metadata = inspected_document.get("metadata") or {}
    document_has_speaker_turns = bool(document_metadata.get("has_speaker_turns"))
    validated_speaker_constraints = [
        str(value).strip() for value in document_metadata.get("validated_speaker_constraints", [])
        if str(value).strip()
    ]
    if document_has_speaker_turns and not validated_speaker_constraints:
        actual_speakers = [str(value).strip() for value in document_metadata.get("speaker_names", []) if str(value).strip()]
        for requested in spec.get("speakers", []):
            requested_key = re.sub(r"[^a-z0-9]", "", str(requested).lower())
            if any(
                requested_key in re.sub(r"[^a-z0-9]", "", actual.lower())
                or re.sub(r"[^a-z0-9]", "", actual.lower()) in requested_key
                for actual in actual_speakers
            ):
                validated_speaker_constraints.append(str(requested).strip())
    if inspected_document.get("status") != "ok":
        return {
            "source": source,
            "evidence": [],
            "status": "error",
            "error": inspected_document.get("error", "Document inspection failed."),
        }

    concept_text = " ".join(
        [
            str(spec.get("subject", "")),
            str(spec.get("event", "")),
            str(spec.get("information_needed", "")),
            str(spec.get("target_attribute", "")),
            " ".join(str(value) for value in spec.get("search_concepts", [])),
            " ".join(str(value) for value in spec.get("constraints", [])),
        ]
    )
    terms = _meaningful_terms(f"{question} {investigated_query} {concept_text}")
    # Basic stemming keeps related forms together without introducing semantic retrieval.
    terms |= {term.rstrip("s") for term in terms if len(term) > 4}
    if any(term.startswith("terminat") for term in terms):
        terms |= {
            "termination", "terminate", "terminated", "breach", "cure", "cured",
            "notice", "insolvency", "convenience", "violate", "violation", "law",
        }
    if "notice" in terms or "period" in terms:
        terms |= {"notice", "period", "days", "written"}
    if any(term.startswith("oblig") for term in terms):
        terms |= {"obligation", "obligations", "must", "shall", "provide", "return", "pay"}

    blocks = inspected_document.get("blocks") or inspected_document.get("pages") or [
        {"content": inspected_document.get("content", ""), "page": None}
    ]
    attribution_hints: set[str] = set()
    if document_has_speaker_turns and validated_speaker_constraints:
        attribution_hints.update(value.lower() for value in validated_speaker_constraints)
    if document_has_speaker_turns and re.search(r"\b(?:according to|did|said|says|submission|addressed by)\b", question, re.I):
        for match in re.findall(
            r"\b(?:according to|did|said|says|submission(?: of)?|addressed by)\s+([A-Z][A-Za-z.']+(?:\s+[A-Z][A-Za-z.']+){0,3})",
            question,
        ):
            cleaned = re.sub(r"['’]s$", "", match).strip()
            if cleaned:
                attribution_hints.add(cleaned.lower())
                attribution_hints.add(cleaned.split()[-1].lower())
    for speaker in validated_speaker_constraints:
        normalized = " ".join(str(speaker).lower().replace("’", "'").split())
        if normalized:
            attribution_hints.add(normalized)
            attribution_hints.add(normalized.split()[-1])
    ledger_speaker = next(
        (str(item.get("speaker_or_source")) for item in spec.get("evidence_requirements", []) if isinstance(item, dict) and item.get("speaker_or_source")),
        "",
    )
    if ledger_speaker:
        attribution_hints.add(ledger_speaker.lower())
    termination_question = any(term.startswith("terminat") for term in terms)
    information = str(spec.get("information_needed", "")).lower()
    factual_question = not bool(spec.get("event"))
    first_content_seen = False
    ranked_evidence: list[tuple[int, dict[str, Any]]] = []
    for block in blocks:
        block_text = str(block.get("content", ""))
        block_turns = list(block.get("speaker_turns") or [])
        if attribution_hints and block_turns:
            # Keep the complete routed turn sequence. The named speaker is the
            # attribution anchor, but an adjacent judge/counsel turn can supply
            # an attribute about that person (for example, a time limit).
            paragraphs = [str(turn.get("text", "")).strip() for turn in block_turns]
            paragraph_speakers = [str(turn.get("speaker", "")) for turn in block_turns]
            paragraph_related_speakers = [
                list(dict.fromkeys([
                    str(block_turns[index - 1].get("speaker", "")) if index > 0 else "",
                    str(block_turns[index + 1].get("speaker", "")) if index + 1 < len(block_turns) else "",
                ]))
                for index in range(len(block_turns))
            ]
        else:
            paragraphs = [part.strip() for part in block_text.split("\n\n")]
            paragraph_speakers = [""] * len(paragraphs)
            paragraph_related_speakers = [[] for _ in paragraphs]
        # PDF extraction frequently returns one page with single newlines only.
        # Create bounded section-like units from clear all-caps headings so a
        # synthesis question can collect evidence from multiple parts of a page.
        if len(paragraphs) == 1 and block.get("page") is not None and "\n" in block_text:
            paragraphs = [
                part.strip() for part in re.split(
                    r"\n(?=(?:[A-Z][A-Z0-9 &/—\-]{3,}|\u2022)\n)",
                    block_text,
                ) if part.strip()
            ]
        current_heading = str(block.get("section", "")).lower()
        current_heading_level = int(block.get("section_level", 0) or 0)
        for paragraph_index, paragraph in enumerate(paragraphs):
            paragraph_speaker = paragraph_speakers[paragraph_index] if paragraph_index < len(paragraph_speakers) else ""
            # Headings identify sections but are not evidence by themselves.
            heading_match = re.fullmatch(r"(#{1,6})\s+(.+)", paragraph)
            if heading_match:
                current_heading_level = len(heading_match.group(1))
                current_heading = heading_match.group(2).lower()
                continue
            if not paragraph or block.get("heading"):
                continue
            if attribution_hints and document_has_speaker_turns and not block_turns:
                # A named-speaker requirement cannot be satisfied by a raw
                # page-level lexical match when no speaker-turn record exists.
                continue
            if attribution_hints and document_has_speaker_turns and block_turns and not paragraph_speaker:
                continue
            if attribution_hints and document_has_speaker_turns and not block_turns and not any(hint in paragraph.lower() for hint in attribution_hints):
                # Speaker/author-scoped questions must not accept a similarly
                # worded passage attributed to another person.
                continue
            section_text = current_heading.lower()
            if (
                termination_question
                and current_heading_level >= 2
                and "termination" not in current_heading
            ):
                continue
            matched_terms = {term for term in terms if term in paragraph.lower()}
            speaker_match = bool(paragraph_speaker)
            attribute_signal = False
            if factual_question:
                if "name" in information and not first_content_seen:
                    attribute_signal = True
                if "email" in information and re.search(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b", paragraph):
                    attribute_signal = True
                if "phone" in information and re.search(r"(?<!\d)(?:\+?\d[\d\s\-()]{7,}\d)(?!\d)", paragraph):
                    attribute_signal = True
                if any(term in information for term in ("university", "college")):
                    attribute_signal = any(term in paragraph.lower() for term in ("university", "college"))
                if any(term in information for term in ("degree", "qualification")):
                    attribute_signal = any(term in paragraph.lower() for term in ("degree", "b.tech", "bachelor", "master", "qualification"))
                if "skill" in information:
                    attribute_signal = "skill" in section_text or "skill" in paragraph.lower()
                if any(term in information for term in ("employer", "company", "work")):
                    attribute_signal = (
                        any(term in section_text for term in ("experience", "employment", "work"))
                        or any(term in paragraph.lower() for term in (" at ", "company", "employed", "worked"))
                    )
            # A single broad match (for example, "termination" in an obligation)
            # is weak evidence. Require two signals when the question has a legal
            # expansion, while preserving one-term questions such as "insolvency".
            minimum_score = 2 if any(term.startswith("terminat") for term in terms) else 1
            if len(matched_terms) >= minimum_score or attribute_signal or speaker_match:
                email_match = re.search(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b", paragraph)
                if factual_question and "name" in information and not first_content_seen:
                    # A CV header is the strongest name evidence. Do not anchor a
                    # long one-page PDF excerpt on a later generic word such as
                    # "name" or "candidate".
                    excerpt = " ".join(paragraph.split())[:320]
                elif factual_question and "email" in information and email_match:
                    normalized = " ".join(paragraph.split())
                    match_start = normalized.lower().find(email_match.group(0).lower())
                    excerpt = normalized[max(0, match_start - 180): match_start + 220]
                elif factual_question and "phone" in information:
                    phone_match = re.search(r"(?<!\d)(?:\+?\d[\d\s\-()]{7,}\d)(?!\d)", paragraph)
                    if phone_match:
                        normalized = " ".join(paragraph.split())
                        match_start = normalized.find(phone_match.group(0))
                        excerpt = normalized[max(0, match_start - 180): match_start + 180]
                    else:
                        excerpt = bounded_excerpt(paragraph, matched_terms)
                elif factual_question and "skill" in information and "strength" in paragraph.lower():
                    normalized = " ".join(paragraph.split())
                    match_start = normalized.lower().find("strength")
                    excerpt = normalized[max(0, match_start - 80): match_start + 820]
                else:
                    excerpt = bounded_excerpt(paragraph, matched_terms)
                item = {"source": source, "text": excerpt}
                if paragraph_speaker:
                    item["speaker"] = paragraph_speaker
                # Speaker-context arrays are derived from transcript parsing
                # and can be shorter than the paragraph list when a window
                # contains a partial/uneven turn.  Missing context is safe;
                # it must not crash the whole investigation.
                related_speakers = (
                    paragraph_related_speakers[paragraph_index]
                    if paragraph_index < len(paragraph_related_speakers)
                    else []
                )
                if related_speakers:
                    item["related_speakers"] = [value for value in related_speakers if value]
                for location_key in ("path", "page", "sheet", "row", "line", "paragraph", "section", "table"):
                    if block.get(location_key) is not None:
                        item[location_key] = block[location_key]
                ranked_evidence.append((len(matched_terms), item))
            first_content_seen = True

    ranked_evidence.sort(key=lambda pair: pair[0], reverse=True)
    return {
        "source": source,
        "evidence": [item for _, item in ranked_evidence],
        "status": "ok",
        "error": None,
    }


inspect_document_tool = StructuredTool.from_function(
    func=inspect_document,
    name="inspect_document",
    description="Open one already-selected Markdown or PDF document and return source-labelled text.",
)

list_documents_tool = StructuredTool.from_function(
    func=list_documents,
    name="list_documents",
    description="List supported files and lightweight metadata without reading their contents.",
)

build_document_map_tool = StructuredTool.from_function(
    func=build_document_map,
    name="build_document_map",
    description="Build or read a cached compact structural map for one supported document without returning its full content.",
)

classify_document_role_tool = StructuredTool.from_function(
    func=classify_document_role,
    name="classify_document_role",
    description="Identify whether a PDF is a hearing transcript, judgment/order, pleading, or submissions by reading only boundary pages.",
)

extract_relevant_section_tool = StructuredTool.from_function(
    func=extract_relevant_section,
    name="extract_relevant_section",
    description="Extract question-matching passages from one already-inspected document.",
)

inspect_document_window_tool = StructuredTool.from_function(
    func=inspect_document_window,
    name="inspect_document_window",
    description="Inspect only query-matching blocks and nearby pages, rows, or paragraphs from a selected document.",
)
