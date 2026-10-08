"""Polished local web interface for the document-investigation agent."""

from __future__ import annotations

import argparse
import cgi
from datetime import datetime
import json
import logging
import os
import re
import time
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from app.graph import DOCUMENTS_DIR, GRAPH_CONFIG, build_graph
from app.tools import SUPPORTED_TYPES, list_documents

LOGGER = logging.getLogger("document_agent.web")


def runtime_build_stamp() -> dict[str, str]:
    project = Path(__file__).resolve().parent
    graph_path = project / "app" / "graph.py"
    modified = datetime.fromtimestamp(graph_path.stat().st_mtime).isoformat(timespec="seconds") if graph_path.exists() else "missing"
    return {
        "project_folder": str(project),
        "graph_modified": modified,
        "gemini_model": os.getenv("GEMINI_MODEL", ""),
        "llm_rpm": os.getenv("LLM_MAX_REQUESTS_PER_MINUTE", "30"),
        "llm_concurrency": os.getenv("LLM_MAX_CONCURRENCY", "2"),
    }


STYLE = """
:root { --ink:#172033; --muted:#667085; --brand:#635bff; --brand2:#8b5cf6; --line:#e7eaf0; --soft:#f7f8fc; --good:#087443; --warn:#9a6700; }
* { box-sizing:border-box; }
body { margin:0; color:var(--ink); font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif; background:linear-gradient(135deg,#f8f9ff,#eef2ff); }
body::before, body::after { content:""; position:fixed; z-index:0; width:420px; height:420px; border-radius:50%; filter:blur(55px); pointer-events:none; opacity:.28; animation:float-light 12s ease-in-out infinite alternate; }
body::before { top:-160px; left:-120px; background:#9d8cff; }
body::after { right:-140px; bottom:-170px; background:#75d9ff; animation-delay:-5s; }
.shell { position:relative; z-index:1; max-width:1120px; margin:0 auto; padding:30px 20px 60px; animation:page-in .5s ease-out both; }
.topbar { display:flex; align-items:center; justify-content:space-between; gap:20px; margin-bottom:42px; }
.brand { display:flex; align-items:center; gap:12px; font-weight:750; letter-spacing:-.02em; }
.mark { width:38px; height:38px; border-radius:12px; display:grid; place-items:center; color:white; background:linear-gradient(135deg,var(--brand),var(--brand2)); box-shadow:0 8px 20px #635bff40; }
.pill { color:var(--muted); border:1px solid var(--line); background:#ffffffb8; border-radius:999px; padding:8px 13px; font-size:13px; }
.hero { max-width:760px; margin:0 auto 26px; text-align:center; }
h1 { font-size:clamp(30px,5vw,54px); line-height:1.02; letter-spacing:-.055em; margin:0 0 14px; }
.hero p { color:var(--muted); font-size:17px; line-height:1.6; margin:0; }
.card { background:#ffffffed; backdrop-filter:blur(12px); border:1px solid #ffffff; border-radius:20px; box-shadow:0 18px 55px #26336b12; padding:24px; animation:card-in .55s ease-out both; }
.form-card { max-width:800px; margin:0 auto; }
label { display:block; font-weight:700; margin-bottom:9px; }
textarea { width:100%; min-height:145px; resize:vertical; border:1px solid #d9deea; border-radius:14px; padding:15px; font:inherit; color:var(--ink); outline:none; }
textarea:focus { border-color:var(--brand); box-shadow:0 0 0 4px #635bff18; }
.actions { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-top:14px; }
.hint { color:var(--muted); font-size:13px; }
button { border:0; border-radius:12px; padding:12px 20px; color:white; background:linear-gradient(135deg,var(--brand),var(--brand2)); font-weight:700; cursor:pointer; box-shadow:0 8px 18px #635bff35; }
button:disabled { opacity:.65; cursor:wait; }
.secondary { color:var(--ink); background:#eef0f6; box-shadow:none; }
.primary-action { position:relative; overflow:hidden; min-width:190px; }
.primary-action::after { content:""; position:absolute; top:-60%; left:-30%; width:35%; height:220%; transform:rotate(25deg); background:#ffffff55; transition:left .55s ease; }
.primary-action:hover::after { left:115%; }
.file-input { position:absolute; width:1px; height:1px; opacity:0; overflow:hidden; clip:rect(0 0 0 0); }
.file-picker { display:flex; align-items:center; justify-content:space-between; gap:14px; width:100%; padding:13px 15px; border:1px dashed #bcb5ff; border-radius:14px; background:linear-gradient(100deg,#fafaff,#f4f2ff); cursor:pointer; transition:border-color .2s ease, box-shadow .2s ease, transform .2s ease; }
.file-picker:hover, .file-picker:focus-within { border-color:var(--brand); box-shadow:0 0 0 4px #635bff14, 0 10px 24px #635bff12; transform:translateY(-1px); }
.file-picker-button { color:white; background:linear-gradient(135deg,var(--brand),var(--brand2)); border-radius:9px; padding:9px 13px; font-size:13px; font-weight:750; white-space:nowrap; }
.file-picker-name { color:var(--muted); font-size:13px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.pending-row { border-color:#bcb5ff !important; background:#f5f3ff !important; }
.pending-badge { display:inline-block; margin-left:7px; color:#4338ca; background:#e7e3ff; border-radius:999px; padding:3px 7px; font-size:11px; font-weight:750; }
.file-row { display:flex; align-items:flex-start; gap:10px; }
.file-row input { margin-top:4px; accent-color:var(--brand); }
.grid { display:grid; grid-template-columns:1.25fr .75fr; gap:18px; margin-top:18px; }
.section-title { display:flex; align-items:center; justify-content:space-between; gap:10px; margin:0 0 15px; font-size:17px; }
.answer { white-space:pre-wrap; line-height:1.65; font-size:16px; }
.source-list { list-style:none; padding:0; margin:0; display:grid; gap:10px; }
.source-list li { border:1px solid var(--line); background:var(--soft); border-radius:12px; padding:12px 14px; line-height:1.5; transition:transform .18s ease, border-color .18s ease, box-shadow .18s ease; }
.source-list li:hover { transform:translateY(-2px); border-color:#c9c4ff; box-shadow:0 8px 20px #635bff12; }
.source-name { font-weight:750; color:#4338ca; }
.meta { color:var(--muted); font-size:13px; }
.audit { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.badge { border-radius:999px; padding:7px 11px; font-size:13px; font-weight:750; }
.approved { color:var(--good); background:#e8f8ef; }
.rejected { color:var(--warn); background:#fff4ce; }
.metric { display:flex; justify-content:space-between; padding:10px 0; border-bottom:1px solid var(--line); }
.metric:last-child { border-bottom:0; }
.warning { color:var(--warn); background:#fff8df; border:1px solid #f3df9b; border-radius:12px; padding:13px; }
.back { display:inline-block; margin-top:20px; color:#4338ca; font-weight:700; text-decoration:none; }
button { transition:transform .18s ease, box-shadow .18s ease, opacity .18s ease; }
button:hover:not(:disabled) { transform:translateY(-2px); box-shadow:0 12px 24px #635bff45; }
.selection-count { color:#4338ca; font-weight:750; }
.selection-tools { display:flex; gap:8px; flex-wrap:wrap; margin:0 0 12px; }
.mini-button { padding:7px 11px; border-radius:9px; font-size:12px; color:#4338ca; background:#f0efff; box-shadow:none; }
input[type=file] { width:100%; padding:10px; border:1px dashed #c9c4ff; border-radius:12px; background:#fafaff; color:var(--muted); }
@keyframes page-in { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:none; } }
@keyframes card-in { from { opacity:0; transform:translateY(14px); } to { opacity:1; transform:none; } }
@keyframes float-light { from { transform:translate3d(-20px, 10px, 0) scale(.92); } to { transform:translate3d(30px, -20px, 0) scale(1.08); } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation-duration:.01ms !important; transition-duration:.01ms !important; } }
@media (max-width:760px) { .topbar { margin-bottom:28px; } .grid { grid-template-columns:1fr; } .actions { align-items:flex-start; flex-direction:column; } button { width:100%; } }
body.night { --ink:#edf1ff; --muted:#aab4d0; --line:#36405d; --soft:#202942; --good:#7ee2a8; --warn:#ffd36b; background:linear-gradient(135deg,#0e1424,#18213a); }
body.night::before { background:#40358f; opacity:.22; }
body.night::after { background:#155d83; opacity:.2; }
body.night .pill { background:#1d2740cc; }
body.night .card { background:#151e33ed; border-color:#2c3754; box-shadow:0 18px 55px #00000035; }
body.night textarea { color:var(--ink); background:#10182a; border-color:#3b4868; }
body.night .secondary, body.night .mini-button { color:var(--ink); background:#27324c; }
body.night .file-picker, body.night input[type=file] { background:#18223a; border-color:#4b5a83; }
body.night .pending-row { background:#242d51 !important; }
body.night .pending-badge { color:#d8d2ff; background:#403b78; }
body.night .source-list li { background:var(--soft); }
body.night .source-name, body.night .back, body.night .selection-count { color:#b9b2ff; }
body.night .approved { color:#9af0bd; background:#173b2b; }
body.night .rejected, body.night .warning { color:#ffd36b; background:#40351c; border-color:#725f2b; }
.theme-toggle { color:var(--ink); background:var(--soft); border:1px solid var(--line); box-shadow:none; padding:9px 12px; font-size:13px; }
.theme-toggle:hover:not(:disabled) { box-shadow:0 8px 18px #00000020; }
.topbar-actions { display:flex; align-items:center; gap:10px; }
"""


def render(body: str, title: str = "Document Investigation Agent") -> bytes:
    return f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{escape(title)}</title><style>{STYLE}</style></head><body><main class='shell'>{body}</main></body></html>".encode("utf-8")


def render_answer_text(value: str) -> str:
    """Safely render bold text and simple lists from an answer."""

    escaped_lines = [escape(line) for line in str(value).splitlines()]
    rendered: list[str] = []
    in_list = False
    list_tag = ""

    def inline(text: str) -> str:
        # Escape first; only this small formatting subset becomes HTML.
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)

    for line in escaped_lines:
        numbered = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        bulleted = re.match(r"^\s*[-*]\s+(.+)$", line)
        if numbered or bulleted:
            tag = "ol" if numbered else "ul"
            if not in_list or list_tag != tag:
                if in_list:
                    rendered.append(f"</{list_tag}>")
                rendered.append(f"<{tag}>")
                in_list, list_tag = True, tag
            match = numbered or bulleted
            rendered.append(f"<li>{inline(match.group(1))}</li>")
            continue
        if in_list:
            rendered.append(f"</{list_tag}>")
            in_list, list_tag = False, ""
        if line.strip():
            rendered.append(f"<p>{inline(line)}</p>")
    if in_list:
        rendered.append(f"</{list_tag}>")
    return "".join(rendered) or "<p></p>"


def chrome(content: str) -> str:
    return (
        "<header class='topbar'><div class='brand'><span class='mark'>◈</span>Document Investigator</div>"
        "<div class='topbar-actions'><button type='button' class='theme-toggle' id='theme-toggle' "
        "onclick='toggleNightMode()' aria-label='Toggle night mode' aria-pressed='false'>🌙 Night mode</button>"
        "<span class='pill'>Local • Gemini powered</span></div></header>"
        "<script>"
        "(function(){if(localStorage.getItem('document-agent-night-mode')==='on'){document.body.classList.add('night');}})();"
        "function toggleNightMode(){var night=document.body.classList.toggle('night');"
        "localStorage.setItem('document-agent-night-mode',night?'on':'off');"
        "var button=document.getElementById('theme-toggle');"
        "if(button){button.setAttribute('aria-pressed',night?'true':'false');button.textContent=night?'☀️ Day mode':'🌙 Night mode';}}"
        "</script>"
        + content
    )


def normalize_selected_sources(values: object) -> list[str]:
    """Normalize only the submitted source IDs, preserving order and scope."""
    if isinstance(values, (str, bytes)):
        values = [values]
    result: list[str] = []
    for value in values or []:
        name = Path(str(value)).name
        if name and name not in result:
            result.append(name)
    return result


def form_page(message: str = "", corpus_dir: str | None = None) -> bytes:
    notice = f"<div class='warning'>{escape(message)}</div>" if message else ""
    catalog = list_documents(corpus_dir).get("documents", []) if corpus_dir else []
    if catalog:
        corpus = "".join(
            f"<li class='file-row'><input form='question-form' type='checkbox' name='selected_files' value='{escape(item['name'], quote=True)}'><span><span class='source-name'>{escape(item['name'])}</span> <span class='meta'>{escape(item.get('description', ''))}</span></span></li>"
            for item in catalog
        )
        corpus_panel = f"<section class='card' style='margin-top:18px'><h2 class='section-title'>Choose investigation sources <span class='meta'><span id='corpus-count'>{len(catalog)} files</span> · <span id='selection-count' class='selection-count'>0 selected</span></span></h2><p class='meta'>Choose the files relevant to your question. Only checked files are used for this investigation.</p><div class='selection-tools'><button type='button' class='mini-button' onclick='setAll(true)'>Select all</button><button type='button' class='mini-button' onclick='setAll(false)'>Clear all</button></div><ul id='corpus-list' class='source-list'>{corpus}</ul><div class='actions'><span class='hint'>New files appear here immediately after you choose them.</span><button id='delete-button' class='secondary' type='submit' form='question-form' formnovalidate name='action' value='delete' onclick=\"return confirm('Delete the checked documents from the local corpus?')\">Delete checked</button></div></section>"
    else:
        corpus_panel = "<section class='card warning' style='margin-top:18px'><strong>No documents loaded.</strong><br>Upload at least one supported document to begin.</section>"
    script = "<script>const count=document.getElementById('selection-count');const del=document.getElementById('delete-button');const uploads=document.getElementById('uploads');const fileName=document.getElementById('file-name');const list=document.getElementById('corpus-list');function boxes(){return [...document.querySelectorAll('input[name=selected_files]')];}function updateSelection(){const items=boxes();const n=items.filter(b=>b.checked).length;if(count)count.textContent=n+' selected';if(del){del.disabled=n===0;del.style.opacity=n===0?'.55':'1';}}function setAll(value){boxes().forEach(b=>b.checked=value);updateSelection();}function addPendingFile(file){if(!list||boxes().some(box=>box.value===file.name))return;const row=document.createElement('li');row.className='file-row pending-row';const checkbox=document.createElement('input');checkbox.form='question-form';checkbox.type='checkbox';checkbox.name='selected_files';checkbox.value=file.name;checkbox.checked=true;checkbox.addEventListener('change',updateSelection);const content=document.createElement('span');const name=document.createElement('span');name.className='source-name';name.textContent=file.name;const badge=document.createElement('span');badge.className='pending-badge';badge.textContent='New upload';const meta=document.createElement('span');meta.className='meta';meta.textContent='Will be saved when you investigate';content.append(name,badge,document.createElement('br'),meta);row.append(checkbox,content);list.appendChild(row);}if(uploads)uploads.addEventListener('change',()=>{const files=[...uploads.files];if(fileName)fileName.textContent=files.length?files.map(file=>file.name).join(', '):'No files selected — upload is optional';files.forEach(addPendingFile);updateSelection();});boxes().forEach(b=>b.addEventListener('change',updateSelection));updateSelection();</script>"
    return render(chrome(
        "<section class='hero'><h1>Ask your documents.</h1><p>Choose your files, ask a question, and receive a checked answer.</p></section>"
        f"<section class='card form-card'>{notice}<form id='question-form' method='post' action='/ask' enctype='multipart/form-data' onsubmit=\"this.querySelector('button[type=submit][value=investigate]').disabled=true;this.querySelector('button[type=submit][value=investigate]').textContent='Investigating...';\">"
        "<label for='question'>What would you like to know?</label>"
        "<textarea id='question' name='question' placeholder='Example: What are the termination conditions across these contracts?' required></textarea>"
        "<label for='uploads'>Add documents <span class='meta'>(optional)</span></label>"
        "<input id='uploads' class='file-input' name='uploads' type='file' multiple accept='.pdf,.md,.txt,.csv,.tsv,.docx,.xlsx'>"
        "<label class='file-picker' for='uploads'><span id='file-name' class='file-picker-name'>No files selected — upload is optional</span><span class='file-picker-button'>Choose files</span></label>"
        "<div class='actions'><span class='hint'>Select only the relevant files. New uploads appear below and are selected automatically.</span><button class='primary-action' type='submit' name='action' value='investigate'>Start investigation&nbsp; →</button></div></form></section>"
        f"{corpus_panel}{script}"
    ))


def result_page(result: dict) -> bytes:
    presentation_mode = os.getenv("PRESENTATION_MODE", "false").strip().lower() in {"1", "true", "yes", "on"}
    selected = "".join(f"<li><span class='source-name'>{escape(name)}</span></li>" for name in result.get("selected_documents", []))
    approved = bool(result.get("approved", False))
    audit_status = str(result.get("audit_status", ""))
    badge = "approved" if approved else "rejected"
    label = "Approved" if approved else ("Auditor unavailable" if audit_status == "unavailable" else "Needs review")
    uploaded = result.get("uploaded_files", [])
    upload_note = f"<p class='meta'>Uploaded for this investigation: {escape(', '.join(uploaded))}</p>" if uploaded else ""
    trace = result.get("investigation_trace", [])
    trace_preview = json.dumps(trace, indent=2, ensure_ascii=False)
    trace_file = result.get("trace_file")
    trace_link = f"<p class='meta'>Saved trace: <code>{escape(str(trace_file))}</code></p>" if trace_file and not presentation_mode else ""
    confidence = result.get("confidence")
    confidence_text = "Not available" if confidence is None else f"{float(confidence):.2f}"
    claim_rows = result.get("claim_verdicts") or []
    claim_items = []
    for row in claim_rows:
        quotes = row.get("cited_quotes") or []
        quote_html = "".join(
            f"<blockquote>{escape(str(item.get('quote', '')))}</blockquote>"
            for item in quotes
        ) or "<em>No cited quote was available.</em>"
        claim_items.append(
            "<li>"
            f"<strong>{escape(str(row.get('verdict', 'UNKNOWN')))}</strong> "
            f"<span>Claim: {escape(str(row.get('claim_text', '')))}</span>"
            f"<br><span>Evidence IDs: {escape(', '.join(str(value) for value in row.get('evidence_ids', [])) or 'none')}</span>"
            f"<br>{quote_html}"
            f"<small>Reason: {escape(str(row.get('auditor_reason') or row.get('reason') or 'Not recorded.'))}</small>"
            "</li>"
        )
    claim_panel = (
        "<section class='card' style='max-width:800px;margin:18px auto 0'>"
        "<h2 class='section-title'>Claim audit details</h2>"
        "<ol>" + "".join(claim_items) + "</ol></section>"
        if claim_items and not presentation_mode else ""
    )
    error = result.get("service_error") or {}
    error_panel = (
        f"<section class='card warning' style='max-width:800px; margin:18px auto 0'><strong>Service diagnostic</strong>"
        f"<p>{escape(str(error.get('status', 'error')))} · {escape(str(error.get('error_type', 'UnknownError')))}"
        f" · HTTP {escape(str(error.get('http_status', 'n/a')))}</p><p>{escape(str(error.get('message', result.get('audit_reason', ''))))}</p></section>"
        if (error or audit_status == "unavailable") and not presentation_mode else ""
    )
    answer_text = str(result.get("final_answer", "No answer produced."))
    if presentation_mode:
        # Keep the answer itself presentation-ready while retaining the full
        # diagnostic in logs and the saved JSON trace.
        answer_text = answer_text.split("\n\nAUDIT WARNING:", 1)[0]
        answer_text = answer_text.split("\n\nAUDITOR UNAVAILABLE:", 1)[0]
        if error and (not answer_text.strip() or "model service failed" in answer_text.lower()):
            answer_text = "The investigation could not be completed. Please try again."
    stamp = result.get("build_stamp") or runtime_build_stamp()
    stamp_text = " | ".join(
        f"{label}: {escape(str(value))}" for label, value in (
            ("Project", stamp.get("project_folder")),
            ("graph.py modified", stamp.get("graph_modified")),
            ("GEMINI_MODEL", stamp.get("gemini_model")),
            ("LLM RPM", stamp.get("llm_rpm")),
            ("LLM concurrency", stamp.get("llm_concurrency")),
        )
    )
    trace_details = (
        f"<details style='margin-top:16px'><summary>Show investigation trace</summary>"
        f"<pre style='white-space:pre-wrap;max-height:520px;overflow:auto;background:#f7f8fc;padding:12px;border-radius:10px;font-size:12px'>{escape(trace_preview)}</pre></details>"
        if not presentation_mode else ""
    )
    footer = (
        f"<footer class='meta' style='margin-top:24px'>Runtime build stamp: {stamp_text}</footer>"
        if not presentation_mode else ""
    )
    return render(chrome(
        "<section class='hero'><h1>Investigation complete.</h1><p>Here is the checked result from the sources you selected.</p></section>"
         f"<section class='card'><h2 class='section-title'>Final answer <span class='badge {badge}'>{label}</span></h2>{upload_note}<div class='answer'>{render_answer_text(answer_text)}</div></section>{error_panel}{claim_panel}"
        f"<aside class='card' style='max-width:800px; margin:18px auto 0'><h2 class='section-title'>Investigation details</h2><div class='metric'><span>Evidence confidence</span><strong>{confidence_text}</strong></div><div class='metric'><span>Audit status</span><strong>{escape(audit_status or ('approved' if approved else 'rejected'))}</strong></div><div class='metric'><span>Retries</span><strong>{int(result.get('retry_count', 0))}</strong></div><h2 class='section-title' style='margin-top:24px'>Sources investigated</h2><ul class='source-list'>{selected or '<li>None</li>'}</ul><p class='meta'>For your next question, choose the relevant sources on the next screen.</p>{trace_link}{trace_details}</aside>"
        f"<a class='back' href='/'>← Ask another question · choose sources</a>{footer}"
    ), "Investigation Result")


class Handler(BaseHTTPRequestHandler):
    corpus_dir = str(DOCUMENTS_DIR)
    runtime_stamp = runtime_build_stamp()

    def do_GET(self) -> None:  # noqa: N802
        if self.path not in ("/", "/ask"):
            self.send_error(404)
            return
        self.send_response(200)
        body = form_page(corpus_dir=self.corpus_dir)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/ask":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > 100 * 1024 * 1024:
            self._send(413, form_page("Upload is too large. The total request limit is 100 MB.", self.corpus_dir))
            return
        content_type = self.headers.get("Content-Type", "")
        uploaded_files: list[str] = []
        selected_uploaded_files: list[str] = []
        if content_type.startswith("multipart/form-data"):
            fields = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type, "CONTENT_LENGTH": str(length)},
                keep_blank_values=True,
            )
            question = str(fields.getfirst("question", "")).strip()
            action = str(fields.getfirst("action", "investigate"))
            selected_files = normalize_selected_sources(fields.getlist("selected_files"))
            uploads = fields["uploads"] if "uploads" in fields else []
            if not isinstance(uploads, list):
                uploads = [uploads]
            if action == "delete":
                deleted: list[str] = []
                corpus_root = Path(self.corpus_dir).resolve()
                for filename in selected_files:
                    target = (corpus_root / filename).resolve()
                    if target.parent != corpus_root or target.suffix.lower() not in SUPPORTED_TYPES:
                        continue
                    if target.is_file():
                        target.unlink()
                        deleted.append(filename)
                message = f"Deleted {len(deleted)} document(s)." if deleted else "No documents were deleted. Check the files you want to remove."
                self._send(200, form_page(message, self.corpus_dir))
                return
            for upload in uploads:
                filename = Path(upload.filename or "").name
                if not filename:
                    continue
                suffix = Path(filename).suffix.lower()
                if suffix not in SUPPORTED_TYPES:
                    self._send(400, form_page(f"Unsupported upload type: {suffix or 'unknown'}", self.corpus_dir))
                    return
                data = upload.file.read(25 * 1024 * 1024 + 1)
                if len(data) > 25 * 1024 * 1024:
                    self._send(413, form_page(f"File is too large: {filename}", self.corpus_dir))
                    return
                target = Path(self.corpus_dir) / filename
                counter = 1
                while target.exists():
                    target = Path(self.corpus_dir) / f"{Path(filename).stem}_{counter}{suffix}"
                    counter += 1
                target.write_bytes(data)
                uploaded_files.append(target.name)
                if filename in selected_files:
                    selected_uploaded_files.append(target.name)
        else:
            fields = parse_qs(self.rfile.read(length).decode("utf-8", errors="replace"))
            question = fields.get("question", [""])[0].strip()
            action = fields.get("action", ["investigate"])[0]
            selected_files = normalize_selected_sources(fields.get("selected_files", []))
            if action == "delete":
                corpus_root = Path(self.corpus_dir).resolve()
                deleted = []
                for filename in selected_files:
                    target = (corpus_root / filename).resolve()
                    if target.parent == corpus_root and target.suffix.lower() in SUPPORTED_TYPES and target.is_file():
                        target.unlink()
                        deleted.append(filename)
                self._send(200, form_page(f"Deleted {len(deleted)} document(s).", self.corpus_dir))
                return
        if not question:
            self._send(400, form_page("Please enter a question.", self.corpus_dir))
            return
        corpus_has_documents = any(
            path.is_file() and path.suffix.lower() in SUPPORTED_TYPES
            for path in Path(self.corpus_dir).iterdir()
        )
        if not corpus_has_documents:
            self._send(400, form_page("No supported documents are available. Upload at least one document before investigating.", self.corpus_dir))
            return
        available_names = {
            path.name for path in Path(self.corpus_dir).iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_TYPES
        }
        document_scope = []
        for filename in selected_files + selected_uploaded_files:
            if filename in available_names and filename not in document_scope:
                document_scope.append(filename)
        if not document_scope:
            self._send(400, form_page("Select at least one document before investigating.", self.corpus_dir))
            return
        try:
            result = build_graph().invoke(
                {"question": question, "documents_dir": self.corpus_dir, "document_scope": document_scope, "source_scope_explicit": True},
                config=GRAPH_CONFIG,
            )
            result["uploaded_files"] = uploaded_files
            result["build_stamp"] = self.runtime_stamp
            trace_dir = Path(self.corpus_dir).parent / "tmp" / "investigation_traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", question)[:60].strip("_") or "question"
            trace_path = trace_dir / f"{int(time.time())}_{safe_name}.json"
            trace_path.write_text(json.dumps(result.get("investigation_trace", []), indent=2, ensure_ascii=False), encoding="utf-8")
            result["trace_file"] = str(trace_path)
            self._send(200, result_page(result))
        except Exception as exc:
            LOGGER.exception("workflow_failed type=%s message=%s", type(exc).__name__, str(exc)[:500])
            self._send(500, form_page("The workflow could not complete. Check model configuration or provider status.", self.corpus_dir))

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local document-agent web interface.")
    parser.add_argument("--documents-dir", default=str(DOCUMENTS_DIR))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not Path(args.documents_dir).is_dir():
        raise SystemExit("Corpus folder does not exist.")
    Handler.corpus_dir = str(Path(args.documents_dir).resolve())
    Handler.runtime_stamp = runtime_build_stamp()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Web interface running at http://{args.host}:{args.port}")
    print(f"Runtime build stamp: {json.dumps(Handler.runtime_stamp, ensure_ascii=False)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
