# Legal Document Investigation Agent

A learning-focused LangGraph prototype for answering questions over a configurable corpus of documents.

This project demonstrates evidence-grounded document analysis: it selects relevant files, inspects bounded document windows, preserves provenance, verifies evidence, and audits the generated answer before returning it.

> **Important:** This is a research prototype, not legal advice. Do not upload confidential documents or use its output as a substitute for a qualified lawyer.

The legal contracts in `documents/` are only an example corpus. This is a general document-investigation workflow that can also answer factual questions over CVs, handbooks, reports, Markdown, TXT, CSV/TSV, PDF, DOCX, and XLSX files.

## Architecture

```text
User question
    ↓
Structured investigation specification
    ↓
Document catalog / metadata
    ↓
Concept-aware document selection
    ↓
Metadata fallback selection
    ↓
Streaming document window
    ↓
Candidate evidence extraction
    ↓
Evidence relevance verification
    ↓
Evidence requirement coverage
    ↓
Evidence sufficiency decision
    ↓
Verified evidence ranking
    ↓
Global evidence/context budget
    ↓
Solver
    ↓
Auditor feedback → bounded re-investigation (up to two retries)
    ↓
Final answer
```

This is intentionally not conventional RAG. It does not use embeddings, a vector database, similarity search, or top-K chunk retrieval. It selects candidate files from a small catalog, then inspects only relevant pages, rows, paragraphs, sections, or lines.

Verification is question-type-aware. Event matching is required for event/policy questions such as
“What is the resignation notice period?”, but it is not applicable to ordinary factual extraction
questions such as “What is the person's name?”. This prevents legal-style rules from rejecting
valid CV facts while retaining false-relevance protection for policy questions.

Not every question can be answered from one passage. For synthesis, multi-hop, and comparison
questions, the analyzer produces bounded evidence requirements. Candidate passages are verified
individually, then `evidence_coverage` records which requirements each provenance-labelled item
satisfies. Sufficiency is decided over the evidence set, and the Solver may synthesize only the
verified items that collectively satisfy the requirements.

The investigation specification also records negative concepts when they matter. For example,
an employee-resignation question can explicitly exclude employer termination, probation, and
contractor clauses. The verifier rejects those distractors before ranking or budgeting evidence.
If verified sources disagree on a critical value, the graph routes back to bounded investigation
and ultimately reports insufficient evidence rather than silently choosing one source.

## Gemini configuration

Create `.env` from `.env.example`:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.5-flash-lite
MAX_EVIDENCE_ITEMS=24
MAX_EVIDENCE_CHARS=12000
MAX_EVIDENCE_PER_SOURCE=8
MAX_CANDIDATE_EVIDENCE_ITEMS=64
```

Never commit `.env` or print the API key. The model remains configurable through `GEMINI_MODEL`.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

## Run the CLI

```powershell
.\.venv\Scripts\python.exe run_agent.py `
  --question "What are the termination conditions across these contracts?" `
  --documents-dir documents
```

The CLI prints selected documents, source-labelled evidence, the final answer, confidence, audit status, and sources.

To use another corpus, point `--documents-dir` at its folder. The same graph can operate over a spreadsheet, handbook, manual, or other supported corpus.

## Reliability behavior

- Missing, corrupt, empty, or unsupported documents return structured errors.
- Large files are streamed into bounded page/row/paragraph/section windows.
- Evidence is ranked and bounded before it reaches either the Solver or Auditor by item, total character, and per-source limits.
- Candidate evidence is verified against the investigated subject, event, requested information, and constraints before it reaches the Solver.
- Synthesis and comparison questions can use multiple verified evidence items; no single passage
  is required to contain the entire answer.
- For factual, attribute, list, and temporal questions, verification evaluates the requested
  attribute without requiring an event. Unknown attributes still produce insufficient evidence.
- Lexically similar but incorrect passages, such as employer termination rules for an employee-resignation question, are rejected.
- If no verified evidence exists, the graph retries with revised concepts when possible and otherwise returns an insufficient-evidence answer.
- Conflicting verified sources are surfaced to the Auditor instead of being silently resolved.
- Every evidence item preserves its source and available location (page, sheet/row, line,
  paragraph, section, and source path where available).
- Invalid document-selection output falls back to catalog metadata.
- Invalid auditor JSON is validated and retried once at the auditor-call level.
- Auditor rejection revises the investigation query, expands the inspection window, and routes back through selection, inspection, and evidence extraction. The graph allows at most two investigation retries.
- Provider failures use safe fallbacks where possible and never expose secrets.

The selected file may still be scanned internally by a streaming parser. That is deliberate:
the LLM receives only bounded matching windows, while the parser can locate those windows without
loading an entire large document into model context.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest -q `
  test_configurable_corpus.py `
  test_non_legal_formats.py `
  test_selection.py `
  test_windowing.py `
  test_inspect_document.py `
  test_extract_relevant_section.py `
  test_reliability.py `
  test_retrieval_quality.py `
  test_context_budget.py `
  test_feedback_retry.py `
  test_general_document_qa.py `
  test_synthesis_qa.py
```

## Interview explanation

> I used LangGraph to orchestrate a document-investigation workflow. Instead of conventional vector-based RAG or sending the entire corpus to the model, the system first builds a lightweight document catalog and selects candidate files. It then streams only relevant windows from those files, preserves evidence provenance, sends the evidence to a solver, and passes the answer through an auditor before returning it.

LangGraph controls the workflow, sufficiency routing, and conditional retry. LangChain provides the common LLM and tool interfaces. Gemini is the configured model provider. The document tools perform deterministic file inspection; the bounded relevance verifier prevents candidate passages from being treated as solver evidence merely because they share words with the question.

## Current limitations

- Scanned PDFs require OCR, which is not included yet.
- Legacy `.xls`, PowerPoint, and image files are not supported yet.
- The metadata fallback is lexical, not semantic.
- The relevance verifier is explainable rule-based matching rather than embedding retrieval, so unusual language may still require better concept expansion.
- The prototype still scans selected files internally; a lightweight non-vector index may eventually be useful for very large corpora.

## Public repository data policy

The public repository intentionally contains source code, tests, and documentation only. Local `.env` files, API keys, uploaded documents, CVs, PDFs, temporary traces, logs, virtual environments, and generated artifacts are excluded by `.gitignore`. Create your own private `documents/` folder when testing locally.

## License

MIT. See [LICENSE](LICENSE).
