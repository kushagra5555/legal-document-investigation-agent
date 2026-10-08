"""Run repeated real-graph benchmark trials and write per-case traces."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.graph import GRAPH_CONFIG, build_graph


def _fact_present(answer: str, alternatives: list[str]) -> bool:
    lowered = " ".join(str(answer).lower().split())
    return any(" ".join(value.lower().split()) in lowered for value in alternatives)


def _declined(answer: str) -> bool:
    lowered = " ".join(str(answer).lower().split())
    return any(marker in lowered for marker in (
        "insufficient evidence", "not found", "not mentioned", "does not contain",
        "cannot answer", "can't answer", "no evidence", "not available",
    ))


def _run_case(graph, case: dict, root: Path) -> dict:
    corpus = Path(case.get("corpus_override", "documents/297962019_2023-09-04.pdf"))
    corpus_dir = corpus.parent if corpus.parent != Path(".") else root
    state = {"question": case["question"], "documents_dir": str(corpus_dir.resolve())}
    if corpus.parent != Path("."):
        state["document_scope"] = [corpus.name]
    started = time.perf_counter()
    try:
        result = graph.invoke(state, config=GRAPH_CONFIG)
        error = None
    except Exception as exc:
        result = {}
        error = f"{type(exc).__name__}: {exc}"
    answer = str(result.get("final_answer", ""))
    required = case.get("required_facts", [])
    fact_results = [_fact_present(answer, alternatives) for alternatives in required]
    pages = sorted({block.get("page") for doc in result.get("inspected_documents", []) for block in doc.get("blocks", []) if block.get("page") is not None})
    gold_pages = set(case.get("gold_pages", []))
    evidence_pages = {item.get("page") for item in result.get("verified_evidence", []) if item.get("page") is not None}
    trace = result.get("investigation_trace", [])
    observed_calls = [event.get("llm_call_count") for event in trace if event.get("llm_call_count") is not None]
    observed_tokens = [event.get("approx_tokens") for event in trace if event.get("approx_tokens") is not None]
    observed_paths = sorted({path for event in trace for path in (event.get("candidate_paths") or [])})
    observed_rungs = sorted({event.get("ladder_rung") for event in trace if event.get("ladder_rung") is not None})
    declined = _declined(answer)
    expected_decline = bool(case.get("expect_decline"))
    answer_correct = (declined and not evidence_pages) if expected_decline else (bool(required) and all(fact_results))
    return {
        "id": case["id"], "question": case["question"], "answer": answer,
        "answer_correct": answer_correct,
        "required_fact_results": fact_results,
        "evidence_from_gold_page": bool(gold_pages & evidence_pages),
        "refused_as_insufficient": declined,
        "auditor_verdict": "approved" if result.get("approved") else "rejected",
        "retry_count": result.get("retry_count", 0), "llm_call_count": max(observed_calls) if observed_calls else None, "approx_tokens": max(observed_tokens) if observed_tokens else None,
        "candidate_paths": observed_paths or None, "ladder_rungs": observed_rungs or None,
        "pages_read": pages, "gold_pages": sorted(gold_pages),
        "verified_evidence_count": len(result.get("verified_evidence", [])),
        "candidate_evidence_count": len(result.get("candidate_evidence", [])), "trace_events": len(trace),
        "trace": trace, "latency_ms": round((time.perf_counter() - started) * 1000, 1), "error": error
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="tests/benchmark/fixtures.json")
    parser.add_argument("--documents-dir", default="documents")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", default="tests/benchmark/baseline_before.json")
    parser.add_argument("--trace-dir", default="tests/benchmark/traces_before")
    args = parser.parse_args()
    fixture = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    root = Path(args.documents_dir).resolve()
    graph = build_graph()
    trace_dir = Path(args.trace_dir); trace_dir.mkdir(parents=True, exist_ok=True)
    trials = []
    for case in fixture["cases"]:
        for run in range(max(1, args.runs)):
            trial = _run_case(graph, case, root); trials.append(trial)
            (trace_dir / f"{case['id']}_run{run + 1}.json").write_text(json.dumps(trial, indent=2), encoding="utf-8")
    summary = []
    for case in fixture["cases"]:
        rows = [row for row in trials if row["id"] == case["id"]]
        summary.append({"id": case["id"], "runs": len(rows), "pass_rate": round(sum(row["answer_correct"] for row in rows) / len(rows), 3), "gold_evidence_rate": round(sum(row["evidence_from_gold_page"] for row in rows) / len(rows), 3), "refused_rate": round(sum(row["refused_as_insufficient"] for row in rows) / len(rows), 3), "auditor_verdicts": sorted({row["auditor_verdict"] for row in rows}), "avg_retries": round(sum(row["retry_count"] for row in rows) / len(rows), 2), "avg_latency_ms": round(sum(row["latency_ms"] for row in rows) / len(rows), 1)})
    Path(args.output).write_text(json.dumps({"fixture": args.fixture, "runs_per_case": args.runs, "summary": summary, "trials": trials}, indent=2), encoding="utf-8")
    print(f"{'CASE':34} {'PASS':8} {'GOLD':8} {'REFUSED':8} {'AUDIT':18} {'RETRIES':8} {'MS':10}")
    print("-" * 100)
    for row in summary:
        print(f"{row['id'][:34]:34} {row['pass_rate']:<8} {row['gold_evidence_rate']:<8} {row['refused_rate']:<8} {','.join(row['auditor_verdicts']):18} {row['avg_retries']:<8} {row['avg_latency_ms']:<10}")
    print(f"\nDetailed JSON: {args.output}\nPer-run traces: {args.trace_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
