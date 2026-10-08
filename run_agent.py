"""Command-line entry point for the configurable document-investigation agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.graph import DOCUMENTS_DIR, GRAPH_CONFIG, build_graph


def safe_print(value: object) -> None:
    """Keep output readable on legacy Windows consoles."""

    print(str(value).encode("ascii", errors="replace").decode("ascii"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask a question over a supported document corpus."
    )
    parser.add_argument("--question", help="Question to investigate.")
    parser.add_argument(
        "--documents-dir",
        default=str(DOCUMENTS_DIR),
        help="Folder containing the supported document corpus.",
    )
    parser.add_argument(
        "--show-analysis",
        action="store_true",
        help="Print the investigation plan created by the analyzer.",
    )
    parser.add_argument(
        "--show-catalog",
        action="store_true",
        help="Print discovered document metadata.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete graph result as JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    question = args.question or input("Question: ").strip()
    documents_dir = Path(args.documents_dir).resolve()
    if not question:
        safe_print("Error: a question is required.")
        return 2
    if not documents_dir.is_dir():
        safe_print(f"Error: corpus folder does not exist: {documents_dir}")
        return 2

    try:
        result = build_graph().invoke(
            {"question": question, "documents_dir": str(documents_dir)},
            config=GRAPH_CONFIG,
        )
    except Exception:
        safe_print("Agent error: the workflow could not complete. Check the model configuration and provider status.")
        return 1

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0

    if args.show_analysis:
        print("\nInvestigation analysis:")
        safe_print(result.get("question_analysis", "No analysis was produced."))
    if args.show_catalog:
        print("\nDocument catalog:")
        for document in result.get("available_documents", []):
            print(
                f"- {document['name']} | {document.get('title', '')} | "
                f"{document.get('description', '')}"
            )

    print("\nSelected documents:")
    for name in result.get("selected_documents", []):
        print(f"- {name}")
    print("\nEvidence:")
    for item in result.get("evidence", []):
        location = f", page {item['page']}" if item.get("page") else ""
        safe_print(f"- [{item['source_document']}{location}] {item['excerpt']}")
    print("\nFinal answer:")
    safe_print(result.get("final_answer", "No final answer was produced."))
    print("\nAudit:")
    print(f"- Approved: {result.get('approved', False)}")
    print(f"- Confidence: {result.get('confidence', 0.0):.2f}")
    print(f"- Retries: {result.get('retry_count', 0)}")
    print("- Sources: " + ", ".join(result.get("sources_used", [])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
