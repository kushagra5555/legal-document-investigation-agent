"""Smoke test for the first LangGraph workflow."""

from pathlib import Path

from app.graph import GRAPH_CONFIG, build_graph


def safe_print(value: str) -> None:
    """Print model text safely on legacy Windows consoles."""

    print(value.encode("ascii", errors="replace").decode("ascii"))


result = build_graph().invoke(
    {
        "question": "What are the termination conditions across these contracts?",
        "documents_dir": str(Path(__file__).resolve().parent / "documents"),
    },
    config=GRAPH_CONFIG,
)

print("LangGraph workflow succeeded.")
safe_print(result["question_analysis"])
print("Documents discovered:")
for document in result["available_documents"]:
    print(f"- {document['name']} ({document['suffix']}, {document['size_bytes']} bytes)")
print("Selected documents:")
for name in result["selected_documents"]:
    print(f"- {name}")
print("Evidence extracted:")
for item in result["evidence"]:
    safe_print(f"- [{item['source_document']}] {item['excerpt']}")
print("Solver answer:")
safe_print(result["solver_answer"])
print("Sources used:")
for name in result["sources_used"]:
    print(f"- {name}")
print("Audit:")
print(f"- Approved: {result['approved']}")
print(f"- Confidence: {result['confidence']:.2f}")
safe_print(f"- Reason: {result['audit_reason']}")
print(f"- Retries: {result['retry_count']}")
print("Final answer:")
safe_print(result["final_answer"])
