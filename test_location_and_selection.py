from app.graph import _extract_location_intent, _fallback_investigation_spec, verify_evidence
from app.state import AgentState
from web_app import normalize_selected_sources


def test_first_page_is_preserved_as_location_constraint():
    intent = _extract_location_intent("What is the name of the Chief Justice mentioned on the first page?")
    assert intent["document_location"] == "first_page"
    assert intent["page_number"] == 1
    assert intent["direct_identification_required"] is True
    assert _fallback_investigation_spec("What is the name of the Chief Justice mentioned on the first page?")["target_entity"] == "Chief Justice"


def test_last_page_is_preserved_as_location_constraint():
    intent = _extract_location_intent("What date appears on the last page?")
    assert intent["document_location"] == "last_page"


def test_title_and_header_are_structural_regions():
    assert _extract_location_intent("Identify the title/header of the document")["region_type"] == "title"
    assert _extract_location_intent("What name is in the header?")["region_type"] == "header"


def test_direct_identification_prefers_requested_page_over_later_mention():
    state: AgentState = {
        "question": "What is the name of the Chief Justice mentioned on the first page?",
        "available_documents": [{"name": "case.pdf", "pages": 8}],
        "investigation_spec": _fallback_investigation_spec("What is the name of the Chief Justice mentioned on the first page?"),
        "candidate_evidence": [
            {"source_document": "case.pdf", "page": 8, "excerpt": "The Chief Justice spoke during the hearing."},
            {"source_document": "case.pdf", "page": 1, "excerpt": "HON'BLE THE CHIEF JUSTICE DY CHANDRACHUD"},
        ],
    }
    result = verify_evidence(state)
    assert result["verified_evidence"]
    assert result["verified_evidence"][0]["page"] == 1
    page_scores = {item["page"]: item["relevance_score"] for item in result["evidence_verification"]}
    assert page_scores[1] > page_scores[8]


def test_selected_source_ids_are_normalized_without_replacing_scope():
    assert normalize_selected_sources(["case.pdf", "case.pdf", "folder/other.pdf"]) == ["case.pdf", "other.pdf"]


def test_empty_selection_is_not_previous_selection():
    assert normalize_selected_sources([]) == []

