"""The smallest useful LangGraph workflow for this project."""

import json
import os
import re
from difflib import SequenceMatcher
from pathlib import Path

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage

from app.llm import LLMCallFailure, build_llm, get_observability, invoke_with_resilience, reset_observability
from app.state import AgentState, DocumentMetadata, EvidenceItem
from app.tools import (
    classify_document_role_tool,
    build_document_map_tool,
    extract_relevant_section_tool,
    inspect_document_window_tool,
    list_documents_tool,
)

DOCUMENTS_DIR = Path(__file__).resolve().parent.parent / "documents"
load_dotenv()
MAX_RETRIES = 2
GRAPH_CONFIG = {"recursion_limit": 50}
MAX_EVIDENCE_ITEMS = int(os.getenv("MAX_EVIDENCE_ITEMS", "24"))
MAX_EVIDENCE_CHARS = int(os.getenv("MAX_EVIDENCE_CHARS", "60000"))
MAX_EVIDENCE_PER_SOURCE = int(os.getenv("MAX_EVIDENCE_PER_SOURCE", "8"))
MAX_CANDIDATE_EVIDENCE_ITEMS = int(os.getenv("MAX_CANDIDATE_EVIDENCE_ITEMS", "64"))


def _trace(state: AgentState, stage: str, **details: object) -> list[dict[str, object]]:
    """Append a bounded, provenance-safe event to the investigation trace."""

    events = list(state.get("investigation_trace") or [])
    metrics = get_observability()
    events.append({
        "stage": stage,
        "candidate_paths": details.pop("candidate_paths", None),
        "ladder_rung": details.pop("ladder_rung", None),
        "llm_call_count": metrics.get("llm_calls"),
        "approx_tokens": metrics.get("approx_tokens"),
        **details,
    })
    return events[-80:]
ALLOWED_INVESTIGATION_ACTIONS = {
    "broaden_search",
    "search_related_concepts",
    "inspect_additional_documents",
    "inspect_broader_window",
    "verify_specific_claim",
    "resolve_contradiction",
    "insufficient_evidence",
}
QUESTION_TYPES = {
    "factual_extraction", "attribute_lookup", "list_extraction", "temporal_lookup",
    "event_extraction", "policy_lookup", "multi_hop", "synthesis", "comparison",
    "document_discovery", "unknown",
}
ANSWER_STRATEGIES = {"direct", "aggregate_evidence", "compare_evidence", "explain_from_evidence", "insufficient_if_missing"}


def _planner_text(value: object) -> str:
    """Convert common LLM structured values into bounded planner text."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple, set)):
        return "; ".join(part for part in (_planner_text(item) for item in value) if part)
    if isinstance(value, dict):
        return "; ".join(part for part in (_planner_text(item) for item in value.values()) if part)
    if value is None:
        return ""
    return str(value).strip()


def _planner_list(value: object, limit: int = 8) -> list[str]:
    values = value if isinstance(value, (list, tuple, set)) else [value]
    result: list[str] = []
    for item in values:
        text = _planner_text(item)
        if text and text not in result:
            result.append(text)
    return result[:limit]


def _question_terms(question: str) -> list[str]:
    """Keep meaningful user wording available even when planning drifts."""
    stop = {
        "what", "which", "when", "where", "who", "why", "how", "does", "did",
        "the", "a", "an", "is", "was", "were", "are", "for", "that", "this",
        "and", "or", "of", "to", "in", "on", "from", "about", "according",
    }
    return list(dict.fromkeys(
        token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]*", question)
        if token.lower() not in stop
    ))[:24]


def _generic_planner_variants(question: str) -> dict[str, object]:
    """Derive domain-neutral relationship signals from the user's wording."""
    text = " ".join(str(question).split()).strip(" ?.")
    lowered = text.lower()
    variants: list[str] = [text]
    result: dict[str, object] = {
        "target_entity": None,
        "target_attribute": None,
        "requested_relationship": None,
        "requested_action": None,
        "expected_answer_type": "free_text",
        "semantic_intent": text,
        "lexical_variants": variants,
        "structural_hints": [],
        "original_question_terms": _question_terms(text),
    }
    match = re.search(r"\bwho\s+led\s+(?:the\s+)?(.+?)(?:\s+(?:that|which)\s+prepared\s+(.+))?$", lowered)
    if match:
        entity = match.group(1).strip()
        context = (match.group(2) or "").strip()
        result.update({
            "target_entity": entity,
            "target_attribute": "leader",
            "requested_relationship": "led_by",
            "requested_action": "prepared" if context else None,
            "expected_answer_type": "person",
            "semantic_intent": f"identify the person who led the {entity}" + (f" that prepared {context}" if context else ""),
            "structural_hints": ["person <- led <- entity", "leader/head of entity"],
        })
        variants.extend(["led by", "team lead", "team leader", "headed by", "head of team", "leadership"])
        if context:
            variants.extend([f"prepared by {context}", f"team responsible for preparing {context}"])
    elif re.search(r"\bproject manager\b", lowered):
        result.update({
            "target_attribute": "project manager",
            "expected_answer_type": "person",
            "semantic_intent": "identify the person serving as project manager",
            "structural_hints": ["person <- manages <- project"],
        })
        variants.extend(["project manager", "managed by", "manager", "led by"])
    elif re.search(r"\bbudget\b", lowered):
        result.update({
            "target_attribute": "budget",
            "expected_answer_type": "number_or_amount",
            "semantic_intent": text,
            "structural_hints": ["entity <- has <- budget/value"],
        })
        variants.extend(["budget", "budgeted", "cost", "funding", "amount"])
    elif re.search(r"\bwhat did (?:the )?author\b", lowered):
        result.update({
            "target_entity": "author",
            "requested_relationship": "said_or_discussed",
            "expected_answer_type": "free_text",
            "structural_hints": ["author <- said/discussed <- topic"],
        })
        variants.extend(["author said", "author discussed", "according to the author"])
    result["lexical_variants"] = list(dict.fromkeys(variants))[:16]
    return result


def _validate_planner_output(question: str, spec: dict[str, object]) -> dict[str, object]:
    """Reject planner descriptions that invent a narrower source scenario."""
    source = " ".join([
        str(spec.get("information_needed", "")),
        " ".join(str(value) for value in spec.get("evidence_requirements", [])),
    ])
    question_terms = set(_question_terms(question))
    output_terms = set(_question_terms(source))
    invented = sorted(output_terms - question_terms)
    # Deterministic fallback prose legitimately adds words such as “identify
    # the person”; only flag a large expansion, or explicit source-scenario
    # vocabulary, as planner drift.
    too_long = len(source.split()) > max(30, len(question.split()) * 3)
    suspicious_terms = {"official", "credits", "front", "matter", "announcement", "filing", "administrative"}
    suspicious = len(set(invented) & suspicious_terms) >= 2 or (too_long and len(invented) >= 8)
    return {
        "status": "OVER_SPECIFIC" if suspicious else "VALIDATED",
        "invented_terms": invented[:16],
        "original_term_coverage": sorted(question_terms & output_terms),
        "fallback_used": suspicious,
        "reason": "Planner introduced unsupported source/location assumptions." if suspicious else "Planner retained the requested semantic components.",
    }


def _normalize_answer_strategy(value: object) -> str:
    """Map mildly malformed strategy prose to the supported bounded set."""
    text = _planner_text(value).lower()
    if text in ANSWER_STRATEGIES:
        return text
    if any(term in text for term in ("compare", "comparison")):
        return "compare_evidence"
    if any(term in text for term in ("aggregate", "synthesi", "multiple", "all evidence")):
        return "aggregate_evidence"
    if any(term in text for term in ("explain", "reason", "trace", "chronolog")):
        return "explain_from_evidence"
    if any(term in text for term in ("insufficient", "missing", "cannot answer")):
        return "insufficient_if_missing"
    return "direct"


def _extract_location_intent(question: str) -> dict[str, object]:
    """Extract explicit document-region constraints without domain vocabulary."""
    text = " ".join(str(question).split())
    lowered = text.lower()
    result: dict[str, object] = {"document_location": None, "region_type": None, "page_number": None, "sheet": None, "column": None, "direct_identification_required": False}
    page_match = re.search(r"\bpage\s+(\d+)\b", lowered)
    if page_match:
        result["document_location"] = f"page_{page_match.group(1)}"
        result["page_number"] = int(page_match.group(1))
    elif re.search(r"\b(?:first|opening)\s+page\b", lowered):
        result["document_location"], result["page_number"] = "first_page", 1
    elif re.search(r"\b(?:last|final)\s+page\b", lowered):
        result["document_location"] = "last_page"
    for phrase, location, region in (("opening section", "opening_section", "section"), ("table of contents", "table_of_contents", "table_of_contents"), ("executive summary", "executive_summary", "section"), ("conclusion", "conclusion", "section"), ("title", "title", "title"), ("header", "header", "header"), ("heading", "heading", "heading")):
        if phrase in lowered:
            result["document_location"] = result["document_location"] or location
            result["region_type"] = region
            break
    if "title/header" in lowered or "title or header" in lowered:
        result["region_type"] = "header"
    sheet_match = re.search(r"\bsheet\s+['\"]?([^,'\"?]+)", text, re.I)
    column_match = re.search(r"\bcolumn\s+([A-Z]{1,3}|\d+)\b", text, re.I)
    if sheet_match:
        result["sheet"] = sheet_match.group(1).strip()
        result["document_location"] = result["document_location"] or "sheet"
    if column_match:
        result["column"] = column_match.group(1).upper()
        result["document_location"] = result["document_location"] or "column"
    result["direct_identification_required"] = bool(re.search(r"\b(?:what is the name|who is|name of|identify|mentioned)\b", lowered)) and bool(result["document_location"] or result["region_type"])
    entity_match = re.search(r"\b(?:name of|who is)\s+(?:the\s+)?([A-Za-z][A-Za-z .'-]+?)(?:\s+mentioned|\s+on\s+the|\s+in\s+the|\?|$)", text, re.I)
    if entity_match:
        result["target_entity"] = " ".join(entity_match.group(1).split()).strip(" .")
    return result


def _exact_name_presence_phrase(question: str) -> str | None:
    """Recognize a request to find one full name, not separate keywords."""

    lowered = question.lower()
    if "present" not in lowered and "found" not in lowered:
        return None
    match = re.match(
        r"^\s*([A-Za-z][A-Za-z .'-]+?)\s+(?:does|is|was|were|are)\b",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    phrase = " ".join(match.group(1).split())
    return phrase if len(phrase.split()) >= 2 else None


def _fallback_investigation_spec(question: str) -> dict[str, object]:
    """Create a bounded, explainable specification when model JSON is unavailable."""

    lowered = question.lower()
    location_intent = _extract_location_intent(question)
    exact_name = _exact_name_presence_phrase(question)
    subject = "employee" if any(term in lowered for term in ("employee", "worker", "staff")) else "person described in document"
    event: str | None = None
    question_type = "factual_extraction"
    answer_strategy = "direct"
    concepts = [question]
    synthesis_question = any(term in lowered for term in ("what evidence", "suggests that", "evidence shows", "experience taking", "end-to-end", "end to end"))
    comparison_question = lowered.startswith("compare ") or "compare " in lowered or " differ " in lowered
    legal_case_question = any(term in lowered for term in (
        "legal issues", "arguments of the parties", "statutory provisions", "precedents",
        "court's reasoning", "court reasoning", "initial dispute", "final conclusion",
    ))
    if exact_name:
        question_type = "attribute_lookup"
        answer_strategy = "direct"
        subject = exact_name
        concepts = [exact_name]
        information_needed = f"whether the exact name {exact_name} appears in the selected document"
    elif legal_case_question:
        question_type = "synthesis"
        answer_strategy = "explain_from_evidence"
        subject = "legal case or court proceeding"
        concepts = [
            "facts background dispute", "issues questions for consideration",
            "submissions arguments petitioner respondent", "statute section article constitution precedent",
            "reasoning analysis held conclusion judgment order",
        ]
        information_needed = "case chronology, issues, party arguments, legal authorities, reasoning, and outcome"
    elif synthesis_question:
        question_type = "synthesis"
        answer_strategy = "aggregate_evidence"
        concepts = ["requirements", "requirements discovery", "workflow mapping", "implementation", "build", "testing", "deployment", "delivery", "end-to-end"]
        information_needed = "evidence supporting the requested conclusion"
    elif comparison_question:
        question_type = "comparison"
        answer_strategy = "compare_evidence"
        concepts = [question]
        information_needed = "evidence for the entities and comparison dimensions"
    elif any(term in lowered for term in ("resign", "resignation", "quit", "leave", "leaving")):
        event = "voluntary employee resignation"
        question_type = "policy_lookup"
        concepts = [
            "employee resignation", "voluntary resignation",
            "employee-initiated resignation", "resignation notice",
            "notice period for resignation",
        ]
    elif "termination" in lowered or "terminate" in lowered:
        event = "employment termination"
        question_type = "policy_lookup"
        concepts = ["employment termination", "termination notice", "notice period"]
    elif any(term in lowered for term in ("notice period", "policy", "contract condition", "obligation")):
        event = "policy requirement"
        question_type = "policy_lookup"
    elif any(term in lowered for term in ("skill", "skills")):
        question_type = "list_extraction"
        concepts = ["skills", "technical skills", "professional skills"]
    elif any(term in lowered for term in ("when did", "joined", "start date", "date")):
        question_type = "temporal_lookup"
    elif any(term in lowered for term in ("name", "email", "phone", "degree", "university", "college", "company", "work", "employed")):
        question_type = "attribute_lookup"
    if "probation" in lowered:
        subject = "probation employee"
        concepts.insert(0, "probation employee")
    if "permanent" in lowered:
        subject = "permanent employee"
        concepts.insert(0, "permanent employee resignation")
    information_needed = (
        information_needed if exact_name else
        "notice period" if any(term in lowered for term in ("notice", "period", "days", "month")) else "the requested answer"
    )
    target_attribute = None
    if exact_name:
        target_attribute = "exact_name_presence"
        concepts = [exact_name]
    elif "phone" in lowered or "mobile" in lowered or "telephone" in lowered or "contact number" in lowered:
        information_needed = "phone number"
        target_attribute = "phone"
        concepts = ["phone", "mobile", "telephone", "contact number"]
    elif "date of birth" in lowered or "birth date" in lowered:
        information_needed = "date of birth"
        target_attribute = "date_of_birth"
        concepts = ["date of birth", "birth date", "dob"]
    elif "case" in lowered and "name" in lowered:
        information_needed = "case name"
        target_attribute = "case_name"
        concepts = ["case name", "case title", "matter", "petition", "proceedings"]
    elif "name" in lowered:
        information_needed = "person's name"
        target_attribute = "name"
        concepts = ["name", "full name", "candidate name"]
    elif "email" in lowered:
        information_needed = "email address"
        target_attribute = "email"
        concepts = ["email", "email address", "contact details"]
    elif "university" in lowered or "college" in lowered:
        information_needed = "university attended"
        target_attribute = "university"
        concepts = ["university", "college", "education"]
    elif "degree" in lowered:
        information_needed = "degree or qualification"
        target_attribute = "degree"
        concepts = ["degree", "qualification", "education"]
    elif "skill" in lowered:
        information_needed = "listed skills"
        target_attribute = "skills"
        concepts = ["skills", "technical skills", "professional skills", "core strengths"]
    elif any(term in lowered for term in ("work", "employed", "company")) and event is None:
        information_needed = "current employer or company"
        target_attribute = "company"
        concepts = ["company", "employer", "work", "employment", "experience"]
    if location_intent.get("target_entity") and location_intent.get("document_location"):
        target_entity = str(location_intent["target_entity"])
        subject = target_entity
        information_needed = f"name of {target_entity}"
        target_attribute = "name"
        concepts = list(dict.fromkeys([target_entity, "name", "full name", "identification", question]))[:8]
    if legal_case_question:
        intent = "analyze_legal_case_record"
        requirements = [
            "factual background or procedural history",
            "legal issues for determination",
            "arguments or submissions of the parties",
            "statutory provisions or precedents",
            "court reasoning or final outcome",
        ]
    elif synthesis_question:
        intent = "evaluate_experience_from_evidence"
        requirements = [
            "evidence of requirements discovery or definition",
            "evidence of implementation or build",
            "evidence of testing",
            "evidence of deployment or delivery",
        ]
    elif comparison_question:
        intent = "compare_entities_from_evidence"
        requirements = [
            "evidence describing the first entity",
            "evidence describing the second entity",
            "evidence addressing the requested comparison dimensions",
        ]
    else:
        intent = "retrieve_policy_requirement" if event else "factual_extraction"
        requirements = [
            f"must identify the applicable {subject}",
            f"must address {event}" if event else "must directly identify the requested attribute",
            f"must state {information_needed}",
        ]
    return {
        "intent": intent,
        "question_type": question_type,
        "target_attribute": target_attribute,
        "answer_strategy": answer_strategy,
        "subject": subject,
        "event": event,
        "information_needed": information_needed,
        "speakers": _named_speakers(question),
        "constraints": ["must be directly supported by the source"],
        "search_concepts": concepts[:8],
        "exact_phrase": exact_name,
        "negative_concepts": (
            ["employer termination", "involuntary termination", "probation termination", "contractor termination"]
            if event and "resignation" in event else []
        ),
        "evidence_requirements": requirements,
        **location_intent,
        "target_entity": location_intent.get("target_entity") or subject,
    }


def _coerce_general_spec(spec: dict[str, object], question: str) -> dict[str, object]:
    """Keep model output bounded while correcting obvious question-shape misclassification."""

    fallback = _fallback_investigation_spec(question)
    location_intent = _extract_location_intent(question)
    lowered = question.lower()
    synthesis = any(term in lowered for term in ("what evidence", "suggests that", "evidence shows", "experience taking", "end-to-end", "end to end"))
    legal_case = any(term in lowered for term in (
        "legal issues", "arguments of the parties", "statutory provisions", "precedents",
        "court's reasoning", "court reasoning", "initial dispute", "final conclusion",
    ))
    comparison = lowered.startswith("compare ") or "compare " in lowered or " differ " in lowered
    direct_attribute = fallback.get("target_attribute")
    if direct_attribute and not spec.get("event"):
        # The model can return syntactically valid but generic JSON. For a
        # clearly named attribute, preserve a deterministic, bounded target so
        # inspection cannot drift to unrelated prose.
        for key in (
            "intent", "question_type", "target_attribute", "answer_strategy",
            "information_needed", "evidence_requirements", "search_concepts",
        ):
            spec[key] = fallback[key]
    if synthesis or comparison or legal_case:
        for key in ("question_type", "answer_strategy", "information_needed", "evidence_requirements", "search_concepts", "intent"):
            spec[key] = fallback[key]
        spec["event"] = None
    if spec.get("question_type") not in QUESTION_TYPES:
        spec["question_type"] = fallback["question_type"]
    if spec.get("answer_strategy") not in ANSWER_STRATEGIES:
        spec["answer_strategy"] = fallback["answer_strategy"]
    requirements = spec.get("evidence_requirements")
    clauses = _split_subquestions(question)
    generic_requirements = {
        "court transcripts", "legal news reports", "case orders", "documents", "sources",
    }
    if len(clauses) > 1 and (
        not isinstance(requirements, list)
        or not requirements
        or all(str(value).strip().lower().rstrip(".") in generic_requirements for value in requirements)
        or all(re.match(r"^must\b.*\b(?:identify|address|state)\b", str(value).strip().lower()) for value in requirements)
    ):
        spec["evidence_requirements"] = clauses
        spec["information_needed"] = "; ".join(clauses)
        spec["search_concepts"] = list(dict.fromkeys(
            [str(value) for value in spec.get("search_concepts", []) if str(value).strip()]
            + clauses
            + _named_speakers(question)
        ))[:8]
    elif not isinstance(requirements, list) or not requirements:
        spec["evidence_requirements"] = fallback["evidence_requirements"]
    if not spec.get("speakers"):
        spec["speakers"] = _named_speakers(question)
    if location_intent.get("document_location") or location_intent.get("region_type"):
        for key, value in location_intent.items():
            if value is not None:
                spec[key] = value
        if location_intent.get("target_entity"):
            spec["target_entity"] = location_intent["target_entity"]
            spec["subject"] = location_intent["target_entity"]
            spec["target_attribute"] = spec.get("target_attribute") or "name"
            spec["information_needed"] = f"name of {location_intent['target_entity']}"
        spec["search_concepts"] = list(dict.fromkeys(
            [str(location_intent.get("target_entity") or ""), "name", "header", "title", "first page", "last page"]
            + [str(value) for value in spec.get("search_concepts", []) if value]
        ))[:8]
    generic = _generic_planner_variants(question)
    validation = _validate_planner_output(question, spec)
    for key in (
        "target_entity", "target_attribute", "requested_relationship", "requested_action",
        "expected_answer_type", "semantic_intent", "structural_hints",
    ):
        if generic.get(key) and not spec.get(key):
            spec[key] = generic[key]
    # The original wording and deterministic variants are always retained. If
    # the model invented a source scenario, replace only the over-specific
    # planning fields; document search and verification remain unchanged.
    if validation["fallback_used"]:
        spec["information_needed"] = str(generic["semantic_intent"])
        spec["evidence_requirements"] = [str(generic["semantic_intent"])]
        if generic.get("target_attribute"):
            spec["target_attribute"] = generic["target_attribute"]
        if generic.get("target_entity"):
            spec["target_entity"] = generic["target_entity"]
        if generic.get("requested_relationship"):
            spec["requested_relationship"] = generic["requested_relationship"]
        if generic.get("requested_action"):
            spec["requested_action"] = generic["requested_action"]
        spec["expected_answer_type"] = generic["expected_answer_type"]
        spec["semantic_intent"] = generic["semantic_intent"]
        spec["structural_hints"] = generic["structural_hints"]
    spec["original_question_terms"] = generic["original_question_terms"]
    spec["planner_variants"] = list(dict.fromkeys([
        *[str(value) for value in spec.get("search_concepts", []) if str(value).strip()],
        *[str(value) for value in generic["lexical_variants"] if str(value).strip()],
    ]))[:20]
    spec["planner_validation"] = validation
    spec["search_concepts"] = list(dict.fromkeys([
        *[str(value) for value in spec.get("search_concepts", []) if str(value).strip()],
        *[str(value) for value in generic["lexical_variants"] if str(value).strip()],
        question,
    ]))[:16]
    return spec


def _named_speakers(question: str) -> list[str]:
    """Extract person-like names for hard transcript attribution filtering."""

    patterns = [
        r"\b(?:[A-Z]\.){1,4}\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b",
        r"\b(?:Chief Justice|Justice|Judge|Professor|Dr\.?)\s+[A-Z][A-Za-z.]+(?:\s+[A-Z][A-Za-z.]+){1,3}\b",
        r"\b(?:according to|said by|spoken by|addressed by|appearing for|appeared for|submission(?: of)?)\s+([A-Z](?:[A-Za-z.']+)(?:\s+[A-Z][A-Za-z.']+){0,3})\b",
        r"\b([A-Z][A-Za-z.']+(?:\s+[A-Z][A-Za-z.']+){1,3})\s+(?=(?:mentioned|mention|said|say|argued|argue|stated|state|appearing|appeared)\b)",
    ]
    ignored = {"Who Was", "What Did", "According To", "Chief Justice", "The Court"}
    matches: list[str] = []
    for pattern in patterns:
        for value in re.findall(pattern, question):
            cleaned = " ".join(value.split()).strip(" ,?.")
            if cleaned not in ignored and cleaned not in matches and len(cleaned.split()) >= 2:
                matches.append(cleaned)
    return matches[:8]


def _speaker_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _speaker_name_matches(query_name: str, actual_name: str) -> bool:
    """Fuzzy-match a question name to a real parsed speaker label."""

    query = _speaker_key(query_name)
    actual = _speaker_key(actual_name)
    if not query or not actual:
        return False
    if query in actual or actual in query:
        return True
    query_parts = [part for part in re.findall(r"[a-z]+", str(query_name).lower()) if part not in {"justice", "chief", "judge", "dr", "professor"}]
    actual_parts = [part for part in re.findall(r"[a-z]+", str(actual_name).lower()) if part not in {"justice", "chief", "judge", "dr", "professor"}]
    if not query_parts or not actual_parts:
        return False
    if query_parts[-1] != actual_parts[-1]:
        return False
    query_initials = "".join(part[0] for part in query_parts[:-1])
    actual_initials = "".join(part[0] for part in actual_parts[:-1])
    return bool(query_initials and actual_initials and (query_initials == actual_initials or query_initials in actual_initials or actual_initials in query_initials))


def _speaker_constraint_diagnostic(state: AgentState, source: str) -> dict[str, object]:
    """Validate question speaker hints against one selected document's table."""

    spec = state.get("investigation_spec") or {}
    requested = [str(value).strip() for value in spec.get("speakers", []) if str(value).strip()]
    document_map = (state.get("document_maps") or {}).get(source, {})
    has_turns = bool(document_map.get("has_speaker_turns"))
    actual = [str(value) for value in document_map.get("speaker_names", []) if str(value).strip()]
    if not requested:
        return {"constraint": "speaker", "detected": False, "validated": False, "dropped": False, "reason": "no speaker named in question", "document": source}
    if not has_turns:
        return {"constraint": "speaker", "detected": True, "validated": False, "dropped": True, "reason": "document has no meaningful speaker-turn table", "document": source, "requested": requested, "actual_speakers": actual}
    matched = [name for name in requested if any(_speaker_name_matches(name, candidate) for candidate in actual)]
    if not matched:
        return {"constraint": "speaker", "detected": True, "validated": False, "dropped": True, "reason": "speaker_constraint_dropped: no matching turns", "document": source, "requested": requested, "actual_speakers": actual}
    return {"constraint": "speaker", "detected": True, "validated": True, "dropped": False, "reason": "matched parsed speaker table", "document": source, "requested": requested, "validated_speakers": matched, "actual_speakers": actual}


def _normalize_investigation_spec(candidate: object, question: str) -> dict[str, object] | None:
    if not isinstance(candidate, dict):
        return None
    required = ("intent", "subject", "information_needed")
    normalized_required = {field: _planner_text(candidate.get(field)) for field in required}
    if any(not normalized_required[field] for field in required):
        return None
    event_text = _planner_text(candidate.get("event"))
    question_type = _planner_text(candidate.get("question_type"))
    if question_type not in QUESTION_TYPES:
        question_type = "unknown"
    normalized = {
        "intent": normalized_required["intent"],
        "question_type": question_type,
        "answer_strategy": _normalize_answer_strategy(candidate.get("answer_strategy")),
        "subject": normalized_required["subject"],
        "event": event_text or None,
        "information_needed": normalized_required["information_needed"],
        "target_attribute": _planner_text(candidate.get("target_attribute")) or None,
        "speakers": _planner_list(candidate.get("speakers")) or _named_speakers(question),
    }
    location_intent = _extract_location_intent(question)
    for key in ("document_location", "region_type", "page_number", "sheet", "column", "direct_identification_required", "target_entity"):
        value = candidate.get(key)
        normalized[key] = value if value is not None else location_intent.get(key)
    if location_intent.get("document_location") or location_intent.get("region_type"):
        normalized["direct_identification_required"] = True
        normalized["target_entity"] = location_intent.get("target_entity") or normalized.get("target_entity") or normalized["subject"]
    for field in ("constraints", "search_concepts", "negative_concepts", "evidence_requirements"):
        normalized[field] = _planner_list(candidate.get(field, []))
    if not normalized["search_concepts"]:
        normalized["search_concepts"] = _fallback_investigation_spec(question)["search_concepts"]
    return normalized


def _spec_search_text(state: AgentState) -> str:
    spec = state.get("investigation_spec") or _fallback_investigation_spec(state["question"])
    concepts = spec.get("search_concepts", [])
    return " ".join([
        state.get("investigation_query") or state["question"],
        state["question"],
        str(spec.get("subject", "")), str(spec.get("event", "")),
        str(spec.get("information_needed", "")), " ".join(concepts),
        " ".join(str(value) for value in spec.get("planner_variants", [])),
        " ".join(str(value) for value in spec.get("original_question_terms", [])),
    ])


def _split_subquestions(question: str) -> list[str]:
    """Split multiple asks without assuming a domain-specific vocabulary."""

    text = " ".join(str(question).split()).strip(" ?.")
    parts = re.split(r"\s*(?:,\s*)?(?:and|;|\?|\bthen\b)\s+(?=(?:who|what|when|where|why|how|which|whether|does|is|are|can|could|did)\b)", text, flags=re.I)
    return [part.strip(" ,") for part in parts if len(part.strip()) > 2] or [text]


def _requirement_ledger(question: str, spec: dict) -> list[dict]:
    """Build atomic, form-aware requirements from the planner plus deterministic splitting."""

    clauses = _split_subquestions(question)
    planned = [str(value) for value in spec.get("evidence_requirements", []) if str(value).strip()]
    count = max(len(clauses), len(planned), 1)
    ledger: list[dict] = []
    for index in range(count):
        clause = clauses[index] if index < len(clauses) else planned[index]
        description = planned[index] if index < len(planned) else clause
        lowered = f"{clause} {description}".lower()
        if (
            "why" in lowered
            or "reason" in lowered
            or re.search(r"\bhow\b.*\b(?:connect|relate|link|contribute|lead|support|work)\b", lowered)
            or "explain" in lowered
        ):
            expected_form = "reason"
            variants = [description, clause, "reason", "because", "rationale"]
            kind = "reasoning"
        elif re.search(r"\b(?:how long|how many minutes|time limit|duration|when)\b", lowered):
            expected_form = "duration_or_date"
            variants = [description, clause, "time allowed", "minutes", "duration", "date"]
            kind = "fact"
        elif re.search(r"\b(?:who|appearing for|according to|speaker|author)\b", lowered):
            expected_form = "person_or_organization"
            variants = [description, clause, "appeared for", "on behalf of", "representing"]
            kind = "attribution"
        elif re.search(r"\b(?:four|three|five|list|which areas|what issues)\b", lowered):
            expected_form = "list"
            variants = [description, clause, "items", "issues", "areas"]
            kind = "list"
        else:
            expected_form = "free_text"
            variants = [description, clause]
            kind = "fact"
        variants.extend(str(value) for value in spec.get("planner_variants", []) if str(value).strip())
        entity_matches = re.findall(r"\b[A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,3}\b", clause)
        assigned_relation = bool(re.search(
            r"\b(?:given|assigned|allowed|ordered|required|told|timed|appointed)\b",
            lowered,
        ))
        relation = (
            "time_limit_assigned" if assigned_relation and expected_form == "duration_or_date"
            else "expected_duration" if expected_form == "duration_or_date"
            else "attribution" if expected_form == "person_or_organization"
            else "fact"
        )
        ledger.append({
            "requirement_id": f"R{index + 1}",
            "what": description,
            "type": kind,
            "entities": list(dict.fromkeys(entity_matches))[:8],
            "speaker_or_source": next((entity for entity in entity_matches if "." in entity or len(entity.split()) >= 2), None),
            "expected_form": expected_form,
            "relation": relation,
            "temporal_role": "assigned" if relation == "time_limit_assigned" else "expected" if relation == "expected_duration" else None,
            "variants": list(dict.fromkeys(value for value in variants if value))[:6],
            "depends_on": [f"R{index}"] if index else [],
            "candidate_document_signals": list(dict.fromkeys([
                *[str(value) for value in spec.get("search_concepts", [])],
                *[str(value) for value in spec.get("planner_variants", [])],
            ]))[:12],
            "status": "UNSEARCHED",
            "candidate_evidence_ids": [],
            "verified_evidence_ids": [],
            "gap": None,
        })
    return ledger


def _form_satisfied(expected_form: str, text: str, question: str = "") -> bool:
    lowered = text.lower()
    if expected_form == "duration_or_date":
        return bool(re.search(r"\b\d+\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b|\b(?:19|20)\d{2}\b|\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", lowered))
    if expected_form == "reason":
        return bool(re.search(r"\b(?:because|since|therefore|thus|so that|as a result|reason|why)\b", lowered))
    if expected_form == "person_or_organization":
        return bool(re.search(r"\b(?:for|on behalf of|representing|appearing|appeared)\b", lowered)) or bool(re.search(r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)+\b", text))
    if expected_form == "list":
        return bool(re.search(r"(?:^|\n)\s*(?:\d+[.)]|first|second|third|fourth|fifth|number\s+\d+)", lowered)) or text.count(";") >= 2
    return bool(text.strip())


SINGLE_VALUED_REQUIREMENT_FORMS = {
    "person", "date", "number", "duration", "organization", "yes/no",
    # Planner/ledger compatibility forms used by this application.
    "duration_or_date", "person_or_organization",
}


def _is_single_valued_requirement(requirement: dict) -> bool:
    expected = str(requirement.get("expected_form", "")).strip().lower()
    return expected in SINGLE_VALUED_REQUIREMENT_FORMS


def _conflict_values(expected_form: str, text: str) -> list[str]:
    """Extract comparable values only for single-valued requirement forms."""

    lowered = str(text).lower()
    if expected_form in {"date", "duration_or_date"}:
        patterns = [
            r"\b(?:19|20)\d{2}\b",
            r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
            r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\b",
        ]
    elif expected_form == "duration":
        patterns = [r"\b\d+(?:\.\d+)?\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b"]
    elif expected_form == "number":
        patterns = [r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])"]
    elif expected_form == "yes/no":
        patterns = [r"\b(?:yes|no|affirmative|negative)\b"]
    else:
        # Person/organization conflicts require an explicit value relation;
        # arbitrary names in explanatory prose are not conflicting values.
        return []
    values: list[str] = []
    for pattern in patterns:
        values.extend(match.group(0).lower() for match in re.finditer(pattern, lowered))
    return list(dict.fromkeys(values))


def _specific_requirement_gap(state: AgentState, requirement: str) -> str:
    pages = sum(
        len(document.get("pages") or [])
        for document in state.get("inspected_documents", [])
    )
    pages = pages or len(state.get("investigation_history", []))
    strategies = ", ".join(str(value) for value in state.get("search_paths_attempted", []) if str(value).strip())
    return (
        f"Requirement '{requirement}' was not verified; searched {pages} page/region(s) "
        f"using strategies: {strategies or 'targeted inspection'}."
    )


def analyze_question(state: AgentState) -> dict[str, object]:
    """Use the configured LLM to classify the investigation needed."""

    reset_observability()
    question = state["question"]
    fallback = _fallback_investigation_spec(question)
    if fallback.get("target_attribute") == "exact_name_presence":
        return {
            "question_analysis": json.dumps(fallback),
            "investigation_spec": fallback,
            "requirement_ledger": _requirement_ledger(question, fallback),
            "original_question_terms": _question_terms(question),
            "planner_variants": fallback.get("search_concepts", []),
            "planner_validation": {"status": "DETERMINISTIC_EXACT_NAME", "fallback_used": True},
            "search_paths_attempted": ["exact_phrase", "structural_map", "targeted_window"],
            "investigation_trace": _trace(
                state,
                "question_understanding",
                question_type=fallback.get("question_type"),
                information_needed=fallback.get("information_needed"),
                requirements=fallback.get("evidence_requirements", []),
                planner_fallback=True,
            ),
        }
    try:
        llm = build_llm()
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a document-corpus investigation planner. Return ONLY valid JSON "
                        "with keys intent, subject, event, information_needed, constraints, "
                        "search_concepts, evidence_requirements, question_type, answer_strategy, and target_attribute. The event "
                        "may be null for ordinary factual or attribute questions. Use a bounded "
                        "question_type such as attribute_lookup, list_extraction, temporal_lookup, "
                        "event_extraction, policy_lookup, comparison, document_discovery, or unknown. "
                        "Keep each list bounded. "
                        "Do not answer the question."
                    )
                ),
                HumanMessage(content=question),
            ]
        )
        raw = str(response.content)
        match = re.search(r"\{[\s\S]*\}", raw)
        candidate = json.loads(match.group(0)) if match else None
        spec = _normalize_investigation_spec(candidate, question) or fallback
        spec = _coerce_general_spec(spec, question)
    except Exception:
        # A temporary planner outage must not discard the question's
        # sub-question structure. Coerce the deterministic fallback so
        # multi-part questions still produce atomic requirements and speakers.
        spec = _coerce_general_spec(fallback, question)
        raw = ""
    return {
        "question_analysis": json.dumps(spec),
        "investigation_spec": spec,
        "requirement_ledger": _requirement_ledger(question, spec),
        "original_question_terms": spec.get("original_question_terms", _question_terms(question)),
        "planner_variants": spec.get("planner_variants", []),
        "planner_validation": spec.get("planner_validation", {}),
        "search_paths_attempted": ["exact_phrase", "structural_map", "targeted_window", "requirement_specific"],
        "investigation_trace": _trace(
            state,
            "question_understanding",
            question_type=spec.get("question_type"),
            original_question=question,
            planner_output=raw[:4000],
            information_needed=spec.get("information_needed"),
            requirements=spec.get("evidence_requirements", []),
            normalized_requirements=spec.get("evidence_requirements", []),
            planner_variants=spec.get("planner_variants", []),
            planner_validation=spec.get("planner_validation", {}),
            fallback_used=spec.get("planner_validation", {}).get("fallback_used", spec == fallback),
            speakers=spec.get("speakers", []),
            planner_fallback=spec == fallback,
        ),
    }


def discover_documents(state: AgentState) -> dict[str, object]:
    """Call the catalog tool without sending document contents to Gemini."""

    documents_dir = Path(state.get("documents_dir", str(DOCUMENTS_DIR)))
    result = list_documents_tool.invoke({"directory": str(documents_dir)})
    documents = result.get("documents", [])
    scope = set(state.get("document_scope") or [])
    if scope:
        documents = [document for document in documents if document["name"] in scope]
    # A small boundary-page check creates a case-record map without reading the
    # whole corpus or using embeddings. It lets later nodes distinguish a hearing
    # transcript from a final judgment when several selected PDFs belong together.
    for document in documents:
        role = classify_document_role_tool.invoke({"path": document["path"]})
        document["record_role"] = role.get("record_role", "unknown")
        document["pages"] = role.get("page_count") or document.get("pages")
    document_maps = {
        str(document["name"]): build_document_map_tool.invoke({"path": document["path"]})
        for document in documents
    }
    for document in documents:
        structural = document_maps.get(str(document["name"]), {})
        document["has_speaker_turns"] = bool(structural.get("has_speaker_turns"))
        document["speaker_names"] = list(structural.get("speaker_names") or [])
        document["speaker_turn_count"] = int(structural.get("speaker_turn_count") or 0)
    assessments = {
        str(document["name"]): {
            "document_relevance": "candidate",
            "document_answerability": "unknown",
            "investigated": False,
            "evidence_found": False,
            "requirements_covered": [],
            "remaining_gaps": [],
        }
        for document in documents
    }
    return {
        "available_documents": documents,
        "active_corpus": documents,
        "corpus_map": {
            "documents": [str(document["name"]) for document in documents],
            "document_count": len(documents),
            "total_pages": sum(int(document.get("pages", 0) or 0) for document in documents),
        },
        "document_maps": document_maps,
        "document_assessments": assessments,
        "document_relevance": {str(document["name"]): "candidate" for document in documents},
        "corpus_metrics": {
            "total_documents": len(documents),
            "total_bytes": sum(int(document.get("size_bytes", 0) or 0) for document in documents),
            "total_pages": sum(int(document.get("pages", 0) or 0) for document in documents),
            "documents_considered": [str(document["name"]) for document in documents],
            "document_maps_cached": len(document_maps),
        },
        "investigation_trace": _trace(
            state,
            "document_discovery",
            available_sources=[str(document["name"]) for document in documents],
            selected_sources=list(state.get("document_scope") or []),
            documents_considered=[str(document["name"]) for document in documents],
            total_pages=sum(int(document.get("pages", 0) or 0) for document in documents),
            document_maps_cached=len(document_maps),
            speaker_structure={
                str(document["name"]): {
                    "has_speaker_turns": bool(document.get("has_speaker_turns")),
                    "speaker_names": list(document.get("speaker_names") or []),
                    "speaker_turn_count": int(document.get("speaker_turn_count") or 0),
                }
                for document in documents
            },
        ),
    }


def _metadata_fallback_selection(question: str, catalog: list[dict]) -> list[str]:
    """Select candidates from lightweight metadata when the LLM selector fails."""

    stop_words = {
        "what", "which", "when", "where", "who", "does", "the", "are", "was",
        "were", "this", "that", "these", "those", "about", "across", "from",
        "with", "into", "have", "has", "how", "can", "may", "any", "all",
    }
    question_terms = {
        term.lower()
        for term in re.findall(r"[a-zA-Z]+", question)
        if len(term) > 3 and term.lower() not in stop_words
    }
    scored: list[tuple[int, str]] = []
    for document in catalog:
        metadata_text = " ".join(
            str(document.get(field, ""))
            for field in ("name", "title", "description", "suffix", "entities")
        ).lower()
        score = sum(1 for term in question_terms if term in metadata_text)
        if score:
            scored.append((score, str(document["name"])))
    if not scored:
        return [str(document["name"]) for document in catalog]
    highest_score = max(score for score, _ in scored)
    return [name for score, name in scored if score == highest_score]


def _map_section_candidates(state: AgentState, selected: list[str]) -> tuple[dict, dict]:
    """Rank compact structural sections before opening raw document content."""

    spec = state.get("investigation_spec") or {}
    signal_text = " ".join([
        state.get("question", ""),
        str(spec.get("information_needed", "")),
        " ".join(str(item) for item in spec.get("search_concepts", [])),
        "; ".join(str(item) for item in (state.get("investigation_gaps") or [])),
    ]).lower()
    terms = {
        term for term in re.findall(r"[a-z0-9]{3,}", signal_text)
        if term not in {"what", "which", "when", "where", "who", "does", "the", "and", "for", "with", "this"}
    }
    attribution_terms: set[str] = set()
    for match in re.findall(
        r"\b(?:according to|did|said|says|submission(?: of)?|addressed by)\s+([A-Z][A-Za-z.']+(?:\s+[A-Z][A-Za-z.']+){0,3})",
        state.get("question", ""),
    ):
        cleaned = re.sub(r"['’]s$", "", match).strip().lower()
        attribution_terms.update(cleaned.split())
    history = state.get("investigation_history") or []
    prior_locations = {
        (entry.get("source"), entry.get("label"), entry.get("page_start"), entry.get("line_start"), entry.get("sheet"), entry.get("row_start"))
        for entry in history
    }
    candidates: dict[str, list[dict]] = {}
    plan: dict[str, dict] = {}
    maps = state.get("document_maps") or {}
    for source in selected:
        sections = []
        for section in maps.get(source, {}).get("sections", []):
            searchable = " ".join(str(section.get(key, "")) for key in ("label", "description", "headings")).lower()
            score = sum(1 for term in terms if term in searchable)
            if attribution_terms and any(term in searchable for term in attribution_terms if len(term) > 2):
                score += 8
            location_key = (source, section.get("label"), section.get("page_start"), section.get("line_start"), section.get("sheet"), section.get("row_start"))
            if location_key in prior_locations:
                score -= 2
            sections.append({**section, "source": source, "signal_score": score})
        sections.sort(key=lambda section: (section.get("signal_score", 0), -int(section.get("page_start", section.get("line_start", section.get("row_start", 0))) or 0)), reverse=True)
        chosen = sections[:8]
        candidates[source] = chosen
        plan[source] = {
            "sections": chosen,
            "strategy": "structural_map_then_targeted_window",
            "prior_locations_considered": sum(1 for section in sections if (source, section.get("label"), section.get("page_start"), section.get("line_start"), section.get("sheet"), section.get("row_start")) in prior_locations),
        }
    return candidates, plan


def select_documents(state: AgentState) -> dict[str, object]:
    """Ask the planner which discovered filenames should be investigated."""

    catalog = [
        {key: value for key, value in document.items() if key != "path"}
        for document in state["available_documents"]
    ]
    names = [document["name"] for document in state["available_documents"]]
    requested_scope = [name for name in (state.get("document_scope") or []) if name in names]
    if state.get("source_scope_explicit") or "document_scope" in state:
        assessments = {
            str(name): {
                **(state.get("document_assessments", {}).get(str(name), {})),
                "document_relevance": "high",
                "selection_reason": "Explicitly included in the active document scope.",
            }
            for name in names
        }
        section_candidates, inspection_plan = _map_section_candidates(state, requested_scope)
        return {
            "selected_documents": requested_scope,
            "candidate_documents": requested_scope,
            "document_relevance": {str(name): ("high" if name in requested_scope else "low") for name in names},
            "document_assessments": assessments,
            "section_candidates": section_candidates,
            "inspection_plan": inspection_plan,
            "investigation_trace": _trace(state, "document_selection", selected=requested_scope, selected_sources=requested_scope, available_sources=names, reason="explicit document scope"),
        }
    selected: list[str] = []
    try:
        llm = build_llm()
        investigation_query = _spec_search_text(state)
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You select documents for a document investigation. "
                        "Use the catalog metadata to select candidates; do not assume the "
                        "filename alone explains the document. Return ONLY a JSON array "
                        "of exact filenames from the provided catalog. "
                        "Do not include explanations or filenames not in the list."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Investigation request: {investigation_query}\n"
                        f"Document catalog: {json.dumps(catalog)}\n"
                        "Select every filename that may contain evidence needed to answer the question."
                    )
                ),
            ]
        )
        match = re.search(r"\[[\s\S]*\]", str(response.content))
        if match:
            try:
                candidate = json.loads(match.group(0))
                if isinstance(candidate, list):
                    selected = [name for name in candidate if name in names]
            except json.JSONDecodeError:
                selected = []
    except Exception:
        selected = []

    # Safe fallback: if the model output is malformed, investigate the known
    # corpus rather than inventing a path or returning an unsupported answer.
    fallback_catalog = catalog
    selected = selected or _metadata_fallback_selection(
        state["question"], fallback_catalog
    ) or names
    assessments = {
        str(name): {
            **(state.get("document_assessments", {}).get(str(name), {})),
            "document_relevance": "high" if name in selected else "low",
            "selection_reason": "LLM/catalog signals selected this candidate." if name in selected else "Not selected by the candidate-document stage.",
        }
        for name in names
    }
    section_candidates, inspection_plan = _map_section_candidates(state, selected)
    return {
        "selected_documents": selected,
        "candidate_documents": selected,
        "document_relevance": {str(name): ("high" if name in selected else "low") for name in names},
        "document_assessments": assessments,
        "section_candidates": section_candidates,
        "inspection_plan": inspection_plan,
        "investigation_trace": _trace(state, "document_selection", selected=selected, selected_sources=selected, available_sources=names, candidate_count=len(selected)),
    }


def inspect_selected_documents(state: AgentState) -> dict[str, object]:
    """Call the inspection tool only for documents selected by the prior node."""

    metadata_by_name = {
        document["name"]: document for document in state["available_documents"]
    }
    inspected = []
    spec = state.get("investigation_spec") or {}
    constraint_diagnostics = []
    target_attribute = spec.get("target_attribute")
    attribute_hints = {
        "name": "name full name candidate profile header",
        "email": "email email address contact",
        "phone": "phone mobile telephone contact number",
        "university": "university college education degree",
        "degree": "degree qualification education",
        "skills": "skills strengths technical skills core strengths",
        "company": "company employer work employment experience",
    }
    investigation_query = (
        str(spec.get("exact_phrase"))
        if target_attribute == "exact_name_presence"
        else f"{state['question']} {attribute_hints.get(str(target_attribute), target_attribute)}"
        if target_attribute and not spec.get("event")
        else _spec_search_text(state)
    )
    section_hints = " ".join(
        str(section.get("label", ""))
        for sections in (state.get("section_candidates") or {}).values()
        for section in sections[:4]
        if section.get("label")
    )
    if section_hints:
        investigation_query = f"{investigation_query} {section_hints}"
    window_size = max(1, int(state.get("inspection_window_size") or 1))
    if (state.get("investigation_spec") or {}).get("question_type") == "list_extraction":
        window_size = max(window_size, 3)
    for name in state["selected_documents"]:
        metadata = metadata_by_name.get(name)
        if metadata:
            constraint = _speaker_constraint_diagnostic(state, name)
            constraint_diagnostics.append(constraint)
            planned_sections = (state.get("section_candidates") or {}).get(name, [])[:8]
            location = spec.get("document_location")
            location_section = None
            if location == "first_page" or spec.get("page_number") == 1:
                location_section = {"page_start": 1, "page_end": 1, "label": "first_page", "signal_score": 1000}
            elif location == "last_page":
                last_page = int(metadata.get("pages") or 0)
                if last_page:
                    location_section = {"page_start": last_page, "page_end": last_page, "label": "last_page", "signal_score": 1000}
            elif spec.get("page_number"):
                page = int(spec["page_number"])
                location_section = {"page_start": page, "page_end": page, "label": f"page_{page}", "signal_score": 1000}
            if location_section:
                planned_sections = [location_section] + [section for section in planned_sections if section.get("label") != location_section["label"]]
            planned_sections = planned_sections or [None]
            for section in planned_sections:
                payload = {
                    "path": metadata["path"],
                    "query": investigation_query,
                    "window_size": window_size,
                    "exact_phrase": str(spec.get("exact_phrase") or ""),
                }
                if spec.get("region_type") in {"header", "title", "heading", "title/header"}:
                    payload["query"] = f"{payload['query']} title header heading"
                if spec.get("sheet"):
                    payload["sheet"] = spec["sheet"]
                if section:
                    if section.get("page_start") is not None:
                        payload["page_start"] = max(1, int(section["page_start"]) - window_size)
                        payload["page_end"] = int(section.get("page_end", section["page_start"])) + window_size
                    if section.get("line_start") is not None:
                        payload["line_start"] = max(1, int(section["line_start"]) - window_size)
                        payload["line_end"] = int(section.get("line_end", section["line_start"])) + window_size
                    if section.get("paragraph_start") is not None:
                        payload["paragraph_start"] = max(1, int(section["paragraph_start"]) - window_size)
                        payload["paragraph_end"] = int(section.get("paragraph_end", section["paragraph_start"])) + window_size
                    if section.get("row_start") is not None:
                        payload["row_start"] = max(1, int(section["row_start"]) - window_size)
                        payload["row_end"] = int(section.get("row_end", section["row_start"])) + window_size
                    if section.get("sheet"):
                        payload["sheet"] = section["sheet"]
                result = inspect_document_window_tool.invoke(payload)
                result.setdefault("metadata", {})["map_section"] = section or {}
                result.setdefault("metadata", {})["has_speaker_turns"] = bool(metadata.get("has_speaker_turns"))
                result.setdefault("metadata", {})["speaker_names"] = list(metadata.get("speaker_names") or [])
                result.setdefault("metadata", {})["validated_speaker_constraints"] = list(constraint.get("validated_speakers") or [])
                result.setdefault("metadata", {})["speaker_constraint_dropped"] = bool(constraint.get("dropped"))
                result.setdefault("metadata", {})["speaker_constraint_reason"] = constraint.get("reason")
                if "transcript of hearing" in str(result.get("content", "")).lower():
                    result.setdefault("metadata", {})["record_kind"] = "hearing_transcript"
                inspected.append(result)
    assessments = dict(state.get("document_assessments", {}))
    for result in inspected:
        source = str(result.get("source", ""))
        metadata = dict(assessments.get(source, {}))
        matched = int(result.get("metadata", {}).get("matched_block_count", 0) or 0)
        metadata.update({
            "investigated": True,
            "evidence_found": matched > 0,
            "document_answerability": "likely" if matched > 0 else "unknown",
            "remaining_gaps": [] if matched > 0 else ["Relevant answer-bearing section not located."],
        })
        assessments[source] = metadata
    history = list(state.get("investigation_history") or [])
    for result in inspected:
        source = str(result.get("source", ""))
        for block in result.get("blocks", []):
            history.append({
                "source": source,
                "page": block.get("page"),
                "sheet": block.get("sheet"),
                "row": block.get("row"),
                "line": block.get("line"),
                "paragraph": block.get("paragraph"),
                "section": block.get("section"),
                "window_size": window_size,
            })
    inspected_locations = [
        {
            "source": result.get("source"),
            "pages": [page.get("page") for page in result.get("pages", []) if page.get("page") is not None],
            "rows": [row.get("row") for row in result.get("rows", []) if row.get("row") is not None],
            "returned_blocks": result.get("metadata", {}).get("returned_block_count", 0),
            "matched_blocks": result.get("metadata", {}).get("matched_block_count", 0),
        }
        for result in inspected
    ]
    document_investigation = dict(state.get("document_investigation") or {})
    documents_with_candidates = set(state.get("documents_with_candidates") or [])
    for result in inspected:
        source = str(result.get("source", ""))
        matched = int(result.get("metadata", {}).get("matched_block_count", 0) or 0)
        entry = dict(document_investigation.get(source) or {})
        entry.update({
            "document": source,
            "status": "CANDIDATES_FOUND" if matched else entry.get("status", "UNSEARCHED"),
            "candidate_regions": entry.get("candidate_regions", []) + [
                {
                    "pages": [page.get("page") for page in result.get("pages", []) if page.get("page") is not None],
                    "rows": [row.get("row") for row in result.get("rows", []) if row.get("row") is not None],
                    "matched_blocks": matched,
                    "search_path": result.get("metadata", {}).get("search_path", "targeted_window"),
                }
            ],
        })
        document_investigation[source] = entry
        if matched:
            documents_with_candidates.add(source)
    return {
        "inspected_documents": inspected,
        "document_assessments": assessments,
        "investigation_history": history[-200:],
        "attribution_context": {
            "question_entities": list((state.get("investigation_spec") or {}).get("entities", [])),
            "speaker_hints": list((state.get("investigation_spec") or {}).get("speakers", [])),
            "locations_read": inspected_locations,
        },
        "search_paths_attempted": list(dict.fromkeys([
            *(state.get("search_paths_attempted") or []),
            "structural_map",
            "targeted_window",
        ])),
        "document_investigation": document_investigation,
        "documents_with_candidates": sorted(documents_with_candidates),
        "investigation_trace": _trace(
            state,
            "targeted_inspection",
            selected_sources=list(state.get("selected_documents") or []),
            investigated_sources=sorted({str(result.get("source")) for result in inspected}),
            window_size=window_size,
            locations=inspected_locations,
            document_investigation=document_investigation,
            constraints=constraint_diagnostics,
            candidate_paths=["structural_map", "targeted_window"],
        ),
    }


def extract_evidence(state: AgentState) -> dict[str, list[EvidenceItem]]:
    """Extract question-matching passages while preserving provenance."""
    evidence: list[EvidenceItem] = []

    for inspected in state.get("inspected_documents", []):
        result = extract_relevant_section_tool.invoke({
            "question": state["question"],
            "investigated_query": (
                f"{state['question']} {state.get('investigation_spec', {}).get('target_attribute', '')}"
                if state.get("investigation_spec", {}).get("target_attribute")
                and not state.get("investigation_spec", {}).get("event")
                else _spec_search_text(state)
            ),
            "investigation_spec": state.get("investigation_spec") or {},
            "inspected_document": inspected,
        })
        for item in result.get("evidence", []):
            evidence_item: EvidenceItem = {
                "source_document": item["source"],
                "excerpt": item["text"],
                "page": item.get("page"),
            }
            for provenance_key in ("path", "sheet", "row", "line", "paragraph", "section", "table", "speaker", "related_speakers"):
                if item.get(provenance_key) is not None:
                    evidence_item[
                        "source_path" if provenance_key == "path" else provenance_key
                    ] = item[provenance_key]
            inspected_metadata = inspected.get("metadata", {})
            evidence_item["has_speaker_turns"] = bool(inspected_metadata.get("has_speaker_turns"))
            evidence_item["validated_speaker_constraints"] = list(inspected_metadata.get("validated_speaker_constraints") or [])
            evidence.append(evidence_item)

    # A retry is an additive investigation pass. Preserve prior candidates so
    # a broader pass cannot erase evidence found by an earlier pass.
    prior_candidates = list(state.get("candidate_evidence") or [])
    combined_candidates = prior_candidates + evidence
    deduped_candidates: list[EvidenceItem] = []
    seen_candidates: set[tuple[object, ...]] = set()
    for item in combined_candidates:
        identity = (item.get("source_document"), item.get("page"), item.get("sheet"), item.get("row"), item.get("excerpt"))
        if identity not in seen_candidates:
            seen_candidates.add(identity)
            deduped_candidates.append(item)
    documents_with_candidates = sorted({str(item.get("source_document")) for item in deduped_candidates if item.get("source_document")})
    document_investigation = dict(state.get("document_investigation") or {})
    for source in documents_with_candidates:
        entry = dict(document_investigation.get(source) or {})
        entry["status"] = "EVIDENCE_FOUND"
        entry["candidate_evidence_count"] = sum(1 for item in deduped_candidates if item.get("source_document") == source)
        document_investigation[source] = entry
    for source in state.get("selected_documents") or []:
        document_investigation.setdefault(str(source), {"document": str(source), "status": "UNSEARCHED"})
    return {
        "evidence": deduped_candidates,
        "candidate_evidence": deduped_candidates,
        "document_investigation": document_investigation,
        "documents_with_candidates": documents_with_candidates,
        "investigation_gaps": list(state.get("investigation_gaps") or []) if evidence else ["Answer-bearing content has not yet been located."],
        "investigation_trace": _trace(
            state,
            "candidate_evidence",
            count=len(deduped_candidates),
            candidate_sources=documents_with_candidates,
            evidence_preview=[
                {
                    "source": item.get("source_document"),
                    "page": item.get("page"),
                    "speaker": item.get("speaker"),
                    "related_speakers": item.get("related_speakers", []),
                    "text": str(item.get("excerpt", ""))[:320],
                }
                for item in deduped_candidates[:40]
            ],
        ),
    }


def _text_tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[a-zA-Z]+", text)}


def _evidence_signature(items: list[EvidenceItem] | list[dict] | None) -> str:
    parts = sorted(
        f"{item.get('source_document')}|{item.get('page')}|{item.get('excerpt', '')}"
        for item in (items or [])
    )
    return "\n".join(parts)


def verify_evidence(state: AgentState) -> dict[str, object]:
    """Reject lexical matches that do not match the investigated subject/event."""

    spec = state.get("investigation_spec") or _fallback_investigation_spec(state["question"])
    subject = str(spec.get("subject", "")).lower()
    event_value = spec.get("event")
    event = str(event_value).lower() if event_value else ""
    information = str(spec.get("information_needed", "")).lower()
    subject_tokens = _text_tokens(subject)
    event_tokens = _text_tokens(event)
    information_tokens = _text_tokens(information)
    verification: list[dict[str, object]] = []
    verified: list[EvidenceItem] = []
    question_type = str(spec.get("question_type", "unknown"))
    answer_strategy = str(spec.get("answer_strategy", "direct"))
    def speaker_key(value: object) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value).lower())

    question_speakers = [speaker_key(value) for value in spec.get("speakers", []) if str(value).strip()]
    requirements = [
        str(value) for value in spec.get("evidence_requirements", [])
        if isinstance(value, str) and value.strip()
    ]
    evidence_coverage = {
        requirement: {"satisfied": False, "evidence_ids": []}
        for requirement in requirements
    }
    requirement_roles = {
        requirement: (
            "assigned" if re.search(r"\b(?:given|assigned|allowed|ordered|required|told|timed|appointed)\b", requirement.lower())
            else "expected" if re.search(r"\b(?:will|would|can|could|expect|estimate|take|cover)\b", requirement.lower())
            else None
        )
        for requirement in requirements
    }

    def temporal_role(text: str) -> str | None:
        lowered_text = " ".join(text.lower().split())
        if re.search(r"\b(?:timed|given|allowed|assigned|ordered|required|told)\b.{0,45}\b\d+\s*(?:minutes?|hours?)\b", lowered_text):
            return "assigned"
        if re.search(r"\b(?:i|we|he|she|they)\b.{0,60}\b(?:will|would|can|could|expect|estimate|take|cover|finish)\b.{0,60}\b\d+\s*(?:minutes?|hours?)\b", lowered_text):
            return "expected"
        return None

    resignation_event = any(term in event for term in ("resign", "quit", "voluntary", "leave"))
    termination_event = "terminat" in event
    candidates = list(state.get("candidate_evidence", state.get("evidence", [])))
    if len(candidates) > MAX_CANDIDATE_EVIDENCE_ITEMS:
        # Preserve a bounded opportunity for every selected source before
        # global ranking.  A large unrelated document must not crowd the only
        # candidate from a smaller document out of verification.
        def candidate_score(item: EvidenceItem) -> int:
            text = str(item.get("excerpt", "")).lower()
            return sum(1 for term in (*subject_tokens, *event_tokens, *information_tokens) if term in text)

        grouped: dict[str, list[EvidenceItem]] = {}
        for item in candidates:
            grouped.setdefault(str(item.get("source_document", "unknown")), []).append(item)
        for items in grouped.values():
            items.sort(key=candidate_score, reverse=True)
        candidates = []
        while len(candidates) < MAX_CANDIDATE_EVIDENCE_ITEMS and any(grouped.values()):
            progressed = False
            for source in sorted(grouped):
                if grouped[source] and len(candidates) < MAX_CANDIDATE_EVIDENCE_ITEMS:
                    candidates.append(grouped[source].pop(0))
                    progressed = True
            if not progressed:
                break
    for item in candidates:
        text = str(item.get("excerpt", ""))
        lowered = text.lower()
        document_has_speaker_turns = (
            bool(item.get("has_speaker_turns"))
            if "has_speaker_turns" in item
            else bool(item.get("speaker") or item.get("related_speakers"))
        )
        validated_speakers = [speaker_key(value) for value in item.get("validated_speaker_constraints", []) if str(value).strip()]
        if document_has_speaker_turns and "validated_speaker_constraints" not in item:
            validated_speakers = list(question_speakers)
        required_speakers = validated_speakers if document_has_speaker_turns else []
        requested_location = spec.get("document_location")
        item_page = item.get("page")
        source_pages = next((int(document.get("pages") or 0) for document in state.get("available_documents", []) if document.get("name") == item.get("source_document")), 0)
        location_match = (
            (requested_location == "first_page" and item_page == 1)
            or (requested_location == "last_page" and source_pages and item_page == source_pages)
            or (str(requested_location or "").startswith("page_") and item_page == spec.get("page_number"))
            or (requested_location in {"title", "header", "heading"} and item_page == 1)
        )
        tokens = _text_tokens(text)
        item_speaker = speaker_key(item.get("speaker", ""))
        related_speakers = [speaker_key(value) for value in item.get("related_speakers", [])]
        direct_speaker_match = bool(item_speaker and any(
            name in item_speaker or item_speaker in name for name in required_speakers
        ))
        indirect_attribute_match = bool(
            required_speakers
            and any(name in related_speakers or related in required_speakers for name in required_speakers for related in related_speakers)
            and bool(re.search(r"\b(?:timed|minutes?|hours?|time limit|allowed|given)\b", lowered))
        )
        speaker_match = not required_speakers or direct_speaker_match or indirect_attribute_match
        employer_termination = (
            any(term in lowered for term in ("employer", "company", "organization"))
            and any(term in lowered for term in ("terminate", "termination"))
            and not any(term in lowered for term in ("resign", "resignation", "quit"))
        )
        probation_mismatch = "probation" in subject and "probation" not in lowered
        permanent_mismatch = "permanent" in subject and (
            "probation" in lowered or "contractor" in lowered
        )
        contractor_mismatch = "contractor" in subject and "contractor" not in lowered
        distractor_mismatch = (
            ("probation" in lowered and "probation" not in subject)
            or ("contractor" in lowered and "contractor" not in subject)
        )
        employee_subject = any(term in lowered for term in ("employee", "employees", "worker", "staff"))
        subject_match = bool(subject_tokens & tokens) or (
            "employee" in subject and employee_subject and not contractor_mismatch and not permanent_mismatch
        )
        if "employer" in subject:
            subject_match = "employer" in lowered or "company" in lowered
        if "contractor" in subject:
            subject_match = "contractor" in lowered
        if "probation" in subject:
            subject_match = "probation" in lowered
        if "permanent" in subject:
            subject_match = "permanent" in lowered and "probation" not in lowered
        if subject in {"person", "general", "any party"} or "person" in subject or "candidate" in subject:
            subject_match = True
        if spec.get("intent") == "analyze_legal_case_record":
            subject_match = True

        negative_concepts = [
            str(value).lower() for value in spec.get("negative_concepts", [])
            if isinstance(value, str)
        ]
        negative_match = any(
            all(token in lowered for token in _text_tokens(concept))
            for concept in negative_concepts
        )
        if resignation_event:
            event_match = any(term in lowered for term in ("resign", "resignation", "quit", "leave", "voluntary")) and not employer_termination
        elif termination_event:
            event_match = any(term in lowered for term in ("terminat", "dismiss", "end employment"))
        elif event:
            event_match = bool(event_tokens & tokens)
        else:
            event_match = "not_applicable"
        if probation_mismatch or permanent_mismatch or contractor_mismatch or distractor_mismatch:
            subject_match = False
        information_match = bool(information_tokens & tokens)
        if re.search(r"\b(?:time|limit|duration|minutes?|hours?)\b", information) and re.search(
            r"\b\d+\s*(?:minutes?|hours?)\b|\btimed\b|\btime limit\b", lowered
        ):
            information_match = True
        evidence_section = str(item.get("section", "")).lower()
        requirement_matches: list[str] = []
        for requirement in requirements:
            requirement_terms = {
                term for term in _text_tokens(requirement)
                if len(term) > 3 and term not in {"must", "evidence", "identify", "address", "state", "requested", "direct", "specific"}
            }
            def matches_requirement_term(term: str) -> bool:
                if term in lowered:
                    return True
                stem = re.sub(r"(tion|ment|ing|ed|s)$", "", term)
                if len(stem) >= 5 and stem in lowered:
                    return True
                normalized_roots = {
                    "implementation": "implement",
                    "deployment": "deploy",
                    "delivery": "deliver",
                    "requirements": "requirement",
                    "testing": "test",
                    "resurrecting": "resurrect",
                    "consequences": "consequence",
                }
                root = normalized_roots.get(term, term)
                return len(root) >= 5 and root in lowered

            hits = sum(1 for term in requirement_terms if matches_requirement_term(term))
            requirement_lower = requirement.lower()
            if (
                re.search(r"\b(?:appearing|appeared)\s+for\b", requirement_lower)
                and direct_speaker_match
                and re.search(r"\bfor\b", lowered)
            ):
                hits = max(hits, 1)
            if (
                re.search(r"\b(?:time|duration|minutes?|hours?)\b", requirement_lower)
                and information_match
            ):
                requested_role = requirement_roles.get(requirement)
                observed_role = temporal_role(text)
                if not requested_role or requested_role == observed_role:
                    hits = max(hits, 1)
                else:
                    hits = 0
            if hits >= 1:
                requirement_matches.append(requirement)
        if any(
            requirement in requirement_matches
            and re.search(r"\b(?:appearing|appeared)\s+for\b", requirement.lower())
            and direct_speaker_match
            and re.search(r"\bfor\b", lowered)
            for requirement in requirements
        ):
            # Short transcript turns such as "For the Union." carry the
            # answer through speaker provenance, even when they do not repeat
            # the planner's words "client" or "representation".
            information_match = True
        if requirement_matches and answer_strategy in {"aggregate_evidence", "compare_evidence", "explain_from_evidence"}:
            information_match = bool(requirement_matches)
        if any(term in information for term in ("name", "email", "phone")):
            information_match = information_match or (
                "email" in information and bool(re.search(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b", text))
            )
            if "phone" in information:
                information_match = information_match or bool(
                    re.search(r"(?<!\d)(?:\+?\d[\d\s\-()]{7,}\d)(?!\d)", text)
                )
            if "name" in information and question_type in {"attribute_lookup", "factual_extraction"}:
                first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
                header_name = re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}(?:\s|$)", first_line)
                information_match = information_match or bool(header_name)
            if "name" in information and location_match:
                target_entity = str(spec.get("target_entity") or spec.get("subject") or "")
                information_match = information_match or (
                    bool(_text_tokens(target_entity) & tokens)
                    and bool(re.search(r"\b[A-Z][A-Z'’. -]{2,}\b", text))
                )
        if any(term in information for term in ("notice", "period", "days", "month", "months")):
            information_match = information_match or any(
                term in lowered for term in ("notice", "period", "days", "month", "months")
            )
        if any(term in information for term in ("university", "college", "education")):
            information_match = information_match or any(term in lowered for term in ("university", "college", "education"))
        if any(term in information for term in ("degree", "qualification")):
            information_match = information_match or any(term in lowered for term in ("degree", "b.tech", "bachelor", "master", "qualification"))
        if "skill" in information:
            information_match = information_match or any(
                term in lowered or term in evidence_section
                for term in ("skill", "strength", "technology", "technical")
            )
        if any(term in information for term in ("employer", "company", "work")):
            information_match = information_match or any(term in lowered for term in ("company", "employer", "worked", "work", " at ", "employed"))
        target_attribute = str(spec.get("target_attribute") or "").lower()
        attribute_tokens = _text_tokens(target_attribute)
        attribute_match = bool(attribute_tokens & tokens)
        if question_type in {"attribute_lookup", "factual_extraction"} and target_attribute and attribute_match:
            # A direct attribute statement can identify the requested subject
            # even when the source sentence omits the organization/person name.
            # The attribute and its value must still pass information matching.
            subject_match = subject_match or information_match
        if question_type in {"attribute_lookup", "factual_extraction"} and required_speakers:
            # A speaker's short answer may omit their own name. Provenance
            # supplies the subject link; the text supplies the requested fact.
            if direct_speaker_match and information_match:
                subject_match = True
            if indirect_attribute_match and information_match:
                subject_match = True
        # In a speaker-attributed legal argument, the speaker provenance is
        # the subject link.  Legal reasoning often refers to the issue using
        # different words than the question (for example, a provision number
        # or "the rights claimed"), so requiring literal subject-token overlap
        # would discard the actual argument before the Solver sees it.
        legal_reasoning_signal = bool(re.search(
            r"\b(?:article|section|constitution|covenant|rights?|repealed|"
            r"attracted|mandamus|petitioners?|originated|provision)\b",
            lowered,
        ))
        if required_speakers and direct_speaker_match and legal_reasoning_signal:
            subject_match = True
        duration_role_matches = True
        duration_requirements = [
            requirement for requirement in requirements
            if re.search(r"\b(?:time|duration|minutes?|hours?)\b", requirement.lower())
        ]
        if duration_requirements:
            observed_role = temporal_role(text)
            assigned_duration_requested = any(
                requirement_roles.get(requirement) == "assigned"
                for requirement in duration_requirements
            )
            duration_evidence = bool(observed_role) or bool(re.search(
                r"\b\d+\s*(?:minutes?|hours?)\b|\btime limit\b|\btimed\b",
                lowered,
            ))
            # A self-estimate is not evidence for an externally assigned
            # limit.  Apply this even when the item failed lexical matching;
            # otherwise a generic duration fallback could re-admit it as
            # supporting evidence and confuse the Solver.
            if assigned_duration_requested and duration_evidence:
                duration_role_matches = observed_role == "assigned"
            else:
                duration_role_matches = any(
                    not requirement_roles.get(requirement)
                    or requirement_roles.get(requirement) == observed_role
                    for requirement in duration_requirements
                    if requirement in requirement_matches
                ) or not any(requirement in requirement_matches for requirement in duration_requirements)
        reasoning_support = bool(
            required_speakers
            and direct_speaker_match
            and legal_reasoning_signal
            and requirement_matches
        )
        # Speaker provenance boosts ranking, but is not a hard gate. A
        # non-transcript document must never be rejected because it has no
        # speaker turns, and transcript context remains available if a
        # validated speaker constraint is only a weak match.
        direct_support = duration_role_matches and subject_match and information_match and (
            (event_match if event else True) or indirect_attribute_match or reasoning_support
        )
        contradiction = bool(
            information_match and any(term in lowered for term in ("no notice", "without notice"))
        )
        relevant = direct_support and not contradiction and not negative_match
        relevance_type = "irrelevant"
        if relevant:
            relevance_type = "direct" if requirement_matches else "supporting"
        elif subject_match and information_match:
            relevance_type = "contextual"
        score = sum([
            4 if relevant else 0,
            2 if subject_match else 0,
            2 if event_match else 0,
            2 if information_match else 0,
            1 if direct_support else 0,
            3 if direct_speaker_match else (1 if indirect_attribute_match else 0),
            -3 if contradiction else 0,
            12 if location_match else (-2 if requested_location and item_page is not None else 0),
        ])
        evidence_id = f"E{len(verification) + 1}"
        if relevant:
            annotated_item = dict(item)
            observed_role = temporal_role(text)
            if observed_role:
                annotated_item["relation"] = "time_limit_assigned" if observed_role == "assigned" else "expected_duration"
                annotated_item["temporal_role"] = observed_role
                annotated_item["referenced_entity"] = str(spec.get("subject", ""))
            elif any(
                requirement in requirement_matches
                and re.search(r"\b(?:appearing|appeared)\s+for\b", requirement.lower())
                for requirement in requirements
            ):
                annotated_item["relation"] = "attribution"
                annotated_item["referenced_entity"] = str(spec.get("subject", ""))
            verified.append(annotated_item)
            for requirement in requirement_matches:
                evidence_coverage[requirement]["satisfied"] = True
                evidence_coverage[requirement]["evidence_ids"].append(evidence_id)
        verification.append({
            "evidence_id": evidence_id,
            "source_document": item.get("source_document"),
            "page": item.get("page"),
            "excerpt": item.get("excerpt", ""),
            "relevant": relevant,
            "subject_match": subject_match,
            "speaker": item.get("speaker"),
            "related_speakers": item.get("related_speakers", []),
            "referenced_entity": item.get("referenced_entity") or spec.get("subject"),
            "relation": (
                "time_limit_assigned" if temporal_role(text) == "assigned"
                else "expected_duration" if temporal_role(text) == "expected"
                else "attribution" if re.search(r"\b(?:appearing|appeared)\s+for\b", lowered)
                else item.get("relation")
            ),
            "temporal_role": temporal_role(text),
            "speaker_match": speaker_match,
            "event_match": event_match,
            "information_match": information_match,
            "constraint_match": direct_support,
            "direct_support": direct_support,
            "contradiction": contradiction,
            "negative_concept_match": negative_match,
            "relevance_type": relevance_type,
            "requirement_matches": requirement_matches,
            "relevance_score": score,
            "location_match": location_match,
            "reason": (
                "The passage directly supports the requested subject, event, and information."
                if relevant else
                "The passage is WRONG_CONTEXT because it is not attributed to the named speaker."
                if required_speakers and not speaker_match else
                "The passage is lexically related but does not directly support the requested subject and event."
            ),
        })
    contradictions: list[dict] = []
    conflict_trace: list[dict] = []
    ledger_by_description = {
        str(item.get("what", "")): item
        for item in state.get("requirement_ledger", [])
        if str(item.get("what", "")).strip()
    }
    for requirement in requirements:
        ledger_by_description.setdefault(requirement, {
            "requirement_id": f"R{requirements.index(requirement) + 1}",
            "what": requirement,
            "expected_form": "duration" if re.search(r"\b(?:notice|period|duration|minutes?|hours?|days?|months?)\b", requirement, re.I) else "free_text",
            "entities": [],
        })
    if spec.get("document_location"):
        verified.sort(key=lambda item: (
            1 if (
                (spec.get("document_location") == "first_page" and item.get("page") == 1)
                or (str(spec.get("document_location") or "").startswith("page_") and item.get("page") == spec.get("page_number"))
            ) else 0
        ), reverse=True)
    # Conflict detection is deliberately scoped to atomic, single-valued
    # requirements.  Complementary explanation passages are not competing
    # values and must remain available to the Solver for synthesis.
    for requirement, ledger_item in ledger_by_description.items():
        expected_form = str(ledger_item.get("expected_form", "free_text")).lower()
        if not _is_single_valued_requirement(ledger_item):
            conflict_trace.append({
                "requirement_id": ledger_item.get("requirement_id"),
                "requirement": requirement,
                "expected_form": expected_form,
                "skipped": True,
                "reason": "conflict_detection_skipped: requirement is not single-valued",
            })
            continue
        # Use verification requirement matches, rather than lexical overlap,
        # to ensure both values concern the same requested attribute.
        matching = [
            (
                entry.get("evidence_id"),
                next((item for item in verified if (
                    item.get("source_document") == entry.get("source_document")
                    and item.get("page") == entry.get("page")
                    and str(item.get("excerpt", "")) == str(entry.get("excerpt", ""))
                )), None),
            )
            for entry in verification
            if entry.get("relevant") and requirement in (entry.get("requirement_matches") or [])
        ]
        value_items = []
        for evidence_id, item in matching:
            if item is None:
                continue
            values = _conflict_values(expected_form, str(item.get("excerpt", "")))
            if values:
                value_items.append((str(evidence_id), item, values))
        distinct_values = {value for _, _, values in value_items for value in values}
        if len(distinct_values) > 1:
            conflict = {
                "type": "conflicting_single_valued_requirement",
                "requirement_id": ledger_item.get("requirement_id"),
                "requirement": requirement,
                "attribute": requirement,
                "entity": ledger_item.get("entities", []),
                "values": sorted(distinct_values),
                "evidence_ids": [item[0] for item in value_items],
                "sources": sorted({str(item[1].get("source_document")) for item in value_items}),
                "reason": "Different values were verified for the same single-valued requirement.",
                "rule": "single-valued requirement + same requirement/attribute + distinct verified values",
            }
            contradictions.append(conflict)
            conflict_trace.append({**conflict, "skipped": False})
    gaps: list[str] = []
    if not verified:
        gaps.append(
            "No verified evidence was found in the inspected windows; investigate broader sections, related concepts, or another candidate document."
        )
    for requirement, details in evidence_coverage.items():
        if not details.get("satisfied"):
            gaps.append(_specific_requirement_gap(state, requirement))
    ledger = []
    for requirement in state.get("requirement_ledger", []):
        description = str(requirement.get("what", ""))
        terms = {term for term in _text_tokens(description) if len(term) > 3}
        expected = str(requirement.get("expected_form", "free_text"))
        speaker_name = speaker_key(requirement.get("speaker_or_source", ""))

        def ledger_item_match(item: EvidenceItem) -> bool:
            text_value = str(item.get("excerpt", ""))
            lowered_value = text_value.lower()
            term_match = bool(terms & _text_tokens(text_value))
            provenance_match = bool(speaker_name and speaker_name == speaker_key(item.get("speaker", "")))
            attribution_fact = expected == "person_or_organization" and provenance_match and bool(re.search(r"\bfor\b|on behalf of|representing", lowered_value))
            duration_fact = expected == "duration_or_date" and _form_satisfied(expected, text_value, state.get("question", ""))
            return term_match or attribution_fact or duration_fact

        candidate_ids = [f"E{index + 1}" for index, item in enumerate(candidates) if ledger_item_match(item)]
        # Evidence IDs are assigned while walking candidates.  Re-numbering
        # the filtered `verified` list here can point the ledger at a
        # different passage (or lose a valid passage whose wording differs
        # from the planner description).  Coverage is the authoritative,
        # requirement-aware mapping produced above, so preserve those IDs.
        covered_ids = list(evidence_coverage.get(description, {}).get("evidence_ids", []))
        verified_ids = covered_ids or [f"E{index + 1}" for index, item in enumerate(verified) if ledger_item_match(item)]
        form_ok = any(_form_satisfied(expected, str(item.get("excerpt", "")), state.get("question", "")) or ledger_item_match(item) for item in verified)
        if verified_ids and form_ok:
            status = "SUPPORTED"
        elif verified_ids:
            status = "PARTIAL"
        elif candidate_ids:
            status = "CANDIDATES_FOUND"
        else:
            status = "UNSEARCHED" if not state.get("investigation_history") else "PARTIAL"
        gap = None if status == "SUPPORTED" else (
            "PARTIAL" if status == "PARTIAL" else "NO_CANDIDATES" if not candidate_ids else "WRONG_CONTEXT"
        )
        ledger.append({**requirement, "status": status, "candidate_evidence_ids": candidate_ids, "verified_evidence_ids": verified_ids, "gap": gap})
    assessments = dict(state.get("document_assessments", {}))
    for item in candidates:
        source = str(item.get("source_document", ""))
        if not source:
            continue
        assessment = dict(assessments.get(source, {}))
        source_verified = [e for e in verified if e.get("source_document") == source]
        assessment.update({
            "evidence_found": bool(source_verified),
            "document_answerability": "likely" if source_verified else "unknown",
            "requirements_covered": [
                requirement for requirement, details in evidence_coverage.items()
                if details.get("satisfied")
            ],
            "remaining_gaps": gaps,
        })
        assessments[source] = assessment
    document_investigation = dict(state.get("document_investigation") or {})
    documents_with_verified = set(state.get("documents_with_verified_evidence") or [])
    for source in state.get("selected_documents") or []:
        source_name = str(source)
        source_candidates = [item for item in candidates if str(item.get("source_document")) == source_name]
        source_verified = [item for item in verified if str(item.get("source_document")) == source_name]
        entry = dict(document_investigation.get(source_name) or {"document": source_name})
        if source_verified:
            entry["status"] = "VERIFIED"
            entry["verified_evidence_count"] = len(source_verified)
            documents_with_verified.add(source_name)
        elif source_candidates:
            entry["status"] = "EVIDENCE_FOUND"
        else:
            entry.setdefault("status", "UNSEARCHED")
        entry["candidate_evidence_count"] = len(source_candidates)
        entry["remaining_unresolved_requirements"] = [
            requirement for requirement, details in evidence_coverage.items()
            if not details.get("satisfied")
        ]
        document_investigation[source_name] = entry
    prior_verified = list(state.get("verified_evidence") or [])
    combined_verified = prior_verified + verified
    deduped_verified: list[EvidenceItem] = []
    seen_verified: set[tuple[object, ...]] = set()
    for item in combined_verified:
        identity = (item.get("source_document"), item.get("page"), item.get("sheet"), item.get("row"), item.get("excerpt"))
        if identity not in seen_verified:
            seen_verified.add(identity)
            deduped_verified.append(item)
    verified = deduped_verified
    return {
        "verified_evidence": verified,
        "requirement_ledger": ledger,
        "evidence_verification": verification,
        "evidence_coverage": evidence_coverage,
        "insufficient_evidence": not bool(verified),
        "investigation_gaps": gaps,
        "investigation_gap": gaps[0] if gaps else "",
        "document_assessments": assessments,
        "document_investigation": document_investigation,
        "documents_with_verified_evidence": sorted(documents_with_verified),
        "contradictions": contradictions,
        "investigation_action": "resolve_contradiction" if contradictions else "",
        "conflict_trace": conflict_trace,
        "attribution_context": {
            **(state.get("attribution_context") or {}),
            "verified_sources": sorted({str(item.get("source_document")) for item in verified}),
            "verified_pages": sorted({item.get("page") for item in verified if item.get("page") is not None}),
        },
        "investigation_trace": _trace(
            state,
            "evidence_verification",
            candidate_count=len(candidates),
            verified_count=len(verified),
            coverage=evidence_coverage,
            evidence_verdicts=verification[:80],
            contradictions=contradictions,
            conflict_flags=conflict_trace,
            gaps=gaps,
            document_investigation=document_investigation,
            constraints=[
                _speaker_constraint_diagnostic(state, str(source))
                for source in (state.get("selected_documents") or [])
            ],
        ),
    }


def assess_evidence_sufficiency(state: AgentState) -> dict[str, object]:
    spec = state.get("investigation_spec") or {}
    strategy = str(spec.get("answer_strategy", "direct"))
    coverage = state.get("evidence_coverage") or {}
    coverage_missing = (
        strategy in {"aggregate_evidence", "compare_evidence", "explain_from_evidence"}
        and bool(coverage)
        and any(not details.get("satisfied") for details in coverage.values())
    )
    hearing_missing_final_judgment = False
    if spec.get("intent") == "analyze_legal_case_record":
        # A hearing transcript may contain submissions but not a final judgment.
        # Preserve that limitation in the answer instead of discarding all useful material.
        coverage_missing = False
        asks_for_final_outcome = bool(re.search(
            r"final (?:conclusion|judgment|decision|order|outcome)|court'?s final",
            state["question"],
            flags=re.IGNORECASE,
        ))
        inspected = state.get("inspected_documents", [])
        hearing_only = bool(inspected) and all(
            document.get("metadata", {}).get("record_kind") == "hearing_transcript"
            for document in inspected
        )
        if asks_for_final_outcome and hearing_only:
            hearing_missing_final_judgment = True
    exhaustion = dict(state.get("exhaustion_certificate") or {})
    exhausted = bool(exhaustion.get("complete")) and bool(exhaustion.get("coverage_complete", True))
    # A failed targeted window is not proof that the fact is absent.  The
    # graph may only make an insufficiency decision after the bounded fallback
    # has completed (or when a real contradiction exists).
    # Verified evidence is enough to let the Solver provide a complete or
    # explicitly partial answer.  Missing complementary requirements must not
    # turn an explanatory answer into a blanket refusal, and conflicts are
    # passed through as competing claims for the Solver to present.
    insufficient = exhausted and not bool(state.get("verified_evidence", []))
    if hearing_missing_final_judgment and exhausted:
        insufficient = True
    gaps = list(state.get("investigation_gaps", []))
    if coverage_missing and not gaps:
        gaps = [
            _specific_requirement_gap(state, requirement)
            for requirement, details in coverage.items()
            if not details.get("satisfied")
        ]
    if insufficient and not gaps:
        gaps = [
            "Insufficient evidence after bounded investigation."
            if exhausted else
            "Evidence is not yet sufficient; further bounded investigation is required."
        ]
    if not insufficient and not exhausted and not state.get("verified_evidence"):
        gaps = gaps or ["No verified evidence yet; continue through the remaining search paths."]
    if exhausted and not state.get("verified_evidence"):
        exhaustion["status"] = "EXHAUSTED_ABSENT"
    ledger = []
    for item in state.get("requirement_ledger", []):
        requirement = str(item.get("what", ""))
        coverage_detail = coverage.get(requirement, {})
        updated = dict(item)
        if exhausted and not coverage_detail.get("satisfied"):
            updated["status"] = "EXHAUSTED_ABSENT"
            updated["gap"] = "No supporting evidence after all bounded search paths."
        elif coverage_detail.get("satisfied"):
            updated["status"] = "SUPPORTED"
        ledger.append(updated)
    return {
        "insufficient_evidence": insufficient,
        "investigation_gaps": gaps,
        "investigation_gap": gaps[0] if gaps else "",
        "investigation_action": (
            "hearing_transcript_missing_final_judgment" if hearing_missing_final_judgment and exhausted
            else "broaden_search" if not exhausted and (coverage_missing or not state.get("verified_evidence"))
            else state.get("investigation_action", "")
        ),
        "exhaustion_certificate": exhaustion,
        "requirement_ledger": ledger,
        "investigation_trace": _trace(
            state,
            "evidence_sufficiency",
            sufficient=not insufficient,
            gaps=gaps,
            retry_count=state.get("retry_count", 0),
            exhaustion_certificate=exhaustion,
        ),
    }


def bounded_exhaustive_search(state: AgentState) -> dict[str, object]:
    """Inspect remaining document regions in bounded deterministic batches.

    This is deliberately a last-mile local inspection path, not a giant LLM
    context and not a vector retrieval system.  It gives the graph an
    auditable distinction between ``not found in the current window`` and
    ``not found after every allowed region was searched``.
    """

    if state.get("exhaustive_search_attempted"):
        return {"exhaustion_certificate": dict(state.get("exhaustion_certificate") or {})}

    spec = state.get("investigation_spec") or _fallback_investigation_spec(state["question"])
    query = _spec_search_text(state)
    batch_size = max(1, int(os.getenv("EXHAUSTIVE_BATCH_PAGES", "12")))
    max_pages = max(1, int(os.getenv("MAX_EXHAUSTIVE_PAGES", "240")))
    metadata_by_name = {str(item.get("name")): item for item in state.get("available_documents", [])}
    existing = list(state.get("inspected_documents") or [])
    regions: list[dict[str, object]] = []
    pages_processed = 0
    coverage_complete = True
    paths = list(dict.fromkeys([*(state.get("search_paths_attempted") or []), "bounded_exhaustive_batches"]))

    # Build independent per-document task queues, then consume them in
    # round-robin order.  One large source therefore cannot consume the whole
    # corpus budget before a small selected source receives an opportunity.
    task_queues: dict[str, list[tuple[dict[str, object], dict[str, object], int]]] = {}
    for source in state.get("selected_documents") or []:
        metadata = metadata_by_name.get(str(source))
        if not metadata:
            continue
        path = str(metadata.get("path", ""))
        page_count = int(metadata.get("pages") or 0)
        tasks: list[tuple[dict[str, object], dict[str, object], int]] = []
        if page_count:
            for start in range(1, page_count + 1, batch_size):
                end = min(start + batch_size - 1, page_count)
                tasks.append((
                    {"path": path, "query": query, "window_size": 1,
                     "exact_phrase": str(spec.get("exact_phrase") or ""),
                     "include_body": True,
                     "page_start": start, "page_end": end},
                    {"source": source, "page_start": start, "page_end": end},
                    end - start + 1,
                ))
        else:
            sections = (state.get("document_maps") or {}).get(str(source), {}).get("sections", [])
            for section in sections:
                payload: dict[str, object] = {"path": path, "query": query, "window_size": 1,
                                               "exact_phrase": str(spec.get("exact_phrase") or "")}
                for key, value in (("line_start", section.get("line_start")), ("line_end", section.get("line_end")),
                                   ("paragraph_start", section.get("paragraph_start")), ("paragraph_end", section.get("paragraph_end")),
                                   ("row_start", section.get("row_start")), ("row_end", section.get("row_end")),
                                   ("sheet", section.get("sheet"))):
                    if value is not None:
                        payload[key] = value
                tasks.append((payload, {"source": source, "section": section.get("label") or section.get("section")}, 1))
        task_queues[str(source)] = tasks

    while pages_processed < max_pages and any(task_queues.values()):
        progressed = False
        for source in list(task_queues):
            if not task_queues[source] or pages_processed >= max_pages:
                continue
            payload, region, cost = task_queues[source].pop(0)
            if pages_processed + cost > max_pages:
                coverage_complete = False
                task_queues[source].insert(0, (payload, region, cost))
                continue
            result = inspect_document_window_tool.invoke(payload)
            result.setdefault("metadata", {})["search_path"] = "bounded_exhaustive_batches"
            result["metadata"]["region_identity"] = dict(region)
            if result.get("metadata", {}).get("sweep_read_mostly_metadata"):
                result["metadata"]["warning"] = "sweep_read_mostly_metadata"
            existing.append(result)
            regions.append(region)
            pages_processed += cost
            progressed = True
        if not progressed:
            break
    if any(task_queues.values()):
        coverage_complete = False

    documents_exhausted = sorted(source for source, tasks in task_queues.items() if not tasks)
    document_investigation = dict(state.get("document_investigation") or {})
    for source in state.get("selected_documents") or []:
        entry = dict(document_investigation.get(str(source)) or {"document": str(source)})
        entry["pages_or_regions_budgeted"] = sum(region.get("page_end", region.get("page_start", 1)) - region.get("page_start", 1) + 1 if region.get("page_start") is not None else 1 for region in regions if region.get("source") == source)
        entry["regions_inspected"] = [region for region in regions if region.get("source") == source]
        entry["exhausted"] = str(source) in documents_exhausted
        if entry["exhausted"] and entry.get("status") not in {"VERIFIED", "EVIDENCE_FOUND"}:
            entry["status"] = "EXHAUSTED_ABSENT"
        document_investigation[str(source)] = entry

    certificate = {
        "complete": True,
        "coverage_complete": coverage_complete,
        "status": "EXHAUSTIVE_SEARCH_COMPLETE" if coverage_complete else "BOUNDED_SEARCH_LIMIT_REACHED",
        "paths_attempted": paths,
        "regions_inspected": regions[-500:],
        "pages_processed": pages_processed,
        "max_pages": max_pages,
        "batch_size": batch_size,
        "selected_sources": list(state.get("selected_documents") or []),
        "documents_exhausted": documents_exhausted,
        "unresolved_requirements": [
            str(requirement) for requirement, details in (state.get("evidence_coverage") or {}).items()
            if not details.get("satisfied")
        ],
    }
    return {
        "inspected_documents": existing,
        "exhaustion_certificate": certificate,
        "exhaustive_search_attempted": True,
        "document_investigation": document_investigation,
        "documents_exhausted": documents_exhausted,
        "search_paths_attempted": paths,
        "investigation_trace": _trace(
            state,
            "bounded_exhaustive_search",
            candidate_paths=paths,
            ladder_rung=5,
            selected_sources=list(state.get("selected_documents") or []),
            regions_inspected=regions[-200:],
            pages_processed=pages_processed,
            exhaustion_certificate=certificate,
            document_investigation=document_investigation,
            sweep_warnings=sorted({
                str(result.get("metadata", {}).get("warning"))
                for result in existing
                if result.get("metadata", {}).get("warning")
            }),
        ),
    }


def rank_evidence(state: AgentState) -> dict[str, list[EvidenceItem]]:
    """Rank verified evidence before applying the global context budget."""

    scores = {
        (item.get("source_document"), item.get("page")): int(item.get("relevance_score", 0))
        for item in state.get("evidence_verification", [])
        if item.get("relevant")
    }
    ranked = sorted(
        state.get("verified_evidence", []),
        key=lambda item: scores.get((item.get("source_document"), item.get("page")), 0),
        reverse=True,
    )
    return {"verified_evidence": ranked}


def _evidence_terms(question: str) -> set[str]:
    return {
        term.lower()
        for term in re.findall(r"[a-zA-Z]+", question)
        if len(term) > 3 and term.lower() not in {
            "what", "which", "when", "where", "who", "does", "the", "are",
            "was", "were", "this", "that", "these", "those", "about", "from",
            "with", "into", "have", "has", "how", "can", "may", "any", "all",
        }
    }


def apply_evidence_budget(
    question: str,
    evidence: list[EvidenceItem],
    max_items: int = MAX_EVIDENCE_ITEMS,
    max_chars: int = MAX_EVIDENCE_CHARS,
    max_per_source: int = MAX_EVIDENCE_PER_SOURCE,
) -> list[EvidenceItem]:
    """Rank and bound evidence while preserving each item's provenance."""

    if max_items <= 0 or max_chars <= 0:
        return []
    terms = _evidence_terms(question)
    deduplicated: list[EvidenceItem] = []
    seen: set[tuple[object, ...]] = set()
    for item in evidence:
        identity = (
            item.get("source_document"), item.get("page"), item.get("sheet"),
            item.get("row"), item.get("line"), item.get("paragraph"), item.get("excerpt"),
        )
        if identity in seen:
            continue
        seen.add(identity)
        deduplicated.append(item)
    ranked = []
    for index, item in enumerate(deduplicated):
        text = str(item.get("excerpt", ""))
        lowered = text.lower()
        score = sum(1 for term in terms if term in lowered)
        ranked.append((score, -index, item))
    ranked.sort(key=lambda value: (value[0], value[1]), reverse=True)

    selected: list[EvidenceItem] = []
    selected_ids: set[int] = set()
    source_counts: dict[str, int] = {}
    used_chars = 0

    def add_item(item: EvidenceItem) -> bool:
        nonlocal used_chars
        if len(selected) >= max_items:
            return False
        source = str(item.get("source_document", "unknown"))
        if source_counts.get(source, 0) >= max_per_source:
            return False
        remaining = max_chars - used_chars
        if remaining <= 0:
            return False
        excerpt = str(item.get("excerpt", ""))
        if not excerpt:
            return False
        if len(excerpt) > remaining:
            if remaining <= 2:
                excerpt = excerpt[:remaining]
            else:
                excerpt = excerpt[: remaining - 2].rsplit(" ", 1)[0].rstrip() + " …"
                excerpt = excerpt[:remaining]
        bounded_item = dict(item)
        bounded_item["excerpt"] = excerpt
        selected.append(bounded_item)
        source_counts[source] = source_counts.get(source, 0) + 1
        used_chars += len(excerpt)
        return True

    # Reserve one highest-ranked item per source first.  This is a bounded
    # fairness guarantee, not a relevance override: the remaining budget is
    # still filled by the global relevance order.
    for index, (_, _, item) in enumerate(ranked):
        source = str(item.get("source_document", "unknown"))
        if source not in source_counts and add_item(item):
            selected_ids.add(index)

    for index, (_, _, item) in enumerate(ranked):
        if index in selected_ids:
            continue
        if len(selected) >= max_items:
            break
        if not add_item(item):
            if used_chars >= max_chars:
                break
    return selected


def budget_evidence(state: AgentState) -> dict[str, object]:
    """Create the bounded evidence context used by both Solver and Auditor."""

    budgeted = apply_evidence_budget(state["question"], state.get("verified_evidence", []))
    evidence_chars = sum(len(str(item.get("excerpt", ""))) for item in budgeted)
    return {
        "budgeted_evidence": budgeted,
        "evidence_budget": {
            "max_items": MAX_EVIDENCE_ITEMS,
            "max_chars": MAX_EVIDENCE_CHARS,
            "max_per_source": MAX_EVIDENCE_PER_SOURCE,
            "selected_items": len(budgeted),
            "selected_chars": sum(len(str(item.get("excerpt", ""))) for item in budgeted),
        },
        "corpus_metrics": {
            **(state.get("corpus_metrics") or {}),
            "candidate_evidence_count": len(state.get("candidate_evidence", [])),
            "verified_evidence_count": len(state.get("verified_evidence", [])),
            "evidence_sent_to_solver_chars": evidence_chars,
            "evidence_sent_to_auditor_chars": evidence_chars,
            "final_evidence_size": evidence_chars,
        },
        "investigation_trace": _trace(
            state,
            "evidence_budget",
            selected_sources=list(state.get("selected_documents") or []),
            evidence_used_sources=sorted({str(item.get("source_document")) for item in budgeted}),
            verified_count=len(state.get("verified_evidence", [])),
            budgeted_count=len(budgeted),
            evidence_chars=evidence_chars,
            max_items=MAX_EVIDENCE_ITEMS,
            max_chars=MAX_EVIDENCE_CHARS,
            max_per_source=MAX_EVIDENCE_PER_SOURCE,
        ),
    }


def _format_evidence(evidence: list[EvidenceItem]) -> str:
    """Format evidence for an LLM without dropping provenance."""

    formatted = []
    for item in evidence:
        locations = []
        for label, key in (("PAGE", "page"), ("SHEET", "sheet"), ("ROW", "row"),
                           ("LINE", "line"), ("PARAGRAPH", "paragraph"), ("SECTION", "section")):
            if item.get(key) is not None:
                locations.append(f"{label}: {item[key]}")
        location_text = f" ({', '.join(locations)})" if locations else ""
        role_parts = []
        for label, key in (("SUBJECT", "subject_entity"), ("REFERENCED_ENTITY", "referenced_entity"),
                           ("RELATION", "relation"), ("TEMPORAL_ROLE", "temporal_role"),
                           ("SPEAKER", "speaker")):
            if item.get(key):
                role_parts.append(f"{label}: {item[key]}")
        role_text = f"\nROLE METADATA: {', '.join(role_parts)}" if role_parts else ""
        formatted.append(f"SOURCE: {item['source_document']}{location_text}{role_text}\n{item['excerpt']}")
    return "\n\n".join(formatted)


def _is_generic_insufficient_answer(answer: str) -> bool:
    """Identify a model fallback that is unsafe when evidence is available."""

    normalized = " ".join(answer.lower().split())
    return normalized.startswith((
        "insufficient evidence",
        "insufficient answer",
        "not enough evidence",
        "unable to determine",
    ))


def _evidence_backed_fallback(evidence: list[EvidenceItem]) -> str:
    """Return a concise, traceable fallback if the solver refuses usable evidence."""

    if not evidence:
        return "Insufficient evidence in the selected sources to answer without guessing."
    item = evidence[0]
    excerpt = " ".join(str(item.get("excerpt", "")).split())
    if len(excerpt) > 520:
        excerpt = excerpt[:517].rsplit(" ", 1)[0] + "…"
    return f"Evidence was found, but the Solver could not produce a written answer. Quote: {excerpt} [{item.get('source_document', 'unknown source')}]"


def _service_failure_answer(failure: LLMCallFailure, evidence: list[EvidenceItem]) -> str:
    pages = sorted({item.get("page") for item in evidence if item.get("page") is not None})
    location = f" Evidence was found on pages {', '.join(map(str, pages))}." if pages else " Evidence was found in the selected source."
    return f"The model service failed during the Solver call ({failure.status}: {failure.message}).{location} The result below is not a generated answer."


def _redact_response_snippet(value: str) -> str:
    return re.sub(r"(?i)(api[_ -]?key|authorization|bearer)\s*[:=]\s*[^\s,}]+", r"\1=[REDACTED]", value[:300])


def _structured_list_fallback(evidence: list[EvidenceItem]) -> str | None:
    """Recover an explicitly numbered list when an LLM misattributes it."""

    for item in evidence:
        text = " ".join(str(item.get("excerpt", "")).split())
        if not re.search(
            r"\b(?:formulated|identified|listed|following|these)\b.{0,100}\b(?:four|three|five)\s+issues\b"
            r"|\b(?:four|three|five)\s+issues\b",
            text,
            re.I,
        ):
            continue
        start = re.search(
            r"\b(?:Article|First(?:ly)?|The first|Number\s+1|1[.)])\b",
            text,
            re.I,
        )
        if not start:
            continue
        segment = text[start.start():]
        pieces = re.split(
            r"(?=\b(?:Number\s+\d+|Second(?:ly)?|Third(?:ly)?|Fourth(?:ly)?|Fifth(?:ly)?|\d+[.)])[,:]?\s*)",
            segment,
            flags=re.I,
        )
        pieces = [piece.strip(" .") for piece in pieces if piece.strip()]
        if len(pieces) < 2:
            continue
        source = item.get("source_document", "unknown source")
        location = f" page {item['page']}" if item.get("page") is not None else ""
        return "The source explicitly lists the issues as:\n" + "\n".join(
            f"- {piece} [{source}{location}]" for piece in pieces[:5]
        )
    return None


def _claim_support(answer: str, evidence: list[EvidenceItem]) -> list[dict]:
    """Create a small deterministic claim-to-evidence index for the Auditor."""

    claims: list[dict] = []
    # Protect initials and dotted abbreviations such as K.M. while splitting
    # prose into claims. A period inside an abbreviation is not a claim end.
    protected = re.sub(
        r"\b(?:[A-Z]\.){2,}",
        lambda match: match.group(0).replace(".", "<DOT>"),
        str(answer),
    )
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", protected):
        sentence = sentence.replace("<DOT>", ".")
        claim = " ".join(sentence.split()).strip(" -")
        if not claim or _is_generic_insufficient_answer(claim):
            continue
        cited = re.findall(r"\[([^\]]+)\]", claim)
        claim_tokens = {
            token for token in _text_tokens(claim)
            if len(token) > 3 and token not in {"that", "this", "with", "from", "provided", "based"}
        }
        support: list[str] = []
        for index, item in enumerate(evidence, start=1):
            excerpt_tokens = _text_tokens(str(item.get("excerpt", "")))
            source = str(item.get("source_document", ""))
            overlap = len(claim_tokens & excerpt_tokens)
            relation = str(item.get("relation", "")).lower()
            attribution_claim = bool(re.search(
                r"\b(?:appeared|appearing|behalf|represent|counsel|for)\b", claim.lower()
            ))
            assigned_time_claim = bool(re.search(
                r"\b(?:given|allowed|timed|assigned|required|limit)\b.*\b\d+\s*(?:minutes?|hours?)\b",
                claim.lower(),
            ))
            relation_support = (
                relation == "attribution" and attribution_claim
            ) or (
                relation == "time_limit_assigned" and assigned_time_claim
            )
            if source in cited or overlap >= 2 or relation_support:
                support.append(f"E{index}")
        claims.append({"claim": claim, "evidence_ids": list(dict.fromkeys(support))})
    return claims


def _parse_solver_payload(text: str) -> tuple[str, list[dict], list[str]]:
    """Accept the structured Solver contract while preserving a safe text fallback."""

    cleaned = str(text).strip()
    # Models frequently wrap JSON in Markdown fences.  Remove only the fence
    # markers; keep the payload unchanged for the strict decoder below.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        # A provider may truncate a response after answer_text.  Salvage the
        # answer value, but never expose the raw JSON envelope as the answer.
        answer_match = re.search(r'"answer_text"\s*:\s*"((?:\\.|[^"\\])*)', cleaned)
        if answer_match:
            raw_value = answer_match.group(1)
            try:
                return json.loads('"' + raw_value + '"').strip(), [], []
            except json.JSONDecodeError:
                return raw_value.replace('\\"', '"').replace('\\n', ' ').strip(), [], []
        return ("The model returned an incomplete structured response. Please try again."
                if cleaned.startswith("{") or cleaned.startswith("```") else cleaned), [], []
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        answer_match = re.search(r'"answer_text"\s*:\s*"((?:\\.|[^"\\])*)', cleaned)
        if answer_match:
            raw_value = answer_match.group(1)
            try:
                return json.loads('"' + raw_value + '"').strip(), [], []
            except json.JSONDecodeError:
                return raw_value.replace('\\"', '"').replace('\\n', ' ').strip(), [], []
        return "The model returned an invalid structured response. Please try again.", [], []
    if not isinstance(payload, dict) or not isinstance(payload.get("answer_text"), str):
        return "The model returned an invalid structured response. Please try again.", [], []
    claims = []
    for index, claim in enumerate(payload.get("claims", []), start=1):
        if not isinstance(claim, dict) or not str(claim.get("text", "")).strip():
            continue
        claims.append({
            "claim_id": str(claim.get("claim_id") or f"C{index}"),
            "claim": str(claim.get("text")),
            "requirement_id": claim.get("requirement_id"),
            "evidence_ids": [str(value) for value in claim.get("evidence_ids", []) if str(value).strip()],
            "type": str(claim.get("type") or "inferred"),
        })
    unanswered = [str(value) for value in payload.get("unanswered_requirements", []) if str(value).strip()]
    return payload["answer_text"].strip(), claims, unanswered


def _solver_response_needs_repair(raw: str, answer: str) -> bool:
    """Detect a provider-cut structured response before treating it as an answer."""

    text = str(raw or "").strip()
    if not text:
        return True
    # A structured response that has no closing object was cut before parsing.
    if text.startswith("{") or text.startswith("```"):
        if not re.search(r"\}\s*`?`?`?$", text):
            return True
    # Salvaged answer_text values ending in a connector are usually the visible
    # prefix of a provider response stopped by the output budget.
    return bool(re.search(
        r"\b(?:of|for|and|or|the|a|an|to|in|on|with|supporting|including|is|was|were|that)\s*$",
        str(answer).strip(),
        re.IGNORECASE,
    ))


def _candidate_score(result: dict[str, object]) -> tuple[int, float, int]:
    verdicts = result.get("claim_verdicts") or []
    supported = sum(1 for item in verdicts if item.get("verdict") in {"ENTAILED", "SUPPORTED"})
    confidence = result.get("confidence")
    return (
        1 if result.get("approved") else 0,
        float(confidence) if isinstance(confidence, (int, float)) else 0.0,
        supported,
    )


def _claim_verdict_rows(state: AgentState, evidence: list[EvidenceItem]) -> list[dict[str, object]]:
    """Perform the deterministic part of auditing against complete cited quotes."""

    evidence_by_id = {f"E{index + 1}": item for index, item in enumerate(evidence)}
    ledger = {
        str(item.get("requirement_id")): item
        for item in state.get("requirement_ledger", [])
        if item.get("requirement_id")
    }
    rows: list[dict[str, object]] = []
    for claim in state.get("solver_claims", []):
        claim_text = str(claim.get("claim") or claim.get("text") or "").strip()
        cited_ids = [str(value) for value in claim.get("evidence_ids", []) if str(value).strip()]
        """
        cited_items = [evidence_by_id[item_id] for item_id in cited_ids if item_id in evidence_by_id]\n+        quotes = [str(item.get("excerpt", "")) for item in cited_items]\n+        normalized_claim = " ".join(claim_text.lower().split())\n+        exact = any(\n+            normalized_claim and normalized_claim in \" \".join(quote.lower().split())\n+            for quote in quotes\n+        )\n+        fuzzy = any(\n+            SequenceMatcher(None, normalized_claim, \" \".join(quote.lower().split())).ratio() >= 0.72\n+            for quote in quotes\n+        )\n+        if not cited_items:\n+            verdict = \"NOT_SUPPORTED\"\n+            reason = \"The claim cites no evidence ID present in the bounded verified evidence.\"\n+        elif exact or fuzzy:\n+            verdict = \"ENTAILED\"\n+            reason = \"The claim is supported by a near-verbatim cited evidence span.\"\n+        elif str(claim.get(\"type\", \"inferred\")).lower() in {\"inferred\", \"conflict\"}:\n+            verdict = \"PARTIALLY_ENTAILED\"\n+            reason = \"The claim is an inference or comparison grounded in its cited evidence spans.\"\n+        else:\n+            verdict = \"PARTIALLY_ENTAILED\"\n+            reason = \"The cited evidence supports part of the claim; paraphrase and connective wording are allowed.\"\n+        requirement = ledger.get(str(claim.get(\"requirement_id\")), {})
        rows.append({\n+            \"claim_id\": claim.get(\"claim_id\"),\n+            \"claim_text\": claim_text,\n+            \"evidence_ids\": cited_ids,\n+            \"cited_quotes\": [\n+                {\"evidence_id\": item_id, \"quote\": str(evidence_by_id[item_id].get(\"excerpt\", \"\")),\n+                 \"source\": evidence_by_id[item_id].get(\"source_document\"), \"page\": evidence_by_id[item_id].get(\"page\")}\n+                for item_id in cited_ids if item_id in evidence_by_id\n+            ],\n+            \"claim_type\": claim.get(\"type\", \"inferred\"),\n+            \"requirement_id\": claim.get(\"requirement_id\"),\n+            \"core\": bool(requirement),\n+            \"verdict\": verdict,\n+            \"reason\": reason,\n+        })\n+    return rows\n+\n+\n+def _claim_confidence(rows: list[dict[str, object]]) -> float | None:\n+    if not rows:\n+        return None\n+    entailed = sum(1 for row in rows if row.get(\"verdict\") == \"ENTAILED\")\n+    partial = sum(1 for row in rows if row.get(\"verdict\") == \"PARTIALLY_ENTAILED\")\n+    return round((entailed + 0.5 * partial) / len(rows), 2)\n+\n+\n def _preserve_best_candidate(state: AgentState, result: dict[str, object]) -> dict[str, object]:
"""

def _claim_verdict_rows(state: AgentState, evidence: list[EvidenceItem]) -> list[dict[str, object]]:
    """Perform deterministic pre-audit against complete cited quotes."""
    evidence_by_id = {f"E{index + 1}": item for index, item in enumerate(evidence)}
    ledger = {
        str(item.get("requirement_id")): item
        for item in state.get("requirement_ledger", [])
        if item.get("requirement_id")
    }
    rows: list[dict[str, object]] = []
    for claim in state.get("solver_claims", []):
        claim_text = str(claim.get("claim") or claim.get("text") or "").strip()
        cited_ids = [str(value) for value in claim.get("evidence_ids", []) if str(value).strip()]
        cited_items = [evidence_by_id[item_id] for item_id in cited_ids if item_id in evidence_by_id]
        quotes = [str(item.get("excerpt", "")) for item in cited_items]
        normalized_claim = " ".join(claim_text.lower().split())
        exact = any(
            normalized_claim and normalized_claim in " ".join(quote.lower().split())
            for quote in quotes
        )
        fuzzy = any(
            SequenceMatcher(None, normalized_claim, " ".join(quote.lower().split())).ratio() >= 0.72
            for quote in quotes
        )
        if not cited_items:
            verdict = "NOT_SUPPORTED"
            reason = "The claim cites no evidence ID present in the bounded verified evidence."
        elif exact or fuzzy:
            verdict = "ENTAILED"
            reason = "The claim is supported by a near-verbatim cited evidence span."
        elif str(claim.get("type", "inferred")).lower() in {"inferred", "conflict"}:
            verdict = "PARTIALLY_ENTAILED"
            reason = "The claim is an inference or comparison grounded in its cited evidence spans."
        else:
            verdict = "PARTIALLY_ENTAILED"
            reason = "The cited evidence supports part of the claim; paraphrase and connective wording are allowed."
        requirement = ledger.get(str(claim.get("requirement_id")), {})
        rows.append({
            "claim_id": claim.get("claim_id"),
            "claim_text": claim_text,
            "evidence_ids": cited_ids,
            "cited_quotes": [
                {
                    "evidence_id": item_id,
                    "quote": str(evidence_by_id[item_id].get("excerpt", "")),
                    "source": evidence_by_id[item_id].get("source_document"),
                    "page": evidence_by_id[item_id].get("page"),
                }
                for item_id in cited_ids if item_id in evidence_by_id
            ],
            "claim_type": claim.get("type", "inferred"),
            "requirement_id": claim.get("requirement_id"),
            "core": bool(requirement),
            "verdict": verdict,
            "reason": reason,
        })
    return rows


def _claim_confidence(rows: list[dict[str, object]]) -> float | None:
    if not rows:
        return None
    entailed = sum(1 for row in rows if row.get("verdict") == "ENTAILED")
    partial = sum(1 for row in rows if row.get("verdict") == "PARTIALLY_ENTAILED")
    return round((entailed + 0.5 * partial) / len(rows), 2)


def _preserve_best_candidate(state: AgentState, result: dict[str, object]) -> dict[str, object]:
    """Keep a better earlier candidate from being overwritten by a retry."""

    current = dict(result)
    current["solver_answer"] = state.get("solver_answer", "")
    current["claim_verdicts"] = result.get("claim_verdicts", [])
    prior = {
        "approved": state.get("best_audit_status") == "approved",
        "confidence": state.get("best_confidence"),
        "claim_verdicts": state.get("best_solver_claims", []),
    }
    if state.get("best_solver_answer") and _candidate_score(prior) > _candidate_score(current):
        return {
            "best_solver_answer": state.get("best_solver_answer", ""),
            "best_solver_claims": state.get("best_solver_claims", []),
            "best_audit_status": state.get("best_audit_status", ""),
            "best_confidence": state.get("best_confidence"),
            "best_audit_reason": state.get("best_audit_reason", ""),
        }
    return {
        "best_solver_answer": state.get("solver_answer", ""),
        "best_solver_claims": state.get("solver_claims", []),
        "best_audit_status": result.get("audit_status", ""),
        "best_confidence": result.get("confidence"),
        "best_audit_reason": result.get("audit_reason", ""),
    }


def _complete_truncated_direct_answer(
    answer: str,
    evidence: list[EvidenceItem],
    spec: dict[str, object],
) -> str:
    """Complete a provider-truncated direct entity answer from verified text.

    This is deliberately generic: it only acts when the planner requires a
    direct name identification and the answer ends at an initial/short token.
    The completion must come from a verified evidence excerpt.
    """
    if not spec.get("direct_identification_required") or str(spec.get("target_attribute", "")).lower() != "name":
        return answer
    if not re.search(r"(?:\b[A-Z]\.?\s*){1,3}$", answer.strip()):
        return answer
    entity = str(spec.get("target_entity") or spec.get("subject") or "").strip()
    if not entity:
        return answer
    entity_pattern = re.compile(re.escape(entity) + r"\s+(.{2,100})", re.IGNORECASE)
    suffix = re.search(r"((?:\b[A-Z]\.?\s*){1,3})$", answer.strip())
    if not suffix:
        return answer
    short = re.sub(r"[^a-z]", "", suffix.group(1).lower())
    for item in evidence:
        excerpt = " ".join(str(item.get("excerpt", "")).split())
        match = entity_pattern.search(excerpt)
        if not match:
            continue
        tail = match.group(1)
        candidate_match = re.match(r"((?:[A-Z][A-Z.'’\-]*\s+){1,5}[A-Z][A-Za-z'’\-]+)", tail)
        if not candidate_match:
            continue
        candidate = candidate_match.group(1).strip(" .,:;\n")
        first = re.sub(r"[^a-z]", "", candidate.split()[0].lower())
        if not first or not (first.startswith(short) or short.startswith(first)):
            continue
        if "." in suffix.group(1):
            parts = candidate.split()
            if len(parts[0]) >= 2 and parts[0].isalpha():
                parts[0] = ".".join(parts[0]) + "."
            candidate = " ".join([parts[0]] + [part.title() if part.isupper() else part for part in parts[1:]])
        return answer[:suffix.start()].rstrip() + " " + candidate
    return answer


def _deterministic_audit(state: AgentState, evidence: list[EvidenceItem]) -> dict[str, object]:
    """Safety-check a solver result without pretending to be an LLM judgment."""

    answer = str(state.get("solver_answer", "")).strip()
    claims = list(state.get("solver_claims", []))
    claim_results = _claim_verdict_rows(state, evidence)
    hard_failures = [
        row for row in claim_results
        if not row.get("evidence_ids")
        or (row.get("verdict") in {"NOT_SUPPORTED", "WRONG_ATTRIBUTION"} and row.get("core"))
    ]
    valid = bool(answer) and not state.get("solver_failed", False) and not hard_failures
    if not claims:
        # A deterministic audit can approve a refusal only when the Solver
        # supplied an explicit, auditable refusal trail.  A non-empty answer
        # with no claims is not enough to establish support, especially when
        # verified evidence exists.
        certificate = state.get("exhaustion_certificate") or {}
        valid = bool(
            _is_generic_insufficient_answer(answer)
            and not evidence
            and certificate.get("complete")
            and state.get("coverage_complete")
        )
    return {
        "approved": valid,
        "audit_status": "fallback_validated" if valid else "unavailable",
        "audit_unavailable": True,
        "confidence": _claim_confidence(claim_results),
        "audit_reason": "Auditor unavailable; deterministic evidence-ID and coverage check only.",
        "audit_issues": [] if valid else ["Deterministic fallback could not establish complete claim support."],
        "claim_verdicts": claim_results,
        "investigation_action": "",
        "audit_feedback": "Auditor transport/validation failed; no investigation retry was consumed.",
    }


def _response_finish_reason(response: object) -> str:
    metadata = getattr(response, "response_metadata", {}) or {}
    value = metadata.get("finish_reason") or metadata.get("finishReason") or metadata.get("finish_reason_name")
    return str(value or "").upper()


def _response_was_truncated(response: object) -> bool:
    reason = _response_finish_reason(response)
    return reason in {"MAX_TOKENS", "LENGTH", "MAX_OUTPUT_TOKENS"} or "MAX_TOKENS" in reason


def solve_question(state: AgentState) -> dict[str, object]:
    """Answer using only extracted, source-labelled evidence."""

    source_evidence = state.get("verified_evidence") or state.get("evidence", [])
    evidence_context = _format_evidence(
        state.get("budgeted_evidence") or apply_evidence_budget(state["question"], source_evidence)
    )
    sources = sorted({str(item.get("source_document")) for item in state.get("evidence", []) if item.get("source_document")})
    spec = state.get("investigation_spec") or {}
    synthesis_instruction = (
        "For synthesis or comparison questions, combine multiple verified evidence items when "
        "they collectively satisfy the evidence requirements. Do not require one passage to "
        "state the entire conclusion, and do not add unsupported conclusions."
        if spec.get("answer_strategy") in {"aggregate_evidence", "compare_evidence", "explain_from_evidence"}
        else "Answer the direct information need from the verified evidence."
    )
    if spec.get("question_type") == "list_extraction":
        synthesis_instruction = (
            "This is a list question. Combine the verified passages from the requested "
            "speaker/document and enumerate the items. If one passage introduces a list "
            "and nearby verified passages continue it, do not replace the list with a "
            "claim that its details are absent. For a speaker-scoped list, treat the "
            "speaker's own numbered/introduction passage as authoritative and do not "
            "substitute a later questioner's or another speaker's similarly worded list."
        )
    elif len(spec.get("evidence_requirements", [])) > 1:
        synthesis_instruction = (
            "The question has multiple requirements. Address each requirement separately "
            "using the verified evidence, and do not declare a requirement absent merely "
            "because it appears in a different page or section."
        )
    if spec.get("intent") == "analyze_legal_case_record":
        synthesis_instruction = (
            "Organize the answer as: procedural/factual background, legal issues, party submissions, "
            "authorities, reasoning/outcome. Cite each factual claim with [filename] and its page when available. "
            "If this source is only a hearing transcript and has no final judgment, say that clearly rather than inventing one."
        )
    messages = [
        SystemMessage(
            content=(
                "You are the solver in a document-investigation workflow. Return ONLY valid JSON with "
                "answer_text, claims, and unanswered_requirements. Each claim must include claim_id, text, "
                "requirement_id, evidence_ids, and type (quoted, inferred, or conflict). Answer only "
                "from the VERIFIED evidence supplied below. Do not use rejected or merely "
                "related passages. Inferences are allowed only when they are explicitly built "
                "from the cited verified spans; do not invent missing facts. Cite claims with [filename]. "
                "When the question names a speaker, author, or person, preserve the distinction "
                "between the question subject and the speaker of each passage. A different speaker "
                "may provide valid evidence about the subject. For relations such as given, assigned, "
                "allowed, ordered, required, told, or appointed, use evidence expressing that relation; "
                "do not substitute a nearby self-estimate or intention such as 'I will take 10 minutes'. "
                "Use the ROLE METADATA and requirement relation fields. "
                "If the verified evidence does not support the information need, return "
                "Insufficient evidence instead of guessing. " + synthesis_instruction
            )
        ),
                HumanMessage(
                    content=(
                f"QUESTION:\n{state['question']}\n\n"
                f"INVESTIGATION SPECIFICATION:\n{json.dumps(spec)}\n\n"
                        f"EVIDENCE COVERAGE:\n{json.dumps(state.get('evidence_coverage', {}))}\n\n"
                        f"DOCUMENT ROLES:\n{json.dumps([{d.get('name'): d.get('record_role', 'unknown')} for d in state.get('available_documents', [])])}\n\n"
                        f"EVIDENCE:\n{evidence_context}"
            )
        ),
    ]
    service_error: dict[str, object] | None = None
    call_events: list[dict[str, object]] = []
    raw_solver_response = ""
    try:
        llm = build_llm()
        response, call_meta = invoke_with_resilience(llm, messages, node="solver")
        call_events.append(call_meta)
        raw_answer = str(response.content)
        raw_solver_response = raw_answer
        answer, structured_claims, unanswered = _parse_solver_payload(raw_answer)
        first_response_truncated = _response_was_truncated(response)
        if first_response_truncated or _solver_response_needs_repair(raw_answer, answer):
            repair, repair_meta = invoke_with_resilience(
                llm,
                [
                    messages[0],
                    HumanMessage(content=(
                        f"QUESTION:\n{state['question']}\n\n"
                        f"EVIDENCE:\n{evidence_context}\n\n"
                        "The previous Solver response was cut off before a complete JSON object. "
                        "Return a complete JSON object now. Keep answer_text to 1-3 complete sentences, "
                        "include only claims supported by the evidence, and include evidence_ids."
                    )),
                ],
                node="solver_repair",
            )
            call_events.append(repair_meta)
            repaired_raw = str(repair.content)
            if _response_was_truncated(repair):
                service_error = {
                    "node": "solver",
                    "status": "truncated",
                    "error_type": "MAX_TOKENS",
                    "message": "Answer was cut off by the model; the concise retry was also incomplete.",
                }
                answer = "The answer was cut off by the model and could not be completed safely. Please retry."
                structured_claims, unanswered = [], []
                raw_solver_response = ""
            else:
                raw_solver_response = repaired_raw
                answer, structured_claims, unanswered = _parse_solver_payload(repaired_raw)
        if not service_error and evidence_context and _is_generic_insufficient_answer(answer):
            repair, repair_meta = invoke_with_resilience(
                llm,
                [
                    SystemMessage(
                        content=(
                            "You are repairing a document answer. Verified, source-labelled "
                            "material is available. Answer the user's exact question directly "
                            "from that material in 1-4 sentences and cite [filename]. Do not "
                            "return a generic insufficient-evidence response."
                        )
                    ),
                    messages[1],
                ],
                node="solver_repair",
            )
            call_events.append(repair_meta)
            repaired_raw = str(repair.content)
            if _response_was_truncated(repair):
                service_error = {
                    "node": "solver",
                    "status": "truncated",
                    "error_type": "MAX_TOKENS",
                    "message": "Answer was cut off by the model during the repair attempt.",
                }
                answer = "The answer was cut off by the model and could not be completed safely. Please retry."
                structured_claims, unanswered = [], []
                raw_solver_response = ""
            else:
                raw_solver_response = repaired_raw
                answer, structured_claims, unanswered = _parse_solver_payload(repaired_raw)
            if _is_generic_insufficient_answer(answer):
                answer = _evidence_backed_fallback(
                    state.get("budgeted_evidence") or source_evidence
                )
        if spec.get("question_type") == "list_extraction" and re.search(
            r"(?:not (?:listed|detailed|enumerated)|specific four|subsequently identified|"
            r"(?:instead|later|elsewhere)\s+(?:identified|listed|described)|"
            r"details?\s+(?:are|were)\s+(?:identified|listed|described))",
            answer,
            re.I,
        ):
            structured = _structured_list_fallback(state.get("budgeted_evidence") or source_evidence)
            if structured:
                answer = structured
    except LLMCallFailure as failure:
        # One compact Solver retry is useful when the provider blocks or times
        # out on a larger prompt. It is deliberately not an investigation retry.
        compact_evidence = apply_evidence_budget(
            state["question"], state.get("budgeted_evidence") or source_evidence,
            max_items=min(4, MAX_EVIDENCE_ITEMS), max_chars=min(4000, MAX_EVIDENCE_CHARS),
        )
        compact_messages = [
            messages[0],
            HumanMessage(content=f"QUESTION:\n{state['question']}\n\nEVIDENCE:\n{_format_evidence(compact_evidence)}"),
        ]
        try:
            compact_response, compact_meta = invoke_with_resilience(llm, compact_messages, node="solver_compact", max_attempts=1)
            call_events.append(compact_meta)
            if _response_was_truncated(compact_response):
                service_error = {
                    "node": "solver_compact",
                    "status": "truncated",
                    "error_type": "MAX_TOKENS",
                    "message": "Answer was cut off by the model during the compact retry.",
                }
                answer = "The answer was cut off by the model and could not be completed safely. Please retry."
                structured_claims, unanswered = [], []
                raw_solver_response = ""
            else:
                answer, structured_claims, unanswered = _parse_solver_payload(str(compact_response.content))
                raw_solver_response = str(compact_response.content)
                service_error = None
        except LLMCallFailure as compact_failure:
            service_error = compact_failure.as_dict()
            answer = _service_failure_answer(compact_failure, compact_evidence)
            structured_claims, unanswered = [], []
    except Exception as exc:
        service_error = {
            "node": "solver",
            "attempt": 1,
            "status": "error",
            "error_type": type(exc).__name__,
            "message": str(exc)[:500],
        }
        answer = f"The model service failed during the Solver call ({type(exc).__name__}: {str(exc)[:300]}). This is not a generated answer."
        structured_claims, unanswered = [], []
    answer_before_completion = answer
    completion_evidence = state.get("budgeted_evidence") or source_evidence
    answer = _complete_truncated_direct_answer(answer, completion_evidence, spec)
    if answer != answer_before_completion and structured_claims:
        structured_claims = [
            {**claim, "claim": _complete_truncated_direct_answer(str(claim.get("claim", "")), completion_evidence, spec)}
            for claim in structured_claims
        ]
    solver_claims = structured_claims or _claim_support(answer, completion_evidence)
    return {
        "solver_answer": answer,
        "solver_claims": solver_claims,
        "solver_unanswered_requirements": unanswered,
        "sources_used": sources,
        "service_error": service_error,
        "llm_call_events": call_events,
        "solver_failed": bool(service_error),
        "retry_count": state.get("retry_count") or 0,
        "investigation_trace": _trace(
            state,
            "solver",
            evidence_count=len(state.get("budgeted_evidence", [])),
            evidence_chars=len(evidence_context),
            sources=sources,
            solver_answer_preview=str(answer)[:600],
            solver_answer_before_completion=str(answer_before_completion)[:600],
            raw_solver_response_preview=_redact_response_snippet(raw_solver_response),
            raw_solver_response=raw_solver_response[:12000],
            raw_solver_response_length=len(raw_solver_response),
            solver_response_needs_repair=_solver_response_needs_repair(raw_solver_response, answer),
            solver_output_limit=getattr(locals().get("llm", None), "max_output_tokens", None),
            answer_completed_from_verified_evidence=answer != answer_before_completion,
            solver_claims=solver_claims[:20],
            llm_call_events=call_events,
            service_error=service_error,
            finish_reason=_response_finish_reason(response) if "response" in locals() else None,
            answer_truncated=bool(service_error and service_error.get("status") == "truncated"),
            final_verdict_reason=(service_error or {}).get("message") if service_error else None,
        ),
    }


def audit_answer(state: AgentState) -> dict[str, object]:
    """Check whether the solver's answer is supported by the evidence."""

    source_evidence = state.get("verified_evidence") or state.get("evidence", [])
    budgeted_evidence = state.get("budgeted_evidence") or apply_evidence_budget(state["question"], source_evidence)
    # Keep the bounded item set and provenance order, but restore the complete
    # verified quote for every item sent to the Auditor.  The previous path
    # could pass a budget-truncated excerpt, creating a false citation failure.
    auditor_evidence: list[EvidenceItem] = []
    for item in budgeted_evidence:
        restored = dict(item)
        excerpt = str(item.get("excerpt", ""))
        for full_item in source_evidence:
            same_location = (
                full_item.get("source_document") == item.get("source_document")
                and full_item.get("page") == item.get("page")
                and full_item.get("sheet") == item.get("sheet")
            )
            full_excerpt = str(full_item.get("excerpt", ""))
            if same_location and (full_excerpt == excerpt or full_excerpt.startswith(excerpt.rstrip(" …"))):
                restored["excerpt"] = full_excerpt
                break
        auditor_evidence.append(restored)
    evidence_context = _format_evidence(auditor_evidence)
    budget_keys = {
        (item.get("source_document"), item.get("page"), item.get("excerpt"))
        for item in budgeted_evidence
    }
    verification_context = [
        verification for verification in state.get("evidence_verification", [])
        if (
            verification.get("source_document"),
            verification.get("page"),
            next((item.get("excerpt") for item in budgeted_evidence if item.get("source_document") == verification.get("source_document") and item.get("page") == verification.get("page")), None),
        ) in budget_keys
    ][:MAX_EVIDENCE_ITEMS]
    if state.get("solver_failed"):
        fallback = _deterministic_audit(state, budgeted_evidence)
        return {
            **fallback,
            "audit_attempts": 0,
            "service_error": state.get("service_error"),
            "investigation_trace": _trace(
                state,
                "auditor",
                audit_status="unavailable",
                fallback_validator="deterministic",
                service_error=state.get("service_error"),
            ),
        }
    system_prompt = (
        "You are an answer auditor. Return ONLY valid JSON with keys approved (boolean), "
        "confidence (number 0 to 1), issues (array of strings), reason (string), and "
        "investigation_action. investigation_action must be one of: "
        "broaden_search, search_related_concepts, inspect_additional_documents, "
        "inspect_broader_window, verify_specific_claim, resolve_contradiction, "
        "insufficient_evidence. Check factual support, source grounding, completeness, "
        "contradictions, and whether the answer exceeds the verified evidence."
    )
    user_prompt = (
        f"QUESTION:\n{state['question']}\n\n"
        f"REQUIREMENT LEDGER:\n{json.dumps(state.get('requirement_ledger', []))}\n\n"
        f"EVIDENCE:\n{evidence_context}\n\n"
        f"EVIDENCE COVERAGE:\n{json.dumps(state.get('evidence_coverage', {}))}\n\n"
        f"VERIFICATION RESULTS:\n{json.dumps(verification_context)}\n\n"
        f"CONTRADICTIONS:\n{json.dumps(state.get('contradictions', [])[:MAX_EVIDENCE_ITEMS])}\n\n"
        f"SOLVER CLAIMS ONLY:\n{json.dumps(state.get('solver_claims', []))}\n\n"
        f"UNANSWERED REQUIREMENTS:\n{json.dumps(state.get('solver_unanswered_requirements', []))}"
    )

    def parse_audit(text: str) -> tuple[dict[str, object] | None, str | None]:
        decoder = json.JSONDecoder()
        for start, character in enumerate(text):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if not isinstance(candidate, dict):
                continue
            required = {"approved", "confidence", "issues", "reason"}
            if not required.issubset(candidate):
                return None, "Auditor JSON was missing required fields."
            if not isinstance(candidate["approved"], bool):
                return None, "Auditor approved field was not boolean."
            if isinstance(candidate["confidence"], bool) or not isinstance(candidate["confidence"], (int, float)):
                return None, "Auditor confidence field was not numeric."
            if not isinstance(candidate["issues"], list) or not isinstance(candidate["reason"], str):
                return None, "Auditor issues or reason field had the wrong type."
            confidence = float(candidate["confidence"])
            if not 0 <= confidence <= 1:
                return None, "Auditor confidence was outside the range 0 to 1."
            return {
                "approved": candidate["approved"],
                "confidence": confidence,
                "audit_issues": [str(issue) for issue in candidate["issues"]],
                "audit_reason": candidate["reason"],
                "investigation_action": (
                    candidate.get("investigation_action")
                    if isinstance(candidate.get("investigation_action"), str)
                    and candidate.get("investigation_action") in ALLOWED_INVESTIGATION_ACTIONS
                    else "verify_specific_claim"
                ),
                "audit_feedback": (
                    f"{candidate['reason']} Issues: "
                    + "; ".join(str(issue) for issue in candidate["issues"])
                ).strip(),
                "claim_verdicts": candidate.get("claim_verdicts", []),
            }, None
        return None, "Auditor did not return a JSON object."

    llm = None
    last_error = "Auditor did not return a usable result."
    call_events: list[dict[str, object]] = []
    parse_failure: dict[str, object] | None = None
    for attempt in range(1, 3):
        try:
            if llm is None:
                llm = build_llm()
            response, call_meta = invoke_with_resilience(
                llm,
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=(
                            user_prompt
                            if attempt == 1
                            else (
                                f"QUESTION:\n{state['question']}\n\n"
                                f"EVIDENCE:\n{evidence_context}\n\n"
                                f"SOLVER CLAIMS:\n{json.dumps(state.get('solver_claims', []))}\n\n"
                                "Return only the required JSON object."
                            )
                        )
                    ),
                ],
                node="auditor",
            )
            call_events.append(call_meta)
            parsed, error = parse_audit(str(response.content))
            if parsed is not None:
                # A model can occasionally approve its own generic fallback even
                # though the extraction stage found direct evidence.  Treat that
                # as a recoverable solver failure instead of presenting a false
                # "approved" insufficient-evidence answer to the user.
                answer_is_generic_insufficient = str(state.get("solver_answer", "")).strip().lower().startswith(
                    "insufficient evidence"
                )
                if answer_is_generic_insufficient and budgeted_evidence:
                    parsed.update({
                        "approved": False,
                        "audit_issues": [
                            "Verified evidence was available, but the solver returned a generic insufficient-evidence answer."
                        ],
                        "audit_reason": "Re-answer the question directly from the verified evidence.",
                        "investigation_action": "verify_specific_claim",
                        "audit_feedback": "Verified evidence exists. Re-extract the relevant passage and answer the requested fact directly.",
                    })
                claims = state.get("solver_claims", [])
                evidence_ids = {f"E{index + 1}" for index, _ in enumerate(budgeted_evidence)}
                claim_verdicts = _claim_verdict_rows(state, auditor_evidence)
                for row in claim_verdicts:
                    row["auditor_reason"] = str(parsed.get("reason") or row.get("reason") or "")
                unsupported_claims = [
                    claim.get("claim", "") for claim in state.get("solver_claims", [])
                    if not claim.get("evidence_ids")
                ]
                certificate = state.get("exhaustion_certificate") or {}
                refusal_with_audit = (
                    _is_generic_insufficient_answer(str(state.get("solver_answer", "")))
                    and not budgeted_evidence
                    and bool(certificate.get("complete"))
                    and bool(state.get("coverage_complete"))
                )
                hard_claim_failures = [
                    row for row in claim_verdicts
                    if row.get("verdict") in {"NOT_SUPPORTED", "WRONG_ATTRIBUTION"}
                    and row.get("core")
                ]
                rule_reject = (
                    not budgeted_evidence
                    or (answer_is_generic_insufficient and bool(budgeted_evidence))
                    or unsupported_claims
                    or (not claim_verdicts and bool(state.get("solver_claims")))
                    or hard_claim_failures
                ) and not refusal_with_audit
                if rule_reject:
                    parsed.update({
                        "approved": False,
                        "confidence": _claim_confidence(claim_verdicts),
                        "audit_issues": [
                            "The solver answer contains claims that are not fully supported by verified evidence."
                            + (f" Unsupported claims: {unsupported_claims[:3]}" if unsupported_claims else "")
                            + (f" Claim failures: {hard_claim_failures[:3]}" if hard_claim_failures else "")
                        ],
                        "audit_reason": "Every material claim must map to verified evidence and complete requirement coverage.",
                        "investigation_action": "verify_specific_claim",
                        "audit_feedback": "Re-investigate the unsupported claim and preserve the correct attribution before approving.",
                    })
                else:
                    # Preserve an explicit factual rejection from the Auditor.
                    # The deterministic checks may upgrade a well-supported
                    # answer only when the provider itself approved it; they
                    # must not silently erase a provider rejection.
                    if parsed.get("approved"):
                        parsed.update({"approved": True, "confidence": None if refusal_with_audit else (_claim_confidence(claim_verdicts) if claim_verdicts else None)})
                parsed["claim_verdicts"] = claim_verdicts
                parsed["audit_attempts"] = attempt
                parsed["audit_result"] = dict(parsed)
                parsed["audit_status"] = "approved" if parsed.get("approved") else "rejected"
                parsed["audit_unavailable"] = False
                parsed["llm_call_events"] = call_events
                parsed["investigation_trace"] = _trace(
                    state,
                    "auditor",
                    approved=parsed.get("approved", False),
                    evidence_count=len(budgeted_evidence),
                    evidence_chars=len(evidence_context),
                    action=parsed.get("investigation_action"),
                    audit_status=parsed.get("audit_status"),
                    audit_issues=parsed.get("audit_issues", []),
                    claim_verdicts=claim_verdicts,
                    claim_details=claim_verdicts,
                    auditor_quote_limit=MAX_EVIDENCE_CHARS,
                )
                parsed.update(_preserve_best_candidate(state, parsed))
                return parsed
            last_error = error or last_error
            parse_failure = {"status": "invalid_json", "message": last_error, "raw_snippet": _redact_response_snippet(str(response.content))}
        except LLMCallFailure as failure:
            last_error = f"{failure.status}: {failure.message}"
            parse_failure = failure.as_dict()
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {str(exc)[:500]}"
            parse_failure = {"status": "error", "error_type": type(exc).__name__, "message": str(exc)[:500]}

    fallback = _deterministic_audit(state, budgeted_evidence)
    if parse_failure and parse_failure.get("status") == "invalid_json":
        # Invalid model formatting is distinct from transport failure. Do not
        # invent a fixed confidence for a rejected result.
        fallback["confidence"] = _claim_confidence(_claim_verdict_rows(state, budgeted_evidence))
        fallback["audit_issues"] = [
            f"Audit validation failed after two attempts: {parse_failure.get('message', 'invalid auditor response')}"
        ] + list(fallback.get("audit_issues", []))
    fallback_result = {
        **fallback,
        "audit_attempts": 2,
        "service_error": parse_failure,
        "llm_call_events": call_events,
        "investigation_trace": _trace(
            state,
            "auditor",
            approved=fallback.get("approved", False),
            audit_status=fallback.get("audit_status"),
            fallback_validator="deterministic",
            evidence_count=len(budgeted_evidence),
            evidence_chars=len(evidence_context),
            service_error=parse_failure,
            action="",
        ),
    }
    fallback_result.update(_preserve_best_candidate(state, fallback_result))
    return fallback_result


def revise_investigation(state: AgentState) -> dict[str, object]:
    """Use audit feedback to change the next selection/inspection pass."""

    issues = state.get("audit_issues") or []
    gaps = [str(gap) for gap in state.get("investigation_gaps", []) if str(gap).strip()]
    feedback = str(
        state.get("audit_feedback")
        or state.get("audit_reason")
        or "; ".join(str(issue) for issue in issues)
        or "; ".join(gaps)
        or "The evidence was not sufficient to approve the answer."
    )
    lowered = feedback.lower()
    action = state.get("investigation_action", "broaden_search")
    if action not in ALLOWED_INVESTIGATION_ACTIONS:
        action = "broaden_search"
    if "contradict" in lowered or action == "resolve_contradiction":
        direction = "Inspect additional candidate sources and compare conflicting statements."
    elif "section" in lowered or "wrong" in lowered or action == "inspect_broader_window":
        direction = "Inspect a broader nearby section and verify the relevant clause or passage."
    elif "misunderstood" in lowered or "misinterpret" in lowered or action == "verify_specific_claim":
        direction = "Re-extract the relevant passage and verify its meaning before answering."
    else:
        direction = "Broaden the investigation and inspect additional nearby context for supporting evidence."

    retry_count = (state.get("retry_count") or 0) + 1
    evidence_signature = _evidence_signature(
        state.get("verified_evidence") or state.get("candidate_evidence")
    )
    current_window = max(1, int(state.get("inspection_window_size") or 1))
    current_spec = dict(state.get("investigation_spec") or _fallback_investigation_spec(state["question"]))
    concepts = [str(value) for value in current_spec.get("search_concepts", [])]
    missing_requirements = [
        str(requirement) for requirement, details in (state.get("evidence_coverage") or {}).items()
        if not details.get("satisfied")
    ]
    concepts.extend(missing_requirements)
    if any(term in lowered for term in ("resignation", "resign", "employee")):
        concepts.extend([
            "voluntary resignation", "employee-initiated resignation",
            "resignation notice", "notice period for resignation",
        ])
        current_spec["event"] = "voluntary employee resignation"
    if "probation" in lowered:
        concepts.append("probation employee notice")
    current_spec["search_concepts"] = list(dict.fromkeys(concepts))[:12]
    current_spec["constraints"] = list(dict.fromkeys([
        *[str(value) for value in current_spec.get("constraints", [])],
        feedback,
    ]))[:8]
    revised_query = (
        f"{state['question']}\n\n"
        f"Investigation gap: {'; '.join(gaps) if gaps else 'the answer-bearing section has not been located'}\n"
        f"Auditor feedback from the previous attempt: {feedback}\n"
        f"Investigation instruction: {direction}"
    )
    return {
        "retry_count": retry_count,
        "audit_feedback": feedback,
        "investigation_action": action,
        "investigation_spec": current_spec,
        "investigation_query": revised_query,
        "inspection_window_size": min(current_window + 1, 3),
        "search_paths_attempted": list(dict.fromkeys([
            *(state.get("search_paths_attempted") or []),
            f"retry_{retry_count}_{action}",
        ])),
        "investigation_trace": _trace(
            state,
            "revise_investigation",
            retry_count=retry_count,
            gap=gaps,
            action=action,
            next_window_size=min(current_window + 1, 3),
            ladder_rung=retry_count,
            retry_reason=(
                "New investigation pass requested because the prior audit identified an unresolved claim or requirement."
            ),
            evidence_signature=evidence_signature,
        ),
        "last_retry_evidence_signature": evidence_signature,
    }


def finalize_answer(state: AgentState) -> dict[str, str]:
    """Expose the solver answer, including an audit warning when needed."""

    def finish(payload: dict[str, object], reason: str) -> dict[str, object]:
        payload["investigation_trace"] = _trace(
            state,
            "finalize_answer",
            final_verdict_reason=reason,
            approved=bool(payload.get("approved", state.get("approved", False))),
            retry_count=state.get("retry_count", 0),
        )
        return payload

    spec = state.get("investigation_spec") or {}
    if spec.get("target_attribute") == "exact_name_presence":
        phrase = str(spec.get("exact_phrase") or spec.get("subject") or "the requested name")
        evidence = state.get("verified_evidence", [])
        if evidence:
            sources = sorted({str(item.get("source_document")) for item in evidence})
            message = f"Yes — the exact name “{phrase}” was found in: {', '.join(sources)}."
            return finish({"final_answer": message, "approved": True, "confidence": 1.0}, "Exact requested name was found in verified evidence.")
        message = f"No — the exact name “{phrase}” was not found in the searchable text of the selected document(s)."
        return finish({
            "final_answer": message,
            "approved": False,
            "confidence": None,
            "audit_reason": "The requested exact name was not found in the inspected source text.",
            "audit_issues": ["No verified evidence matched the requested name."],
        }, "The requested exact name was not found in verified evidence.")
    inferred_hearing_limitation = (
        (spec.get("intent") == "analyze_legal_case_record")
        and bool(re.search(r"final (?:conclusion|judgment|decision|order|outcome)|court'?s final", state.get("question", ""), re.I))
        and bool(state.get("inspected_documents"))
        and all(item.get("metadata", {}).get("record_kind") == "hearing_transcript" for item in state.get("inspected_documents", []))
    )
    if state.get("investigation_action") == "hearing_transcript_missing_final_judgment" or inferred_hearing_limitation:
        message = (
            "This selected document is a hearing transcript, not a final judgment. "
            "Its final page ends with ‘END OF DAY’S PROCEEDINGS’, so it can support a summary "
            "of submissions made during that hearing but cannot support a claim about the court’s "
            "final conclusion. Upload the final judgment or order for that part of the analysis."
        )
        return finish({
            "final_answer": message,
            "approved": False,
            "confidence": None,
            "audit_reason": "The selected source does not contain a final judgment.",
            "audit_issues": ["Hearing transcript only; final judgment absent."],
        }, "The selected source is a hearing transcript and does not contain a final judgment.")
    if state.get("solver_failed"):
        error = state.get("service_error") or {}
        message = str(state.get("solver_answer") or "The model service failed before producing an answer.")
        return finish({
            "final_answer": message,
            "approved": False,
            "confidence": None,
            "audit_status": "unavailable",
            "service_error": error,
            "audit_reason": "Solver model service failure; investigation was not classified as insufficient evidence.",
            "audit_issues": ["The Solver service was unavailable."],
        }, "Solver service failure; no factual insufficiency conclusion was made.")
    if state.get("audit_status") == "fallback_validated":
        return finish({
            "final_answer": state.get("solver_answer", ""),
            "approved": True,
            "confidence": None,
            "audit_status": "fallback_validated",
            "audit_reason": "Auditor unavailable; deterministic evidence check passed.",
            "audit_issues": [],
        }, "Auditor unavailable; deterministic fallback validation passed.")
    if state.get("audit_status") == "unavailable":
        return finish({
            "final_answer": (
            f"{state.get('solver_answer', '')}\n\n"
                "AUDITOR UNAVAILABLE: The answer was generated from the investigated evidence, "
                "but independent Auditor validation was unavailable."
            ),
            "approved": False,
            "confidence": None,
            "audit_status": "unavailable",
            "service_error": state.get("service_error"),
            "audit_reason": "Auditor transport or validation failure; no factual rejection was made.",
            "audit_issues": ["Auditor unavailable; deterministic fallback did not approve the result."],
        }, "Auditor unavailable; answer was not classified as factually rejected.")
    # A later rejected retry must not replace a better source-backed candidate
    # that was already recorded by an earlier audit pass.
    if (
        state.get("best_solver_answer")
        and state.get("best_audit_status") == "approved"
        and not state.get("approved", False)
    ):
        return finish({
            "final_answer": state["best_solver_answer"],
            "solver_answer": state["best_solver_answer"],
            "solver_claims": state.get("best_solver_claims", []),
            "approved": True,
            "confidence": state.get("best_confidence"),
            "audit_status": "approved",
            "audit_reason": "Preserved the earlier approved candidate after a later retry was rejected.",
            "audit_issues": [],
        }, "An earlier approved candidate was preserved over a rejected retry.")
    certificate = state.get("exhaustion_certificate") or {}
    if state.get("exhaustive_search_attempted") and not certificate.get("coverage_complete", True) and not state.get("verified_evidence"):
        pages_read = int(certificate.get("pages_processed") or 0)
        strategies = ", ".join(str(value) for value in certificate.get("paths_attempted") or []) or "bounded search"
        message = (
            f"Investigation stopped at its limit (pages read: {pages_read}, strategies tried: {strategies}). "
            "The selected document was not fully covered, so the system is not claiming that the information is absent."
        )
        return finish({
            "final_answer": message,
            "approved": False,
            "confidence": None,
            "audit_status": "investigation_budget_exhausted",
            "audit_reason": "The bounded search budget was reached before complete document coverage.",
            "audit_issues": ["Search budget exhausted without an absence certificate."],
        }, "Search budget exhausted without enough coverage to certify absence.")
    if state.get("insufficient_evidence", False):
        if state.get("verified_evidence"):
            # Evidence exists, so this is a partial-support situation rather
            # than a whole-answer insufficiency finding.
            return finish({
                "final_answer": state.get("solver_answer", ""),
                "approved": False,
                "confidence": _claim_confidence(state.get("claim_verdicts", [])),
                "audit_status": "partial_support",
                "audit_reason": "Some requirements have verified evidence; unsupported requirements are listed separately.",
                "audit_issues": state.get("audit_issues", []),
            }, "Partial verified support was preserved instead of converting it to a blanket refusal.")
        unresolved = list(certificate.get("unresolved_requirements") or state.get("investigation_gaps") or [])
        detail = (
            "Insufficient evidence after bounded investigation."
            if certificate.get("complete")
            else "The investigation is incomplete; the remaining search paths have not been exhausted."
        )
        if unresolved:
            detail += " Unresolved requirements: " + "; ".join(str(item) for item in unresolved[:6]) + "."
        message = (
            detail + " The system will not make an unsupported assumption."
        )
        return finish({
            "solver_answer": message,
            "final_answer": message,
            "sources_used": [],
            "approved": False,
            "confidence": None,
            "audit_reason": "No verified evidence was available after the bounded investigation state was reached.",
            "audit_issues": ["Insufficient verified evidence."],
        }, "No verified evidence was available after the bounded investigation.")
    if state.get("approved", False):
        return finish({"final_answer": state["solver_answer"]}, "Auditor approved the source-backed answer.")
    return finish({
        "final_answer": (
            f"{state['solver_answer']}\n\n"
            "AUDIT WARNING: The answer was not fully approved. "
            f"Confidence: {state.get('confidence', 0.0):.2f}. "
            f"Issues: {state.get('audit_issues', [])}"
        )
    }, str(state.get("audit_reason") or "Auditor did not approve the answer."))


def route_after_audit(state: AgentState) -> str:
    """Choose approval, one retry, or bounded finalization."""

    if state.get("audit_status") in {"unavailable", "fallback_validated"} or state.get("audit_unavailable"):
        return "finalize"
    if state.get("approved", False):
        return "finalize"
    current_signature = _evidence_signature(
        state.get("verified_evidence") or state.get("candidate_evidence")
    )
    if (
        state.get("retry_count", 0)
        and current_signature == state.get("last_retry_evidence_signature")
    ):
        # Do not spend another investigation retry when the evidence set did
        # not change; the Solver/Auditor must finish from the same evidence.
        return "finalize"
    if (state.get("retry_count") or 0) < MAX_RETRIES:
        return "revise_investigation"
    return "finalize"


def route_after_sufficiency(state: AgentState) -> str:
    """Avoid calling the Solver when verification found no direct support."""

    if (state.get("investigation_spec") or {}).get("target_attribute") == "exact_name_presence":
        if state.get("verified_evidence"):
            return "finalize"
        if not state.get("exhaustive_search_attempted"):
            return "bounded_exhaustive_search"
        return "finalize"
    if state.get("investigation_action") == "hearing_transcript_missing_final_judgment":
        return "finalize"
    if (
        state.get("verified_evidence")
        and not state.get("insufficient_evidence")
    ):
        return "sufficient"
    if (state.get("retry_count") or 0) < MAX_RETRIES:
        return "revise_investigation"
    if not state.get("exhaustive_search_attempted"):
        return "bounded_exhaustive_search"
    return "finalize"


def build_graph():
    """Compile the graph; pass documents_dir in the initial state to choose a corpus."""

    workflow = StateGraph(AgentState)
    workflow.add_node("analyze_question", analyze_question)
    workflow.add_node("discover_documents", discover_documents)
    workflow.add_node("select_documents", select_documents)
    workflow.add_node("inspect_selected_documents", inspect_selected_documents)
    workflow.add_node("extract_evidence", extract_evidence)
    workflow.add_node("verify_evidence", verify_evidence)
    workflow.add_node("assess_evidence_sufficiency", assess_evidence_sufficiency)
    workflow.add_node("bounded_exhaustive_search", bounded_exhaustive_search)
    workflow.add_node("rank_evidence", rank_evidence)
    workflow.add_node("budget_evidence", budget_evidence)
    workflow.add_node("solve_question", solve_question)
    workflow.add_node("audit_answer", audit_answer)
    workflow.add_node("revise_investigation", revise_investigation)
    workflow.add_node("finalize_answer", finalize_answer)
    workflow.add_edge(START, "analyze_question")
    workflow.add_edge("analyze_question", "discover_documents")
    workflow.add_edge("discover_documents", "select_documents")
    workflow.add_edge("select_documents", "inspect_selected_documents")
    workflow.add_edge("inspect_selected_documents", "extract_evidence")
    workflow.add_edge("extract_evidence", "verify_evidence")
    workflow.add_edge("verify_evidence", "assess_evidence_sufficiency")
    workflow.add_conditional_edges(
        "assess_evidence_sufficiency",
        route_after_sufficiency,
        {
            "sufficient": "rank_evidence",
            "revise_investigation": "revise_investigation",
            "bounded_exhaustive_search": "bounded_exhaustive_search",
            "finalize": "finalize_answer",
        },
    )
    workflow.add_edge("bounded_exhaustive_search", "extract_evidence")
    workflow.add_edge("rank_evidence", "budget_evidence")
    workflow.add_edge("budget_evidence", "solve_question")
    workflow.add_edge("solve_question", "audit_answer")
    workflow.add_conditional_edges(
        "audit_answer",
        route_after_audit,
        {"finalize": "finalize_answer", "revise_investigation": "revise_investigation"},
    )
    workflow.add_edge("revise_investigation", "select_documents")
    workflow.add_edge("finalize_answer", END)
    return workflow.compile()
