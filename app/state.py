"""State shared by the first LangGraph workflow."""

from typing import TypedDict


class DocumentMetadata(TypedDict):
    name: str
    path: str
    suffix: str
    size_bytes: int
    title: str
    date: str | None
    pages: int | None
    entities: list[str]
    description: str


class EvidenceItem(TypedDict, total=False):
    source_document: str
    source_path: str
    excerpt: str
    page: int | None
    sheet: str
    row: int
    line: int
    paragraph: int
    section: str
    table: str
    speaker: str
    related_speakers: list[str]
    subject_entity: str
    referenced_entity: str
    relation: str
    temporal_role: str
    polarity: str


class AgentState(TypedDict, total=False):
    question: str
    documents_dir: str
    document_scope: list[str]
    source_scope_explicit: bool
    question_analysis: str
    original_question_terms: list[str]
    planner_variants: list[str]
    planner_validation: dict
    investigation_spec: dict
    requirement_ledger: list[dict]
    available_documents: list[DocumentMetadata]
    active_corpus: list[DocumentMetadata]
    corpus_map: dict
    document_maps: dict
    candidate_documents: list[str]
    document_relevance: dict
    document_assessments: dict
    selected_documents: list[str]
    section_candidates: dict
    inspection_plan: dict
    inspected_documents: list[dict]
    evidence: list[EvidenceItem]
    candidate_evidence: list[EvidenceItem]
    verified_evidence: list[EvidenceItem]
    evidence_verification: list[dict]
    evidence_coverage: dict
    attribution_context: dict
    investigation_gaps: list[str]
    investigation_gap: str
    contradictions: list[dict]
    evidence_budget: dict
    insufficient_evidence: bool
    investigation_action: str
    budgeted_evidence: list[EvidenceItem]
    solver_answer: str
    solver_claims: list[dict]
    solver_unanswered_requirements: list[str]
    sources_used: list[str]
    approved: bool
    confidence: float
    audit_issues: list[str]
    audit_reason: str
    audit_feedback: str
    audit_result: dict
    audit_status: str
    audit_unavailable: bool
    service_error: dict
    llm_call_events: list[dict]
    solver_failed: bool
    best_solver_answer: str
    best_solver_claims: list[dict]
    best_audit_status: str
    best_confidence: float | None
    best_audit_reason: str
    audit_attempts: int
    retry_count: int
    investigation_query: str
    inspection_window_size: int
    final_answer: str
    investigation_trace: list[dict]
    investigation_history: list[dict]
    search_paths_attempted: list[str]
    exhaustion_certificate: dict
    exhaustive_search_attempted: bool
    document_investigation: dict
    documents_with_candidates: list[str]
    documents_with_verified_evidence: list[str]
    documents_exhausted: list[str]
    corpus_metrics: dict
