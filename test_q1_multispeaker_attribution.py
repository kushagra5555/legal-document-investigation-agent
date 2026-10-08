import unittest

from app.graph import _claim_support, _coerce_general_spec, _requirement_ledger, verify_evidence
from app.tools import extract_relevant_section, inspect_document_window


class Q1MultiSpeakerAttributionTests(unittest.TestCase):
    def test_named_person_and_adjacent_judicial_time_limit(self):
        question = (
            "Who was K.M. Nataraj appearing for, and what time limit was he given "
            "for his submissions?"
        )
        spec = _coerce_general_spec(
            {
                "intent": "factual_extraction",
                "question_type": "attribute_lookup",
                "answer_strategy": "direct",
                "subject": "K.M. Nataraj",
                "event": "appearing for and time limit",
                "information_needed": "Client representation; Time allocation for oral submissions",
                "evidence_requirements": ["Court transcripts", "Legal news reports", "Case orders"],
                "search_concepts": ["K.M. Nataraj", "appearing", "time limit"],
                "speakers": ["K.M. Nataraj"],
            },
            question,
        )
        inspected = inspect_document_window(
            "documents/297962019_2023-09-04.pdf",
            "K.M. Nataraj appearing for time limit minutes Union",
            window_size=1,
            page_start=23,
            page_end=23,
        )
        extracted = extract_relevant_section(
            question,
            inspected,
            "K.M. Nataraj appearing for time limit minutes Union",
            spec,
        )
        candidates = [
            {
                "source_document": item["source"],
                "excerpt": item["text"],
                "page": item.get("page"),
                "speaker": item.get("speaker"),
                "related_speakers": item.get("related_speakers", []),
            }
            for item in extracted["evidence"]
        ]
        verified = verify_evidence(
            {
                "question": question,
                "investigation_spec": spec,
                "requirement_ledger": _requirement_ledger(question, spec),
                "candidate_evidence": candidates,
            }
        )
        self.assertTrue(any("For the Union" in item["excerpt"] for item in verified["verified_evidence"]))
        self.assertTrue(any("10 minutes" in item["excerpt"] for item in verified["verified_evidence"]))
        self.assertTrue(any(item.get("speaker", "").startswith("JUSTICE SANJAY") for item in verified["verified_evidence"]))
        self.assertTrue(all(item["status"] == "SUPPORTED" for item in verified["requirement_ledger"]))
        claims = _claim_support(
            "K.M. Nataraj appeared for the Union. He was timed to 10 minutes.",
            verified["verified_evidence"],
        )
        self.assertTrue(all(claim["evidence_ids"] for claim in claims))


if __name__ == "__main__":
    unittest.main()
