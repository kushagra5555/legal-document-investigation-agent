import unittest

from app.graph import _complete_truncated_direct_answer, _parse_solver_payload


class SolverPayloadTests(unittest.TestCase):
    def test_markdown_fenced_json_is_parsed(self):
        answer, claims, unanswered = _parse_solver_payload(
            '```json\n{"answer_text":"The case is Alpha.","claims":[]}\n```'
        )
        self.assertEqual(answer, "The case is Alpha.")
        self.assertEqual(claims, [])
        self.assertEqual(unanswered, [])

    def test_truncated_json_salvages_answer_without_showing_envelope(self):
        answer, claims, unanswered = _parse_solver_payload(
            '```json\n{"answer_text":"The documents mention the case Alpha'
        )
        self.assertEqual(answer, "The documents mention the case Alpha")
        self.assertNotIn("answer_text", answer)
        self.assertEqual(claims, [])
        self.assertEqual(unanswered, [])

    def test_direct_entity_answer_is_completed_from_verified_evidence(self):
        answer = _complete_truncated_direct_answer(
            "Based on the transcript, the Chief Justice mentioned is Chief Justice D.Y",
            [{"source_document": "case.pdf", "page": 1, "excerpt": "HON'BLE THE CHIEF JUSTICE DY CHANDRACHUD"}],
            {"direct_identification_required": True, "target_attribute": "name", "target_entity": "Chief Justice"},
        )
        self.assertIn("D.Y. Chandrachud", answer)

    def test_complete_direct_entity_answer_is_unchanged(self):
        answer = "The named person is Chief Justice D.Y. Chandrachud."
        self.assertEqual(
            _complete_truncated_direct_answer(
                answer,
                [{"excerpt": "Chief Justice DY CHANDRACHUD"}],
                {"direct_identification_required": True, "target_attribute": "name", "target_entity": "Chief Justice"},
            ),
            answer,
        )


if __name__ == "__main__":
    unittest.main()
