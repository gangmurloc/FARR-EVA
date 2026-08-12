from __future__ import annotations

import unittest

from farr_star.contracts import (
    QuestionContract,
    answer_contract_compliance,
    heuristic_contract,
    parse_contract,
)


class ContractTests(unittest.TestCase):
    def test_llm_contract_preserves_explicit_return_alternatives(self) -> None:
        contract = parse_contract(
            """{
              "reasoning_type":"comparison_chain",
              "answer_type":"work",
              "target":"the film whose director was born earlier",
              "expected_hops":3,
              "requires_comparison":true,
              "return_compared_entity":true,
              "allowed_answers":["Film A","Film B"],
              "constraints":["compare both directors' birth dates"],
              "retrieval_queries":["Film A director birth date","Film B director birth date"]
            }""",
            "Which film has the director born earlier, Film A or Film B?",
        )
        self.assertEqual(contract.reasoning_type, "comparison_chain")
        self.assertEqual(contract.allowed_answers, ["Film A", "Film B"])
        self.assertTrue(
            answer_contract_compliance("Film A", contract)[0]
        )
        self.assertFalse(
            answer_contract_compliance("17 March 1940", contract)[0]
        )
        self.assertFalse(
            answer_contract_compliance("Some Director", contract)[0]
        )

    def test_boolean_and_numeric_contracts(self) -> None:
        boolean = heuristic_contract(
            "Were both people born in the same country?"
        )
        self.assertEqual(boolean.answer_type, "boolean")
        self.assertTrue(answer_contract_compliance("yes", boolean)[0])
        self.assertFalse(answer_contract_compliance("France", boolean)[0])

        numeric = QuestionContract(answer_type="number")
        self.assertTrue(answer_contract_compliance("215th", numeric)[0])
        self.assertFalse(answer_contract_compliance("Vienna", numeric)[0])

    def test_fallback_extracts_forced_choice_without_llm_json(self) -> None:
        contract = parse_contract(
            "not valid JSON",
            (
                "Which film has the director who was born earlier, "
                "The Rise And Rise Of Michael Rimmer or Bar 51?"
            ),
        )
        self.assertEqual(
            contract.allowed_answers,
            ["The Rise And Rise Of Michael Rimmer", "Bar 51"],
        )
        self.assertTrue(
            answer_contract_compliance(
                "The Rise And Rise Of Michael Rimmer",
                contract,
            )[0]
        )
        self.assertFalse(
            answer_contract_compliance("Kevin Billington", contract)[0]
        )

    def test_why_question_overrides_bad_date_contract(self) -> None:
        contract = parse_contract(
            """{
              "reasoning_type":"temporal",
              "answer_type":"date",
              "target":"the date of death",
              "expected_hops":2,
              "retrieval_queries":["death date"]
            }""",
            "Why did Grand Duke Kirill's wife die?",
        )
        self.assertEqual(contract.answer_type, "cause")
        self.assertEqual(contract.reasoning_type, "compositional")
        self.assertFalse(
            answer_contract_compliance("2 March 1936", contract)[0]
        )
        self.assertFalse(
            answer_contract_compliance(
                "She died on 2 March 1936",
                contract,
            )[0]
        )
        self.assertTrue(
            answer_contract_compliance("stroke", contract)[0]
        )


if __name__ == "__main__":
    unittest.main()
