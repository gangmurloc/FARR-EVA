from __future__ import annotations

import unittest
from unittest.mock import patch

from farr_star.contracts import QuestionContract
from farr_star.pipeline import (
    FARRSTAR,
    STARConfig,
    validate_solution,
)
from farr.types import FARRResult, FARRStats


def expert(answer: str, evidence: list[dict]) -> FARRResult:
    return FARRResult(
        answer=answer,
        evidence=evidence,
        stats=FARRStats(llm_calls=2, retrieval_calls=1),
    )


class ScriptedLLM:
    model_name = "dummy"

    def __call__(self, prompt: str) -> str:
        if "[FARR_STAR:CONTRACT]" in prompt:
            return """{
              "reasoning_type":"comparison_chain",
              "answer_type":"work",
              "target":"the film whose director was born earlier",
              "expected_hops":3,
              "requires_comparison":true,
              "return_compared_entity":true,
              "allowed_answers":["Film A","Film B"],
              "constraints":["return the winning film"],
              "retrieval_queries":[]
            }"""
        if "[FARR_STAR:ADJUDICATE]" in prompt:
            # Deliberately bad judge output: the deterministic target gate
            # must prevent an intermediate director from being returned.
            return """{
              "candidate_a_label":"UNCERTAIN",
              "candidate_a_target_match":true,
              "candidate_b_label":"SUPPORTED",
              "candidate_b_target_match":true,
              "selected":"B",
              "answer":"Some Director",
              "missing_queries":[],
              "reason":"scripted bad selection"
            }"""
        raise AssertionError(f"Unexpected prompt: {prompt[:80]}")


class DateLLM:
    model_name = "dummy"

    def __call__(self, prompt: str) -> str:
        if "[FARR_STAR:CONTRACT]" in prompt:
            return """{
              "reasoning_type":"comparison_chain",
              "answer_type":"date",
              "target":"the date the husband died",
              "expected_hops":2,
              "requires_comparison":false,
              "return_compared_entity":false,
              "allowed_answers":[],
              "constraints":[],
              "retrieval_queries":[]
            }"""
        if "[FARR_STAR:ADJUDICATE]" in prompt:
            return """{
              "candidate_a_label":"UNSUPPORTED",
              "candidate_a_target_match":false,
              "candidate_b_label":"SUPPORTED",
              "candidate_b_target_match":true,
              "selected":"B",
              "answer":"March 7, 1983",
              "missing_queries":[],
              "reason":"more specific date"
            }"""
        raise AssertionError(f"Unexpected prompt: {prompt[:80]}")


class ComparisonRepairLLM:
    model_name = "dummy"

    def __call__(self, prompt: str) -> str:
        if "[FARR_STAR:CONTRACT]" in prompt:
            return """{
              "reasoning_type":"comparison_chain",
              "answer_type":"work",
              "target":"the film whose director died earlier",
              "expected_hops":3,
              "requires_comparison":true,
              "return_compared_entity":true,
              "allowed_answers":["Film A","Film B"],
              "constraints":["compare both director death dates"],
              "retrieval_queries":[]
            }"""
        if "[FARR_STAR:ADJUDICATE]" in prompt:
            return """{
              "candidate_a_label":"SUPPORTED",
              "candidate_a_target_match":true,
              "candidate_b_label":"UNSUPPORTED",
              "candidate_b_target_match":true,
              "selected":"A",
              "answer":"Film A",
              "missing_queries":[],
              "reason":"scripted reversed comparison"
            }"""
        if "[FARR_STAR:REPAIR]" in prompt:
            return """{
              "answer":"Film B",
              "target_match":true,
              "complete":true,
              "steps":[
                {
                  "claim":"Film A's director died in 2000",
                  "doc_id":1,
                  "quote":"Director A died in 2000"
                },
                {
                  "claim":"Film B's director died in 1999",
                  "doc_id":2,
                  "quote":"Director B died in 1999"
                }
              ]
            }"""
        if "[FARR_STAR:VERIFY]" in prompt:
            return """{
              "label":"SUPPORTED",
              "target_match":true,
              "reason":"1999 is earlier than 2000",
              "corrected_answer":""
            }"""
        raise AssertionError(f"Unexpected prompt: {prompt[:80]}")


class ConservativeGateLLM:
    model_name = "dummy"

    def __call__(self, prompt: str) -> str:
        if "[FARR_STAR:CONTRACT]" in prompt:
            return """{
              "reasoning_type":"bridge",
              "answer_type":"entity",
              "target":"the requested final entity",
              "expected_hops":2,
              "requires_comparison":false,
              "return_compared_entity":false,
              "allowed_answers":[],
              "constraints":[],
              "retrieval_queries":[]
            }"""
        if "[FARR_STAR:ADJUDICATE]" in prompt:
            return """{
              "candidate_a_label":"SUPPORTED",
              "candidate_a_target_match":true,
              "candidate_b_label":"SUPPORTED",
              "candidate_b_target_match":true,
              "selected":"A",
              "answer":"FARR answer",
              "missing_queries":[],
              "reason":"both appear supported"
            }"""
        raise AssertionError(f"Unexpected prompt: {prompt[:80]}")


class PipelineTests(unittest.TestCase):
    def test_target_gate_rejects_intermediate_answer(self) -> None:
        evidence = [
            {
                "page_content": "Film A was directed by Director A.",
                "metadata": {"title": "Film A"},
            },
            {
                "page_content": "Director A was born in 1930.",
                "metadata": {"title": "Director A"},
            },
        ]
        with (
            patch(
                "farr_star.pipeline.FARRV2"
            ) as farr_class,
            patch(
                "farr_star.pipeline.ircot",
                return_value=expert("Some Director", evidence),
            ),
        ):
            farr_class.return_value.answer.return_value = expert(
                "Film A",
                evidence,
            )
            result = FARRSTAR(
                lambda query, top_k: evidence[:top_k],
                ScriptedLLM(),
                star_config=STARConfig(
                    enable_contract_retrieval=False,
                    enable_repair=False,
                ),
            ).answer(
                "Which film has the director born earlier, Film A or Film B?"
            )

        self.assertEqual(result.answer, "Film A")
        self.assertEqual(result.stats.selected_expert, "farr")
        self.assertEqual(result.stats.route, "contract_gate")

    def test_when_question_does_not_force_shorter_year(self) -> None:
        evidence = [
            {
                "page_content": "The husband died on March 7, 1983.",
                "metadata": {"title": "Husband"},
            }
        ]
        with (
            patch("farr_star.pipeline.FARRV2") as farr_class,
            patch(
                "farr_star.pipeline.ircot",
                return_value=expert("March 7, 1983", evidence),
            ),
        ):
            farr_class.return_value.answer.return_value = expert(
                "1983",
                evidence,
            )
            result = FARRSTAR(
                lambda query, top_k: evidence[:top_k],
                DateLLM(),
                star_config=STARConfig(
                    enable_contract_retrieval=False,
                    enable_repair=False,
                ),
            ).answer("When did the singer's husband die?")

        self.assertEqual(result.answer, "March 7, 1983")
        self.assertEqual(result.stats.route, "ircot_default")
        self.assertEqual(result.contract.reasoning_type, "compositional")

    def test_comparison_disagreement_forces_evidence_repair(self) -> None:
        evidence = [
            {
                "page_content": (
                    "Film A was directed by Director A. "
                    "Director A died in 2000."
                ),
                "metadata": {"title": "Film A"},
            },
            {
                "page_content": (
                    "Film B was directed by Director B. "
                    "Director B died in 1999."
                ),
                "metadata": {"title": "Film B"},
            },
        ]
        with (
            patch("farr_star.pipeline.FARRV2") as farr_class,
            patch(
                "farr_star.pipeline.ircot",
                return_value=expert("Film B", evidence),
            ),
        ):
            farr_class.return_value.answer.return_value = expert(
                "Film A",
                evidence,
            )
            result = FARRSTAR(
                lambda query, top_k: evidence[:top_k],
                ComparisonRepairLLM(),
                star_config=STARConfig(
                    enable_contract_retrieval=False,
                    enable_repair=True,
                    verify_repair=True,
                ),
            ).answer(
                "Which film has the director who died earlier, "
                "Film A or Film B?"
            )

        self.assertEqual(result.answer, "Film B")
        self.assertEqual(result.stats.route, "evidence_repair")
        self.assertEqual(result.stats.selected_expert, "synthesized")

    def test_ambiguous_judgment_keeps_ircot_default(self) -> None:
        evidence = [
            {
                "page_content": "The evidence mentions both candidates.",
                "metadata": {"title": "Evidence"},
            }
        ]
        with (
            patch("farr_star.pipeline.FARRV2") as farr_class,
            patch(
                "farr_star.pipeline.ircot",
                return_value=expert("IRCoT answer", evidence),
            ),
        ):
            farr_class.return_value.answer.return_value = expert(
                "FARR answer",
                evidence,
            )
            result = FARRSTAR(
                lambda query, top_k: evidence[:top_k],
                ConservativeGateLLM(),
                star_config=STARConfig(
                    enable_contract_retrieval=False,
                    enable_repair=True,
                ),
            ).answer("What entity satisfies the second relation?")

        self.assertEqual(result.answer, "IRCoT answer")
        self.assertEqual(result.stats.route, "ircot_default")

    def test_local_evidence_chain_validation(self) -> None:
        evidence = [
            {
                "page_content": "Film A was directed by Director A.",
                "metadata": {"title": "Film A"},
            },
            {
                "page_content": "Director A was born in 1930.",
                "metadata": {"title": "Director A"},
            },
        ]
        solution = {
            "answer": "Film A",
            "target_match": True,
            "complete": True,
            "steps": [
                {
                    "claim": "Film A director",
                    "doc_id": 1,
                    "quote": "Film A was directed by Director A",
                },
                {
                    "claim": "Director A birth year",
                    "doc_id": 2,
                    "quote": "Director A was born in 1930",
                },
            ],
        }
        contract = QuestionContract(
            reasoning_type="comparison_chain",
            answer_type="work",
            expected_hops=3,
            allowed_answers=["Film A", "Film B"],
        )
        valid, steps, _ = validate_solution(
            solution,
            contract,
            evidence,
            min_quote_words=3,
        )
        self.assertTrue(valid)
        self.assertEqual(len(steps), 2)


if __name__ == "__main__":
    unittest.main()
