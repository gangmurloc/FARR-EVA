from __future__ import annotations

import unittest
from unittest.mock import patch

from farr_star.epr_pipeline import FARREPR
from farr_star.evidence_ranker import (
    CandidateObservation,
    candidate_features,
    contains_normalized_phrase,
)
from farr.types import FARRResult, FARRStats, HopTrace


def result(answer: str, flare: str = "") -> FARRResult:
    stats = FARRStats(
        llm_calls=10,
        retrieval_calls=7,
        planned_hops=3,
        completed_hops=3,
        final_verification_label="SUPPORTED",
        selected_candidate="FLARE",
        candidate_answers={"rag": "RAG answer", "flare": flare},
        hop_traces=[
            HopTrace(
                hop=index,
                subquestion="step",
                intermediate_answer=f"fact {index}",
            )
            for index in range(1, 4)
        ],
    )
    return FARRResult(
        answer=answer,
        evidence=[
            {
                "page_content": "Ada Lovelace wrote the first algorithm.",
                "metadata": {"title": "Ada Lovelace"},
            }
        ],
        stats=stats,
    )


class FakeRetriever:
    def __init__(self) -> None:
        self.documents = [
            {
                "page_content": "Ada Lovelace wrote the first algorithm.",
                "metadata": {"title": "Ada Lovelace"},
            }
        ]

    def __call__(self, query: str, top_k: int):
        return self.documents[:top_k]


class FakeRouter:
    def choose(self, observations):
        self.observations = observations
        return (
            "flare",
            0.75,
            {"flare": 1.0, "ircot": 0.2, "farr-v2": 0.1},
        )


class FakeFARRRouter:
    def choose(self, observations):
        return (
            "farr-v2",
            0.75,
            {"flare": 0.1, "ircot": 0.2, "farr-v2": 1.0},
        )


class EPRTests(unittest.TestCase):
    def test_exact_evidence_match_respects_token_boundaries(self) -> None:
        self.assertFalse(contains_normalized_phrase("no", "known fact"))
        self.assertTrue(contains_normalized_phrase("no", "The answer is no."))

    def test_evidence_features_reward_grounded_answer(self) -> None:
        observation = CandidateObservation(
            question_id="q1",
            dataset="test",
            question="Who wrote the first algorithm?",
            method="flare",
            answer="Ada Lovelace",
            row={},
            peer_answers={
                "flare": "Ada Lovelace",
                "ircot": "Charles Babbage",
                "farr-v2": "Ada Lovelace",
            },
            evidence=FakeRetriever().documents,
        )
        features = candidate_features(observation)
        self.assertEqual(features["exact_document_count"], 1.0)
        self.assertEqual(features["exact_title_count"], 1.0)
        self.assertEqual(features["union_coverage"], 1.0)
        self.assertEqual(features["peer_exact_agreements"], 1.0)

    def test_epr_selects_internal_flare_without_extra_flare_run(self) -> None:
        farr = result("FARR answer", "Ada Lovelace")
        ircot_result = result("IRCoT answer")
        router = FakeRouter()
        with (
            patch("farr_star.epr_pipeline.FARRV2") as farr_class,
            patch(
                "farr_star.epr_pipeline.ircot",
                return_value=ircot_result,
            ),
        ):
            farr_class.return_value.answer.return_value = farr
            output = FARREPR(
                FakeRetriever(),
                lambda prompt: "",
                router,
            ).answer("Who wrote the first algorithm?")

        self.assertEqual(output.answer, "Ada Lovelace")
        self.assertEqual(output.stats.selected_expert, "flare")
        self.assertEqual(output.stats.route, "evidence_pairwise_ranker")
        self.assertEqual(output.stats.llm_calls, 20)
        flare = next(
            value
            for value in router.observations
            if value.method == "flare"
        )
        self.assertEqual(flare.row["llm_calls"], 7)
        self.assertEqual(flare.row["retrieval_calls"], 4)
        self.assertEqual(len(router.observations), 3)

    def test_legacy_internal_key_is_reported_as_farr(self) -> None:
        farr = result("FARR answer", "FLARE answer")
        with (
            patch("farr_star.epr_pipeline.FARRV2") as farr_class,
            patch(
                "farr_star.epr_pipeline.ircot",
                return_value=result("IRCoT answer"),
            ),
        ):
            farr_class.return_value.answer.return_value = farr
            output = FARREPR(
                FakeRetriever(),
                lambda prompt: "",
                FakeFARRRouter(),
            ).answer("Who answered the question?")

        self.assertEqual(output.answer, "FARR answer")
        self.assertEqual(output.stats.selected_expert, "farr")
        self.assertIn("farr=", output.stats.router_probabilities)
        self.assertNotIn("farr-v2", output.stats.router_probabilities)


if __name__ == "__main__":
    unittest.main()
