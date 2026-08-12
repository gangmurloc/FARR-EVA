from __future__ import annotations

import unittest
from unittest.mock import patch

from farr_star.odr_pipeline import FARRODR
from farr_star.oracle_router import (
    feature_from_rows,
    feature_from_runtime,
)
from farr.types import FARRResult, FARRStats


def result(answer: str, flare: str = "") -> FARRResult:
    stats = FARRStats(
        llm_calls=2,
        retrieval_calls=1,
        completed_hops=2,
        final_verification_label="SUPPORTED",
        selected_candidate="FLARE",
        candidate_answers={"rag": "RAG answer", "flare": flare},
    )
    return FARRResult(
        answer=answer,
        evidence=[
            {
                "page_content": "Evidence.",
                "metadata": {"title": "Evidence"},
            }
        ],
        stats=stats,
    )


class FakeRouter:
    def choose(self, feature: str):
        self.feature = feature
        return (
            "flare",
            0.8,
            {"flare": 0.8, "ircot": 0.15, "farr-v2": 0.05},
        )


class ODRTests(unittest.TestCase):
    def test_runtime_and_training_features_match(self) -> None:
        farr = result("FARR answer", "FLARE answer")
        ircot = result("IRCoT answer")
        runtime = feature_from_runtime(
            "Who answered the question?",
            "FLARE answer",
            farr,
            ircot,
        )
        rows = {
            "flare": {
                "question": "Who answered the question?",
                "prediction": "FLARE answer",
            },
            "ircot": {
                "prediction": "IRCoT answer",
                "completed_hops": 2,
            },
            "farr-v2": {
                "question": "Who answered the question?",
                "prediction": "FARR answer",
                "final_verification_label": "SUPPORTED",
                "selected_candidate": "FLARE",
                "rag_candidate": "RAG answer",
                "flare_candidate": "FLARE answer",
                "completed_hops": 2,
            },
        }
        self.assertEqual(runtime, feature_from_rows(rows))

    def test_odr_can_select_internal_flare_without_extra_expert(self) -> None:
        farr = result("FARR answer", "FLARE answer")
        ircot_result = result("IRCoT answer")
        router = FakeRouter()
        with (
            patch("farr_star.odr_pipeline.FARRV2") as farr_class,
            patch(
                "farr_star.odr_pipeline.ircot",
                return_value=ircot_result,
            ),
        ):
            farr_class.return_value.answer.return_value = farr
            output = FARRODR(
                lambda query, top_k: [],
                lambda prompt: "",
                router,
            ).answer("Who answered the question?")

        self.assertEqual(output.answer, "FLARE answer")
        self.assertEqual(output.stats.selected_expert, "flare")
        self.assertEqual(output.stats.llm_calls, 4)
        self.assertEqual(output.stats.route, "oracle_distilled_router")


if __name__ == "__main__":
    unittest.main()

