from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from farr_star.eva_selector import EvidenceVerifiedAbstainingSelector
from farr_star.evidence_verifier import (
    VerifierConfig,
    candidate_claims,
    evidence_units,
    extract_question_features,
)


class IdentityScaler:
    def transform(self, values):
        return np.asarray(values, dtype=float)


class FakeEvidenceScorer:
    entailment_index = 0
    contradiction_index = 2

    def rerank(self, pairs):
        return np.linspace(0.55, 0.95, len(pairs), dtype=float)

    def nli_scores(self, pairs):
        result = []
        for premise, hypothesis in pairs:
            supported = any(
                token.lower() in premise.lower()
                for token in hypothesis.split()
                if len(token) > 5
            )
            result.append(
                [
                    0.8 if supported else 0.25,
                    0.15,
                    0.05 if supported else 0.6,
                ]
            )
        return np.asarray(result, dtype=float)


class FARREVATests(unittest.TestCase):
    def test_claim_and_evidence_preparation_is_bounded(self) -> None:
        claims = candidate_claims(
            question="Where was the author born?",
            answer="London",
            trace="The author was born in London. ||| London is in England.",
            max_trace_claims=1,
        )
        self.assertEqual(claims[0].kind, "answer")
        self.assertEqual(len(claims), 2)
        units = evidence_units(
            [
                {
                    "page_content": (
                        "The author was born in London. London is in England."
                    ),
                    "metadata": {"title": "Biography"},
                }
            ]
        )
        self.assertTrue(units)
        self.assertTrue(all(len(value.text) <= 850 for value in units))

    def test_extracted_features_exclude_identity_and_runtime_fields(self) -> None:
        candidates = [
            {
                "dataset": "hotpotqa",
                "question_id": "q1",
                "source_split": "train",
                "experiment_split": "train",
                "method": method,
                "question": "Where was the author born?",
                "prediction": answer,
                "intermediate_answers": trace,
            }
            for method, answer, trace in (
                ("flare-embedded", "London", "The author was born in London."),
                ("ircot", "Paris", "The author was born in Paris."),
                ("farr", "London", "The author was born in London."),
            )
        ]
        rows = extract_question_features(
            question="Where was the author born?",
            candidates=candidates,
            documents=[
                {
                    "page_content": "The author was born in London in 1901.",
                    "metadata": {"title": "Biography"},
                },
                {
                    "page_content": "Paris is the capital of France.",
                    "metadata": {"title": "Paris"},
                },
            ],
            scorer=FakeEvidenceScorer(),
            config=VerifierConfig(lexical_top_k=2, neural_top_k=1),
        )
        self.assertEqual(len(rows), 3)
        forbidden = {
            "method",
            "llm_calls",
            "retrieval_calls",
            "verification",
            "selected_internal",
        }
        for row in rows:
            self.assertFalse(forbidden & set(row["features"]))
            self.assertIn("answer_entail_max", row["features"])
            self.assertIn("proof_margin_min", row["features"])

    def test_abstention_defaults_to_farr_until_margin_is_sufficient(self) -> None:
        rows = [
            {"method": method, "features": {"score": score}}
            for method, score in (
                ("flare-embedded", 0.3),
                ("ircot", 0.1),
                ("farr", 0.2),
            )
        ]
        model = SimpleNamespace(
            coef_=np.asarray([[1.0]]),
            intercept_=np.asarray([0.0]),
        )
        conservative = EvidenceVerifiedAbstainingSelector(
            feature_names=["score"],
            scaler=IdentityScaler(),
            model=model,
            switch_threshold=0.70,
        )
        selected, probability, _, switched = conservative.choose(rows)
        self.assertEqual(selected, "farr")
        self.assertFalse(switched)
        self.assertLess(probability, 0.70)

        permissive = EvidenceVerifiedAbstainingSelector(
            feature_names=["score"],
            scaler=IdentityScaler(),
            model=model,
            switch_threshold=0.50,
        )
        selected, _, _, switched = permissive.choose(rows)
        self.assertEqual(selected, "flare-embedded")
        self.assertTrue(switched)


if __name__ == "__main__":
    unittest.main()
