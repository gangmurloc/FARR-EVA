from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Any

import joblib


CANDIDATES = ("flare", "ircot", "farr-v2")


def normalize_answer(text: str) -> str:
    value = str(text).lower()
    value = "".join(
        character
        for character in value
        if character not in string.punctuation
    )
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def answer_shape(answer: str) -> str:
    value = normalize_answer(answer)
    if value in {"yes", "no"}:
        return "boolean"
    if re.search(r"\d", value):
        return "numeric"
    return "short" if len(value.split()) <= 5 else "long"


def question_shape(question: str) -> str:
    value = " ".join(str(question).lower().split())
    if value.startswith(
        (
            "is ", "are ", "was ", "were ", "do ", "does ", "did ",
            "has ", "have ", "had ", "can ", "could ", "will ", "would ",
        )
    ):
        return "boolean"
    if " or " in value:
        return "comparison"
    return value.split()[0] if value.split() else "other"


def router_feature_text(
    *,
    question: str,
    flare_answer: str,
    ircot_answer: str,
    farr_answer: str,
    farr_label: str,
    farr_selected: str,
    rag_internal: str,
    flare_internal: str,
    farr_hops: int,
    ircot_hops: int,
) -> str:
    answers = {
        "flare": str(flare_answer),
        "ircot": str(ircot_answer),
        "farr-v2": str(farr_answer),
    }
    agreements = []
    for left, right in (
        ("flare", "ircot"),
        ("flare", "farr-v2"),
        ("ircot", "farr-v2"),
    ):
        if (
            normalize_answer(answers[left])
            == normalize_answer(answers[right])
        ):
            agreements.append(f"{left}_{right}")

    return "\n".join(
        (
            f"question {question}",
            f"qshape __{question_shape(question)}__",
            f"flare_answer {answers['flare']}",
            f"ircot_answer {answers['ircot']}",
            f"farr_answer {answers['farr-v2']}",
            (
                f"flare_shape __{answer_shape(answers['flare'])}__ "
                f"ircot_shape __{answer_shape(answers['ircot'])}__ "
                f"farr_shape __{answer_shape(answers['farr-v2'])}__"
            ),
            f"agreements {' '.join(agreements) or 'none'}",
            f"farr_label __{str(farr_label).lower()}__",
            f"farr_selected __{str(farr_selected).lower()}__",
            f"rag_internal {rag_internal}",
            f"flare_internal {flare_internal}",
            f"farr_hops {int(farr_hops)} ircot_hops {int(ircot_hops)}",
        )
    )


def feature_from_rows(methods: dict[str, dict[str, Any]]) -> str:
    flare = methods["flare"]
    ircot = methods["ircot"]
    farr = methods["farr-v2"]
    return router_feature_text(
        question=str(farr.get("question", "")),
        flare_answer=str(flare.get("prediction", "")),
        ircot_answer=str(ircot.get("prediction", "")),
        farr_answer=str(farr.get("prediction", "")),
        farr_label=str(farr.get("final_verification_label", "")),
        farr_selected=str(farr.get("selected_candidate", "")),
        rag_internal=str(farr.get("rag_candidate", "")),
        flare_internal=str(farr.get("flare_candidate", "")),
        farr_hops=int(farr.get("completed_hops", 0)),
        ircot_hops=int(ircot.get("completed_hops", 0)),
    )


def feature_from_runtime(
    question: str,
    flare_answer: str,
    farr_result: Any,
    ircot_result: Any,
) -> str:
    farr_stats = farr_result.stats
    return router_feature_text(
        question=question,
        flare_answer=flare_answer,
        ircot_answer=str(ircot_result.answer),
        farr_answer=str(farr_result.answer),
        farr_label=str(farr_stats.final_verification_label),
        farr_selected=str(farr_stats.selected_candidate),
        rag_internal=str(farr_stats.candidate_answers.get("rag", "")),
        flare_internal=str(
            farr_stats.candidate_answers.get("flare", flare_answer)
        ),
        farr_hops=int(farr_stats.completed_hops),
        ircot_hops=int(ircot_result.stats.completed_hops),
    )


class OracleDistilledRouter:
    def __init__(self, artifact_path: str | Path) -> None:
        artifact = joblib.load(artifact_path)
        self.model = artifact["model"]
        self.metadata = dict(artifact.get("metadata", {}))
        candidates = tuple(
            self.metadata.get("candidates", CANDIDATES)
        )
        if set(candidates) != set(CANDIDATES):
            raise ValueError(
                f"Router candidates do not match runtime: {candidates}"
            )

    def choose(
        self,
        feature: str,
    ) -> tuple[str, float, dict[str, float]]:
        probabilities = self.model.predict_proba([feature])[0]
        classes = list(self.model.classes_)
        scores = {
            str(label): float(probability)
            for label, probability in zip(classes, probabilities)
        }
        selected = max(scores, key=scores.get)
        return selected, scores[selected], scores

