from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib


@lru_cache(maxsize=4)
def _load_artifact(artifact_path: str) -> Dict[str, Any]:
    return joblib.load(Path(artifact_path))


def model_family(model_name: str) -> str:
    name = str(model_name).lower()
    if "qwen" in name:
        return "qwen"
    if "llama" in name:
        return "llama"
    return "other"


def infer_hotpot_type(question: str) -> str:
    q = " ".join(str(question).lower().split())
    markers = (
        " both ",
        " same ",
        " older",
        " younger",
        " higher",
        " lower",
        " more ",
        " less ",
        "which of",
        "which one",
    )
    if q.startswith(("are ", "were ", "is ", "was ")) or any(
        marker in f" {q} " for marker in markers
    ):
        return "comparison"
    return "bridge"


def selector_feature_text(
    model_name: str,
    question: str,
    rag_answer: str,
    flare_answer: str,
    question_type: str | None = None,
    level: str = "hard",
) -> str:
    family = model_family(model_name)
    qtype = question_type or infer_hotpot_type(question)
    return (
        f"MODEL={family} TYPE={qtype} LEVEL={level} "
        f"QUESTION {question} "
        f"CANDIDATE_A {rag_answer} "
        f"CANDIDATE_B {flare_answer} "
        f"LEN_A={len(rag_answer.split())} LEN_B={len(flare_answer.split())}"
    )


class CandidateSelector:
    def __init__(self, artifact_path: str) -> None:
        artifact = _load_artifact(str(Path(artifact_path).resolve()))
        self.model = artifact["model"]
        self.threshold = float(artifact["threshold"])
        self.metadata: Dict[str, Any] = {
            key: value for key, value in artifact.items() if key != "model"
        }

    def choose(
        self,
        model_name: str,
        question: str,
        rag_answer: str,
        flare_answer: str,
    ) -> Tuple[str, float]:
        text = selector_feature_text(
            model_name=model_name,
            question=question,
            rag_answer=rag_answer,
            flare_answer=flare_answer,
        )
        probabilities = self.model.predict_proba([text])[0]
        by_label = {
            label: float(probability)
            for label, probability in zip(self.model.classes_, probabilities)
        }
        choice = max(by_label, key=by_label.get)
        return choice, by_label[choice]
