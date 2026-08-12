from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import compat  # noqa: F401
from .contracts import answer_contract_compliance, heuristic_contract
from .oracle_router import CANDIDATES, answer_shape, normalize_answer
from farr.documents import content_tokens, doc_text, doc_title
from farr.types import Document


@dataclass
class CandidateObservation:
    question_id: str
    dataset: str
    question: str
    method: str
    answer: str
    row: dict[str, Any]
    peer_answers: dict[str, str]
    evidence: list[Document]
    target_f1: float = 0.0

    @property
    def group_key(self) -> tuple[str, str]:
        return self.dataset, self.question_id


def token_f1(left: str, right: str) -> float:
    left_tokens = normalize_answer(left).split()
    right_tokens = normalize_answer(right).split()
    if not left_tokens or not right_tokens:
        return float(left_tokens == right_tokens)
    overlap = sum(
        (Counter(left_tokens) & Counter(right_tokens)).values()
    )
    if not overlap:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall)


def contains_normalized_phrase(phrase: str, text: str) -> bool:
    normalized_phrase = normalize_answer(phrase)
    normalized_text = normalize_answer(text)
    if not normalized_phrase or not normalized_text:
        return False
    return (
        f" {normalized_phrase} "
        in f" {normalized_text} "
    )


def candidate_text(observation: CandidateObservation) -> str:
    contract = heuristic_contract(observation.question)
    return "\n".join(
        (
            f"question {observation.question}",
            f"candidate {observation.answer}",
            f"expert __{observation.method}__",
            f"reasoning __{contract.reasoning_type}__",
            f"answer_type __{contract.answer_type}__",
            (
                "verification __"
                f"{str(observation.row.get('final_verification_label', '')).lower()}"
                "__"
            ),
            (
                "selected_internal __"
                f"{str(observation.row.get('selected_candidate', '')).lower()}"
                "__"
            ),
        )
    )


def candidate_features(
    observation: CandidateObservation,
) -> dict[str, float]:
    answer = normalize_answer(observation.answer)
    answer_tokens = set(answer.split())
    question_tokens = content_tokens(observation.question)
    contract = heuristic_contract(observation.question)
    contract_ok, _ = answer_contract_compliance(
        observation.answer,
        contract,
    )

    exact_documents = 0
    exact_titles = 0
    max_document_coverage = 0.0
    max_title_coverage = 0.0
    max_joint_relevance = 0.0
    union_tokens: set[str] = set()
    for document in observation.evidence:
        text = normalize_answer(doc_text(document))
        title = normalize_answer(doc_title(document))
        document_tokens = set(text.split())
        title_tokens = set(title.split())
        union_tokens |= document_tokens | title_tokens
        exact_documents += int(
            contains_normalized_phrase(answer, text)
        )
        exact_titles += int(
            contains_normalized_phrase(answer, title)
        )
        coverage = (
            len(answer_tokens & document_tokens) / len(answer_tokens)
            if answer_tokens
            else 0.0
        )
        title_coverage = (
            len(answer_tokens & title_tokens) / len(answer_tokens)
            if answer_tokens
            else 0.0
        )
        relevance = (
            len(question_tokens & (document_tokens | title_tokens))
            / max(len(question_tokens), 1)
        )
        max_document_coverage = max(max_document_coverage, coverage)
        max_title_coverage = max(max_title_coverage, title_coverage)
        max_joint_relevance = max(
            max_joint_relevance,
            coverage * relevance,
        )

    peer_values = [
        value
        for method, value in observation.peer_answers.items()
        if method != observation.method
    ]
    agreements = sum(
        normalize_answer(value) == answer for value in peer_values
    )
    peer_f1 = [
        token_f1(observation.answer, value) for value in peer_values
    ]
    trace = str(observation.row.get("intermediate_answers", ""))
    normalized_trace = normalize_answer(trace)
    union_coverage = (
        len(answer_tokens & union_tokens) / len(answer_tokens)
        if answer_tokens
        else 0.0
    )
    result: dict[str, float] = {
        f"method={observation.method}": 1.0,
        f"shape={answer_shape(observation.answer)}": 1.0,
        f"reasoning={contract.reasoning_type}": 1.0,
        f"answer_type={contract.answer_type}": 1.0,
        (
            "verification="
            f"{str(observation.row.get('final_verification_label', '')).lower()}"
        ): 1.0,
        (
            "selected_internal="
            f"{str(observation.row.get('selected_candidate', '')).lower()}"
        ): 1.0,
        "answer_tokens": float(len(answer.split())),
        "answer_chars_log": math.log1p(len(str(observation.answer))),
        "contract_ok": float(contract_ok),
        "exact_document_count": float(exact_documents),
        "exact_title_count": float(exact_titles),
        "max_document_coverage": max_document_coverage,
        "max_title_coverage": max_title_coverage,
        "union_coverage": union_coverage,
        "max_joint_relevance": max_joint_relevance,
        "peer_exact_agreements": float(agreements),
        "peer_max_f1": max(peer_f1, default=0.0),
        "peer_mean_f1": (
            sum(peer_f1) / len(peer_f1) if peer_f1 else 0.0
        ),
        "trace_exact": float(
            contains_normalized_phrase(answer, normalized_trace)
        ),
        "llm_calls": float(observation.row.get("llm_calls", 0)),
        "retrieval_calls": float(
            observation.row.get("retrieval_calls", 0)
        ),
        "completed_hops": float(
            observation.row.get("completed_hops", 0)
        ),
        "revision_applied": float(
            observation.row.get("revision_applied", 0)
        ),
    }
    return result


class EvidencePairwiseRanker:
    def __init__(
        self,
        c_value: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.c_value = float(c_value)
        self.random_state = int(random_state)
        self.text_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000,
            sublinear_tf=True,
        )
        self.numeric_vectorizer = DictVectorizer(sparse=True)
        self.model = LogisticRegression(
            C=self.c_value,
            fit_intercept=False,
            solver="liblinear",
            dual=True,
            max_iter=2000,
            random_state=self.random_state,
        )

    def _fit_candidate_matrix(
        self,
        observations: Sequence[CandidateObservation],
    ) -> sparse.csr_matrix:
        text = self.text_vectorizer.fit_transform(
            [candidate_text(value) for value in observations]
        )
        numeric = self.numeric_vectorizer.fit_transform(
            [candidate_features(value) for value in observations]
        )
        return sparse.hstack([text, numeric], format="csr")

    def _candidate_matrix(
        self,
        observations: Sequence[CandidateObservation],
    ) -> sparse.csr_matrix:
        text = self.text_vectorizer.transform(
            [candidate_text(value) for value in observations]
        )
        numeric = self.numeric_vectorizer.transform(
            [candidate_features(value) for value in observations]
        )
        return sparse.hstack([text, numeric], format="csr")

    def fit(
        self,
        observations: Sequence[CandidateObservation],
    ) -> "EvidencePairwiseRanker":
        matrix = self._fit_candidate_matrix(observations)
        groups: dict[
            tuple[str, str], list[int]
        ] = defaultdict(list)
        for index, observation in enumerate(observations):
            groups[observation.group_key].append(index)

        dataset_pairs = Counter()
        raw_pairs = []
        for indexes in groups.values():
            for left, right in combinations(indexes, 2):
                delta = (
                    observations[left].target_f1
                    - observations[right].target_f1
                )
                if delta == 0.0:
                    continue
                winner, loser = (
                    (left, right) if delta > 0 else (right, left)
                )
                dataset = observations[winner].dataset
                dataset_pairs[dataset] += 1
                raw_pairs.append(
                    (winner, loser, abs(delta), dataset)
                )
        if len(raw_pairs) < 100:
            raise RuntimeError("Not enough informative candidate pairs.")

        differences = []
        labels = []
        weights = []
        for winner, loser, delta, dataset in raw_pairs:
            difference = matrix[winner] - matrix[loser]
            weight = delta / dataset_pairs[dataset]
            differences.extend((difference, -difference))
            labels.extend((1, 0))
            weights.extend((weight, weight))
        pair_matrix = sparse.vstack(differences, format="csr")
        weight_array = np.asarray(weights, dtype=float)
        weight_array *= len(weight_array) / weight_array.sum()
        self.model.fit(
            pair_matrix,
            np.asarray(labels),
            sample_weight=weight_array,
        )
        self.training_metadata = {
            "candidate_count": len(observations),
            "question_count": len(groups),
            "informative_pairs": len(raw_pairs),
            "dataset_pairs": dict(dataset_pairs),
            "random_state": self.random_state,
        }
        return self

    def utilities(
        self,
        observations: Sequence[CandidateObservation],
    ) -> np.ndarray:
        matrix = self._candidate_matrix(observations)
        return np.asarray(matrix @ self.model.coef_[0]).ravel()

    def choose(
        self,
        observations: Sequence[CandidateObservation],
    ) -> tuple[str, float, dict[str, float]]:
        if {value.method for value in observations} != set(CANDIDATES):
            raise ValueError("One observation per candidate is required.")
        utilities = self.utilities(observations)
        scores = {
            observation.method: float(score)
            for observation, score in zip(observations, utilities)
        }
        ranked = sorted(scores, key=scores.get, reverse=True)
        gap = scores[ranked[0]] - scores[ranked[1]]
        confidence = 1.0 / (1.0 + math.exp(-gap))
        return ranked[0], confidence, scores


class EvidencePairwiseRouter:
    def __init__(self, artifact_path: str | Path) -> None:
        artifact = joblib.load(artifact_path)
        self.ranker: EvidencePairwiseRanker = artifact["ranker"]
        self.metadata = dict(artifact.get("metadata", {}))

    def choose(
        self,
        observations: Sequence[CandidateObservation],
    ) -> tuple[str, float, dict[str, float]]:
        return self.ranker.choose(observations)
