from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .evidence_verifier import feature_vector


CANDIDATES = ("flare-embedded", "ircot", "farr")
ANCHOR = "farr"


@dataclass
class EvidenceVerifiedAbstainingSelector:
    # Legacy serialization name retained so the pre-Test-C locked joblib
    # artifact remains loadable and hash-identical.  The manuscript-facing
    # method name is Evidence-Vector Arbitration; the implemented threshold
    # is an anchored utility comparison, not selective abstention.
    feature_names: list[str]
    scaler: StandardScaler
    model: LogisticRegression
    switch_threshold: float
    anchor: str = ANCHOR

    def utilities(
        self,
        rows: Sequence[dict[str, Any]],
    ) -> dict[str, float]:
        methods = {str(row["method"]) for row in rows}
        if methods != set(CANDIDATES):
            raise ValueError(f"Exact candidate pool required, got {methods}")
        matrix = np.vstack(
            [feature_vector(row, self.feature_names) for row in rows]
        )
        scaled = self.scaler.transform(matrix)
        intercept = float(
            np.asarray(self.model.intercept_, dtype=float).reshape(-1)[0]
        )
        values = np.asarray(
            scaled @ self.model.coef_[0] + intercept
        ).reshape(-1)
        return {
            str(row["method"]): float(value)
            for row, value in zip(rows, values)
        }

    def choose(
        self,
        rows: Sequence[dict[str, Any]],
    ) -> tuple[str, float, dict[str, float], bool]:
        utilities = self.utilities(rows)
        alternatives = [method for method in CANDIDATES if method != self.anchor]
        best_alternative = max(alternatives, key=utilities.get)
        margin = utilities[best_alternative] - utilities[self.anchor]
        probability = 1.0 / (1.0 + math.exp(-max(min(margin, 40.0), -40.0)))
        switched = probability >= self.switch_threshold
        selected = best_alternative if switched else self.anchor
        return selected, probability, utilities, switched

    def save(self, path: str, metadata: dict[str, Any]) -> None:
        joblib.dump(
            {"selector": self, "metadata": dict(metadata)},
            path,
        )

    @classmethod
    def load(cls, path: str) -> tuple["EvidenceVerifiedAbstainingSelector", dict[str, Any]]:
        artifact = joblib.load(path)
        selector = artifact.get("selector")
        if not isinstance(selector, cls):
            raise TypeError("Artifact is not an EVA selector.")
        return selector, dict(artifact.get("metadata", {}))


def fit_pairwise_selector(
    rows: Sequence[dict[str, Any]],
    targets: dict[tuple[str, str, str], float],
    *,
    feature_names: Sequence[str],
    c_value: float,
    random_state: int = 42,
) -> tuple[StandardScaler, LogisticRegression, dict[str, Any]]:
    matrix = np.vstack([feature_vector(row, feature_names) for row in rows])
    scaler = StandardScaler().fit(matrix)
    scaled = scaler.transform(matrix)
    groups: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(rows):
        key = (str(row["dataset"]), str(row["question_id"]))
        groups.setdefault(key, []).append(index)
    dataset_pairs: dict[str, int] = {}
    raw_pairs = []
    for key, indexes in groups.items():
        if len(indexes) != len(CANDIDATES):
            raise ValueError(f"Incomplete training feature group: {key}")
        for left, right in combinations(indexes, 2):
            left_row = rows[left]
            right_row = rows[right]
            left_key = (*key, str(left_row["method"]))
            right_key = (*key, str(right_row["method"]))
            delta = float(targets[left_key]) - float(targets[right_key])
            if abs(delta) < 1e-12:
                continue
            winner, loser = (left, right) if delta > 0 else (right, left)
            dataset_pairs[key[0]] = dataset_pairs.get(key[0], 0) + 1
            raw_pairs.append((winner, loser, abs(delta), key[0]))
    if len(raw_pairs) < 100:
        raise RuntimeError("Not enough informative evidence candidate pairs.")
    differences = []
    labels = []
    weights = []
    for winner, loser, delta, dataset in raw_pairs:
        difference = scaled[winner] - scaled[loser]
        weight = delta / dataset_pairs[dataset]
        differences.extend((difference, -difference))
        labels.extend((1, 0))
        weights.extend((weight, weight))
    pair_matrix = np.vstack(differences)
    weight_array = np.asarray(weights, dtype=float)
    weight_array *= len(weight_array) / weight_array.sum()
    model = LogisticRegression(
        C=float(c_value),
        fit_intercept=False,
        solver="liblinear",
        max_iter=5000,
        random_state=random_state,
    )
    model.fit(
        pair_matrix,
        np.asarray(labels, dtype=int),
        sample_weight=weight_array,
    )
    metadata = {
        "question_count": len(groups),
        "candidate_count": len(rows),
        "informative_pairs": len(raw_pairs),
        "dataset_pairs": dataset_pairs,
        "c_value": float(c_value),
        "random_state": random_state,
        "converged": bool(model.n_iter_[0] < model.max_iter),
        "iterations": int(model.n_iter_[0]),
    }
    return scaler, model, metadata
