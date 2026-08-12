#!/usr/bin/env python3
"""Locked one-shot Test-C evaluation for FARR-EVA."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from farr_star.eva_selector import (
    CANDIDATES,
    EvidenceVerifiedAbstainingSelector,
)
from run_benchmark import summarize


ROOT = Path(__file__).resolve().parent
BASE_METHODS = ("rag", "flare", "rarr", "ircot", "farr", "flare-embedded")
DATASETS = ("hotpotqa", "2wiki", "musique")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-inputs", nargs="+", required=True)
    parser.add_argument("--feature-inputs", nargs="+", required=True)
    parser.add_argument("--feature-metadata", nargs="+")
    parser.add_argument(
        "--test-lock",
        default="data/farr_eva_test_c.lock.json",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--output-dir",
        default="outputs/farr_eva_test_c/merged",
    )
    parser.add_argument(
        "--report",
        default="reports/farr_eva_test_c/primary_statistics.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_rows(paths: Sequence[Path]) -> list[dict[str, Any]]:
    result = []
    for path in paths:
        result.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return result


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    count = len(ordered)
    result = {}
    running = 0.0
    for index, name in enumerate(ordered):
        value = min(1.0, (count - index) * p_values[name])
        running = max(running, value)
        result[name] = running
    return result


def bootstrap(
    differences: dict[tuple[str, str], float],
    *,
    reps: int,
    seed: int,
    stratified: bool,
) -> tuple[list[float], float]:
    by_dataset = {
        dataset: np.asarray(
            [value for key, value in differences.items() if key[0] == dataset],
            dtype=float,
        )
        for dataset in DATASETS
        if any(key[0] == dataset for key in differences)
    }
    rng = np.random.default_rng(seed)
    samples = np.empty(reps, dtype=float)
    for start in range(0, reps, 500):
        count = min(500, reps - start)
        if stratified:
            means = []
            for values in by_dataset.values():
                indexes = rng.integers(
                    0, len(values), size=(count, len(values))
                )
                means.append(values[indexes].mean(axis=1))
            samples[start:start + count] = np.vstack(means).mean(axis=0)
        else:
            values = next(iter(by_dataset.values()))
            indexes = rng.integers(0, len(values), size=(count, len(values)))
            samples[start:start + count] = values[indexes].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    tail = 2.0 * min(
        float(np.mean(samples <= 0.0)),
        float(np.mean(samples >= 0.0)),
    )
    return [float(low), float(high)], max(tail, 1.0 / reps)


def main() -> None:
    args = parse_args()
    candidate_paths = [ROOT / value for value in args.candidate_inputs]
    feature_paths = [ROOT / value for value in args.feature_inputs]
    feature_metadata_paths = [
        ROOT / value
        for value in (
            args.feature_metadata
            or [f"{value}.meta.json" for value in args.feature_inputs]
        )
    ]
    output_dir = ROOT / args.output_dir
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    predictions_path = output_dir / "eva_predictions.jsonl"
    report_path = ROOT / args.report
    outputs = (results_path, summary_path, predictions_path, report_path)
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise FileExistsError("Test-C analysis output exists; use --overwrite.")

    test_lock_path = ROOT / args.test_lock
    lock = json.loads(test_lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_BEFORE_TEST_C_INFERENCE":
        raise RuntimeError("Invalid Test-C lock status.")
    manifest_path = ROOT / str(lock["manifest"])
    artifact_path = ROOT / str(lock["selector_artifact"])
    if sha256(manifest_path) != lock["manifest_sha256"]:
        raise RuntimeError("Test-C manifest changed after locking.")
    if sha256(artifact_path) != lock["selector_artifact_sha256"]:
        raise RuntimeError("EVA artifact changed after locking.")
    selector, artifact_metadata = EvidenceVerifiedAbstainingSelector.load(
        str(artifact_path)
    )
    if not artifact_metadata.get("validation_success"):
        raise RuntimeError("Selector artifact did not pass validation.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        (dataset, str(question_id))
        for dataset, values in manifest["datasets"].items()
        for question_id in values["test"]
    }

    candidate_rows = load_rows(candidate_paths)
    candidate_groups: dict[
        tuple[str, str], dict[str, dict[str, Any]]
    ] = defaultdict(dict)
    for row in candidate_rows:
        key = (str(row["dataset"]), str(row["question_id"]))
        method = str(row["method"])
        if method in candidate_groups[key]:
            raise ValueError(f"Duplicate Test-C candidate: {key}/{method}")
        candidate_groups[key][method] = row
    if set(candidate_groups) != expected:
        raise RuntimeError(
            f"Candidate/manifest mismatch: missing={len(expected-set(candidate_groups))}, "
            f"extra={len(set(candidate_groups)-expected)}"
        )
    incomplete_candidates = {
        key: sorted(set(BASE_METHODS) - set(values))
        for key, values in candidate_groups.items()
        if set(values) != set(BASE_METHODS)
    }
    if incomplete_candidates:
        raise RuntimeError(
            f"Incomplete Test-C candidates: {len(incomplete_candidates)}"
        )

    feature_metadata = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in feature_metadata_paths
    ]
    for path, values in zip(feature_paths, feature_metadata):
        if values.get("status") != "COMPLETE":
            raise RuntimeError(f"Incomplete Test-C features: {path}")
        if sha256(path) != values.get("output_sha256"):
            raise RuntimeError(f"Test-C feature hash mismatch: {path}")
        if values.get("verifier_config") != artifact_metadata.get(
            "verifier_config"
        ):
            raise RuntimeError("Test-C verifier config differs from training.")
    feature_rows = load_rows(feature_paths)
    feature_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_features = set()
    for row in feature_rows:
        key = (
            str(row["dataset"]),
            str(row["question_id"]),
            str(row["method"]),
        )
        if key in seen_features:
            raise ValueError(f"Duplicate Test-C feature: {key}")
        seen_features.add(key)
        feature_groups[key[:2]].append(row)
    if set(feature_groups) != expected:
        raise RuntimeError("Test-C feature question set mismatch.")

    eva_rows = []
    prediction_rows = []
    differences: dict[tuple[str, str], float] = {}
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key in sorted(expected):
        selected, probability, utilities, switched = selector.choose(
            feature_groups[key]
        )
        selected_source = candidate_groups[key][selected]
        farr_source = candidate_groups[key]["farr"]
        row = dict(selected_source)
        row.update(
            {
                "method": "farr-eva",
                "selected_expert": selected,
                "route": "evidence_verified_abstaining_arbitration",
                "selector_confidence": probability,
                "selector_used": int(switched),
                "selection_reason": " ||| ".join(
                    f"{method}={utilities[method]:.6f}"
                    for method in CANDIDATES
                ),
            }
        )
        eva_rows.append(row)
        delta = float(row["f1"]) - float(farr_source["f1"])
        differences[key] = delta
        candidate_values = {
            method: float(candidate_groups[key][method]["f1"])
            for method in CANDIDATES
        }
        by_dataset[key[0]].append(
            {
                "eva_f1": float(row["f1"]),
                "eva_em": float(row["exact_match"]),
                "farr_f1": float(farr_source["f1"]),
                "farr_em": float(farr_source["exact_match"]),
                "oracle_f1": max(candidate_values.values()),
                "selected": selected,
                "switched": switched,
                **candidate_values,
            }
        )
        prediction_rows.append(
            {
                "dataset": key[0],
                "question_id": key[1],
                "selected": selected,
                "switch_probability": probability,
                "switched": switched,
                "utilities": utilities,
                "prediction": row["prediction"],
                "f1": row["f1"],
                "fixed_farr_f1": farr_source["f1"],
                "delta": delta,
            }
        )

    dataset_reports = {}
    raw_p_values = {}
    for index, dataset in enumerate(DATASETS):
        values = by_dataset[dataset]
        local_differences = {
            key: value for key, value in differences.items() if key[0] == dataset
        }
        ci95, p_value = bootstrap(
            local_differences,
            reps=args.bootstrap_reps,
            seed=args.seed + index,
            stratified=False,
        )
        raw_p_values[dataset] = p_value
        eva_f1 = float(np.mean([value["eva_f1"] for value in values]))
        farr_f1 = float(np.mean([value["farr_f1"] for value in values]))
        dataset_reports[dataset] = {
            "count": len(values),
            "farr_eva_f1": eva_f1,
            "fixed_farr_f1": farr_f1,
            "delta_f1": eva_f1 - farr_f1,
            "ci95": ci95,
            "p_value": p_value,
            "farr_eva_em": float(
                np.mean([value["eva_em"] for value in values])
            ),
            "fixed_farr_em": float(
                np.mean([value["farr_em"] for value in values])
            ),
            "exact_candidate_oracle_f1": float(
                np.mean([value["oracle_f1"] for value in values])
            ),
            "switch_rate": float(
                np.mean([value["switched"] for value in values])
            ),
            "selected_counts": dict(
                Counter(str(value["selected"]) for value in values)
            ),
            "candidate_baselines": {
                method: float(
                    np.mean([value[method] for value in values])
                )
                for method in CANDIDATES
            },
        }
    adjusted = holm(raw_p_values)
    for dataset, value in adjusted.items():
        dataset_reports[dataset]["holm_adjusted_p"] = value
    macro_eva = float(
        np.mean([value["farr_eva_f1"] for value in dataset_reports.values()])
    )
    macro_farr = float(
        np.mean([value["fixed_farr_f1"] for value in dataset_reports.values()])
    )
    macro_ci, macro_p = bootstrap(
        differences,
        reps=args.bootstrap_reps,
        seed=args.seed + 100,
        stratified=True,
    )
    success = macro_eva > macro_farr and macro_ci[0] > 0.0

    all_rows = candidate_rows + eva_rows
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            summarize(all_rows, [*BASE_METHODS, "farr-eva"]),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    predictions_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in prediction_rows
        ),
        encoding="utf-8",
    )
    report = {
        "schema": "farr-eva-test-c-primary-v1",
        "status": "PASS" if success else "FAIL_DO_NOT_REVERT_TO_OLD_MODEL",
        "question_count": len(expected),
        "primary": {
            "comparison": "FARR-EVA minus validation-fixed FARR",
            "farr_eva_macro_f1": macro_eva,
            "fixed_farr_macro_f1": macro_farr,
            "delta_macro_f1": macro_eva - macro_farr,
            "ci95": macro_ci,
            "p_value": macro_p,
            "success_rule": "positive point estimate and 95% CI excluding zero",
            "success": success,
        },
        "datasets": dataset_reports,
        "dataset_p_value_correction": "Holm over three datasets",
        "locks": {
            "test_lock": str(test_lock_path.relative_to(ROOT)),
            "test_lock_sha256": sha256(test_lock_path),
            "manifest_sha256": sha256(manifest_path),
            "selector_artifact_sha256": sha256(artifact_path),
        },
        "inputs": {
            "candidates": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                for path in candidate_paths
            ],
            "features": [
                {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
                for path in feature_paths
            ],
        },
        "bootstrap": {
            "repetitions": args.bootstrap_reps,
            "macro_method": "paired dataset-stratified percentile bootstrap",
            "dataset_method": "paired percentile bootstrap",
            "seed": args.seed,
        },
        "guardrail": (
            "A failed Test-C cannot be repaired by reverting to the old "
            "shortcut selector or retuning on Test-C."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Test-C: status={report['status']} EVA={macro_eva:.6f} "
        f"FARR={macro_farr:.6f} delta={macro_eva-macro_farr:+.6f} "
        f"CI={macro_ci}"
    )
    print(f"Report: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
