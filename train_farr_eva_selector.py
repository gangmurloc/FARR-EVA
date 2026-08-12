#!/usr/bin/env python3
"""Train and validation-lock the evidence-verified abstaining selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from farr_star.eva_selector import (
    ANCHOR,
    CANDIDATES,
    EvidenceVerifiedAbstainingSelector,
    fit_pairwise_selector,
)


ROOT = Path(__file__).resolve().parent
DATASETS = ("hotpotqa", "2wiki", "musique")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="data/farr_eva_features/development.jsonl",
    )
    parser.add_argument(
        "--feature-report",
        default="data/farr_eva_features/development_report.json",
    )
    parser.add_argument(
        "--train-candidates",
        default="data/farr_eva_exact_pool/train.jsonl",
    )
    parser.add_argument(
        "--validation-candidates",
        default="data/farr_eva_exact_pool/validation.jsonl",
    )
    parser.add_argument(
        "--c-grid",
        default="0.03,0.1,0.3,1,3,10",
    )
    parser.add_argument(
        "--threshold-grid",
        default="0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.975,0.99",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--minimum-macro-gain", type=float, default=0.005)
    parser.add_argument("--dataset-drop-tolerance", type=float, default=0.002)
    parser.add_argument(
        "--artifact",
        default="artifacts/farr_eva_v1.joblib",
    )
    parser.add_argument(
        "--report",
        default="artifacts/farr_eva_v1_validation_report.json",
    )
    parser.add_argument(
        "--lock",
        default="artifacts/farr_eva_v1.lock.json",
    )
    parser.add_argument(
        "--predictions",
        default="artifacts/farr_eva_v1_validation_predictions.jsonl",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def candidate_targets(
    paths: Sequence[Path],
) -> tuple[
    dict[tuple[str, str, str], float],
    dict[tuple[str, str, str], dict[str, Any]],
]:
    targets = {}
    rows = {}
    for path in paths:
        for row in load_jsonl(path):
            key = (
                str(row["dataset"]),
                str(row["question_id"]),
                str(row["method"]),
            )
            if key in targets:
                raise ValueError(f"Duplicate target row: {key}")
            if key[2] not in CANDIDATES:
                continue
            targets[key] = float(row["f1"])
            rows[key] = row
    return targets, rows


def feature_profiles(all_names: Sequence[str]) -> dict[str, list[str]]:
    names = list(all_names)
    answer_only = [
        name
        for name in names
        if name.startswith("answer_")
        or name == "contract_ok"
        or name.startswith("shape=")
    ]
    nli_core = [
        name
        for name in names
        if any(
            token in name
            for token in (
                "entail",
                "contradiction",
                "margin",
                "weighted_support",
                "supported_fraction",
                "evidence_title_diversity",
            )
        )
        or name == "contract_ok"
        or name.startswith("shape=")
    ]
    evidence_only = [
        name
        for name in names
        if name not in {"answer_tokens_log", "answer_trace_alignment"}
        and not name.startswith("shape=")
    ]
    profiles = {
        "full_proof": names,
        "evidence_only": evidence_only,
        "answer_evidence_only": answer_only,
        "nli_core": nli_core,
    }
    for name, values in profiles.items():
        if not values:
            raise RuntimeError(f"Empty feature profile: {name}")
    return profiles


def group_features(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["dataset"]), str(row["question_id"]))].append(row)
    for key, values in groups.items():
        if {str(row["method"]) for row in values} != set(CANDIDATES):
            raise ValueError(f"Incomplete feature group: {key}")
    return groups


def evaluate(
    selector: EvidenceVerifiedAbstainingSelector,
    groups: dict[tuple[str, str], list[dict[str, Any]]],
    targets: dict[tuple[str, str, str], float],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[tuple[str, str], float]]:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    predictions = []
    differences = {}
    for key, rows in groups.items():
        selected, probability, utilities, switched = selector.choose(rows)
        values = {
            method: float(targets[(*key, method)]) for method in CANDIDATES
        }
        selected_f1 = values[selected]
        anchor_f1 = values[ANCHOR]
        differences[key] = selected_f1 - anchor_f1
        winners = [
            method
            for method, value in values.items()
            if abs(value - max(values.values())) < 1e-12
        ]
        by_dataset[key[0]].append(
            {
                "selected_f1": selected_f1,
                "anchor_f1": anchor_f1,
                "oracle_f1": max(values.values()),
                "selected": selected,
                "switched": switched,
                "any_best": selected in winners,
                **values,
            }
        )
        predictions.append(
            {
                "dataset": key[0],
                "question_id": key[1],
                "selected": selected,
                "switch_probability": probability,
                "switched": switched,
                "utilities": utilities,
                "selected_f1": selected_f1,
                "anchor_f1": anchor_f1,
                "delta": selected_f1 - anchor_f1,
            }
        )
    datasets = {}
    for dataset in DATASETS:
        values = by_dataset[dataset]
        selected = float(np.mean([row["selected_f1"] for row in values]))
        anchor = float(np.mean([row["anchor_f1"] for row in values]))
        datasets[dataset] = {
            "count": len(values),
            "selector_f1": selected,
            "fixed_farr_f1": anchor,
            "delta": selected - anchor,
            "oracle_f1": float(
                np.mean([row["oracle_f1"] for row in values])
            ),
            "switch_rate": float(
                np.mean([row["switched"] for row in values])
            ),
            "any_best_accuracy": float(
                np.mean([row["any_best"] for row in values])
            ),
            "selected_counts": dict(
                Counter(str(row["selected"]) for row in values)
            ),
            "baselines": {
                method: float(np.mean([row[method] for row in values]))
                for method in CANDIDATES
            },
        }
    macro_selector = float(
        np.mean([value["selector_f1"] for value in datasets.values()])
    )
    macro_anchor = float(
        np.mean([value["fixed_farr_f1"] for value in datasets.values()])
    )
    summary = {
        "datasets": datasets,
        "macro_selector_f1": macro_selector,
        "macro_fixed_farr_f1": macro_anchor,
        "macro_delta": macro_selector - macro_anchor,
        "minimum_dataset_delta": min(
            value["delta"] for value in datasets.values()
        ),
        "switch_rate": float(
            np.mean([row["switched"] for values in by_dataset.values() for row in values])
        ),
    }
    return summary, predictions, differences


def stratified_bootstrap(
    differences: dict[tuple[str, str], float],
    *,
    reps: int,
    seed: int,
) -> tuple[list[float], float]:
    by_dataset = {
        dataset: np.asarray(
            [value for key, value in differences.items() if key[0] == dataset],
            dtype=float,
        )
        for dataset in DATASETS
    }
    rng = np.random.default_rng(seed)
    samples = np.empty(reps, dtype=float)
    for start in range(0, reps, 500):
        count = min(500, reps - start)
        means = []
        for values in by_dataset.values():
            indexes = rng.integers(0, len(values), size=(count, len(values)))
            means.append(values[indexes].mean(axis=1))
        samples[start:start + count] = np.vstack(means).mean(axis=0)
    low, high = np.quantile(samples, [0.025, 0.975])
    tail = 2.0 * min(
        float(np.mean(samples <= 0.0)),
        float(np.mean(samples >= 0.0)),
    )
    return [float(low), float(high)], max(tail, 1.0 / reps)


def main() -> None:
    args = parse_args()
    feature_path = ROOT / args.features
    feature_report_path = ROOT / args.feature_report
    train_path = ROOT / args.train_candidates
    validation_path = ROOT / args.validation_candidates
    artifact_path = ROOT / args.artifact
    report_path = ROOT / args.report
    lock_path = ROOT / args.lock
    predictions_path = ROOT / args.predictions
    outputs = (artifact_path, report_path, lock_path, predictions_path)
    if any(path.exists() for path in outputs) and not args.overwrite:
        raise FileExistsError("EVA artifact outputs exist; use --overwrite.")
    for path in outputs:
        if args.overwrite:
            path.unlink(missing_ok=True)

    feature_rows = load_jsonl(feature_path)
    feature_report = json.loads(
        feature_report_path.read_text(encoding="utf-8")
    )
    if sha256(feature_path) != feature_report["output_sha256"]:
        raise RuntimeError("Merged evidence feature hash mismatch.")
    targets, candidate_rows = candidate_targets([train_path, validation_path])
    feature_keys = {
        (str(row["dataset"]), str(row["question_id"]), str(row["method"]))
        for row in feature_rows
    }
    if feature_keys != set(targets):
        raise RuntimeError(
            f"Feature/target mismatch: features={len(feature_keys)}, "
            f"targets={len(targets)}, missing={len(set(targets)-feature_keys)}, "
            f"extra={len(feature_keys-set(targets))}"
        )
    train_rows = [
        row for row in feature_rows if str(row["experiment_split"]) == "train"
    ]
    validation_rows = [
        row
        for row in feature_rows
        if str(row["experiment_split"]) == "validation"
    ]
    train_groups = group_features(train_rows)
    validation_groups = group_features(validation_rows)
    all_names = sorted(
        {name for row in feature_rows for name in row["features"]}
    )
    profiles = feature_profiles(all_names)
    c_grid = parse_float_list(args.c_grid)
    threshold_grid = parse_float_list(args.threshold_grid)
    grid_results = []
    fitted = {}
    for profile_name, feature_names in profiles.items():
        for c_value in c_grid:
            scaler, model, fit_metadata = fit_pairwise_selector(
                train_rows,
                targets,
                feature_names=feature_names,
                c_value=c_value,
                random_state=args.seed,
            )
            if not fit_metadata["converged"]:
                continue
            fitted[(profile_name, c_value)] = (
                scaler,
                model,
                fit_metadata,
                feature_names,
            )
            for threshold in threshold_grid:
                selector = EvidenceVerifiedAbstainingSelector(
                    feature_names=list(feature_names),
                    scaler=scaler,
                    model=model,
                    switch_threshold=float(threshold),
                )
                metrics, _, _ = evaluate(
                    selector,
                    validation_groups,
                    targets,
                )
                safe = metrics["minimum_dataset_delta"] >= -float(
                    args.dataset_drop_tolerance
                )
                grid_results.append(
                    {
                        "profile": profile_name,
                        "c_value": c_value,
                        "switch_threshold": threshold,
                        "safe_dataset_constraint": safe,
                        **metrics,
                    }
                )
    if not grid_results:
        raise RuntimeError("No converged EVA validation configurations.")
    best = max(
        grid_results,
        key=lambda value: (
            int(value["safe_dataset_constraint"]),
            value["macro_delta"] if value["safe_dataset_constraint"] else value["minimum_dataset_delta"],
            value["minimum_dataset_delta"],
            -value["switch_rate"],
        ),
    )
    scaler, model, fit_metadata, feature_names = fitted[
        (str(best["profile"]), float(best["c_value"]))
    ]
    selector = EvidenceVerifiedAbstainingSelector(
        feature_names=list(feature_names),
        scaler=scaler,
        model=model,
        switch_threshold=float(best["switch_threshold"]),
    )
    validation_metrics, predictions, differences = evaluate(
        selector,
        validation_groups,
        targets,
    )
    ci95, p_value = stratified_bootstrap(
        differences,
        reps=args.bootstrap_reps,
        seed=args.seed + 17,
    )
    success = bool(
        validation_metrics["macro_delta"] >= args.minimum_macro_gain
        and validation_metrics["minimum_dataset_delta"]
        >= -args.dataset_drop_tolerance
        and ci95[0] > 0.0
    )
    metadata = {
        "schema": "farr-eva-artifact-v1",
        "method": "evidence-verified abstaining arbitration",
        "candidate_pool": list(CANDIDATES),
        "anchor": ANCHOR,
        "feature_profile": best["profile"],
        "feature_names": list(feature_names),
        "c_value": best["c_value"],
        "switch_threshold": best["switch_threshold"],
        "train_candidates_sha256": sha256(train_path),
        "validation_candidates_sha256": sha256(validation_path),
        "development_features_sha256": sha256(feature_path),
        "verifier_config": feature_report["verifier_config"],
        "fit_metadata": fit_metadata,
        "inference_exclusions": feature_report["inference_exclusions"],
        "validation_success": success,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    selector.save(str(artifact_path), metadata)
    artifact_hash = sha256(artifact_path)
    report = {
        "schema": "farr-eva-validation-selection-v1",
        "status": "PASS" if success else "FAIL_DO_NOT_RUN_TEST_C",
        "selection_rule": (
            "Prefer configurations satisfying per-dataset drop tolerance; "
            "then maximize validation macro delta, minimum dataset delta, "
            "and finally prefer fewer switches."
        ),
        "success_rule": {
            "minimum_macro_gain": args.minimum_macro_gain,
            "dataset_drop_tolerance": args.dataset_drop_tolerance,
            "macro_ci_lower_must_exceed_zero": True,
        },
        "selected_configuration": best,
        "validation": {
            **validation_metrics,
            "ci95": ci95,
            "p_value": p_value,
        },
        "artifact": str(artifact_path.relative_to(ROOT)),
        "artifact_sha256": artifact_hash,
        "feature_report": str(feature_report_path.relative_to(ROOT)),
        "feature_report_sha256": sha256(feature_report_path),
        "grid": grid_results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in predictions
        ),
        encoding="utf-8",
    )
    if success:
        lock = {
            "schema": "farr-eva-validation-lock-v1",
            "artifact": str(artifact_path.relative_to(ROOT)),
            "artifact_sha256": artifact_hash,
            "validation_report": str(report_path.relative_to(ROOT)),
            "validation_report_sha256": sha256(report_path),
            "feature_profile": best["profile"],
            "c_value": best["c_value"],
            "switch_threshold": best["switch_threshold"],
            "status": "LOCKED_BEFORE_TEST_C",
            "prohibition": (
                "Do not change features, models, threshold, candidates, "
                "metrics, or Test-C IDs after this lock is created."
            ),
        }
        lock_path.write_text(
            json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        f"EVA validation: status={report['status']} "
        f"macro_delta={validation_metrics['macro_delta']:+.6f} "
        f"min_dataset_delta={validation_metrics['minimum_dataset_delta']:+.6f} "
        f"CI={ci95}"
    )
    print(f"Report: {report_path.relative_to(ROOT)}")
    if not success:
        lock_path.unlink(missing_ok=True)
        raise SystemExit(2)
    print(f"Locked artifact: {lock_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
