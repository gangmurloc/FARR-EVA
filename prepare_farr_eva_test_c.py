#!/usr/bin/env python3
"""Reserve a fresh, question-disjoint 6K Test-C before EVA finalization."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prepare_experiment_splits import (
    _hotpot_groups,
    _musique_groups,
    _sample_test,
    _shuffle,
    _wiki_groups,
)


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-manifest",
        default="data/farr_epr_9k3k6k_splits.json",
    )
    parser.add_argument(
        "--test-b-manifest",
        default="data/farr_epr_confirmation_test_b.json",
    )
    parser.add_argument(
        "--output",
        default="data/farr_eva_test_c.json",
    )
    parser.add_argument(
        "--reservation-lock",
        default="data/farr_eva_test_c.reservation.lock.json",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def remaining(
    groups: dict[str, list[str]],
    excluded: set[str],
) -> dict[str, list[str]]:
    return {
        name: [value for value in values if value not in excluded]
        for name, values in groups.items()
        if any(value not in excluded for value in values)
    }


def distribution(ids: list[str], groups: dict[str, list[str]]) -> dict[str, int]:
    lookup = {
        value: name for name, values in groups.items() for value in values
    }
    return dict(sorted(Counter(lookup[value] for value in ids).items()))


def split_balanced(
    ids: list[str],
    groups: dict[str, list[str]],
    seed: int,
) -> tuple[list[str], list[str]]:
    lookup = {
        value: name for name, values in groups.items() for value in values
    }
    by_group = {
        name: _shuffle(
            [value for value in ids if lookup[value] == name],
            seed,
            f"test-c:musique:{name}",
        )
        for name in sorted(groups)
    }
    gpu0_counts = {name: len(values) // 2 for name, values in by_group.items()}
    remaining_count = 1000 - sum(gpu0_counts.values())
    for name in [name for name, values in by_group.items() if len(values) % 2][
        :remaining_count
    ]:
        gpu0_counts[name] += 1
    gpu0, gpu1 = [], []
    for name, values in by_group.items():
        first = gpu0_counts[name]
        gpu0.extend(values[:first])
        gpu1.extend(values[first:])
    gpu0 = _shuffle(gpu0, seed, "test-c:musique:gpu0")
    gpu1 = _shuffle(gpu1, seed, "test-c:musique:gpu1")
    if len(gpu0) != 1000 or len(gpu1) != 1000:
        raise RuntimeError(f"MuSiQue Test-C imbalance: {len(gpu0)}/{len(gpu1)}")
    return gpu0, gpu1


def all_ids(manifest: dict[str, Any], dataset: str) -> set[str]:
    values = manifest["datasets"][dataset]
    return {
        str(value)
        for split in ("train", "validation", "test")
        for value in values.get(split, [])
    }


def main() -> None:
    args = parse_args()
    original_path = ROOT / args.original_manifest
    test_b_path = ROOT / args.test_b_manifest
    output = ROOT / args.output
    reservation_lock = ROOT / args.reservation_lock
    if (output.exists() or reservation_lock.exists()) and not args.overwrite:
        raise FileExistsError(
            "Test-C reservation exists. Never overwrite after development "
            "feature extraction begins."
        )
    original = json.loads(original_path.read_text(encoding="utf-8"))
    test_b = json.loads(test_b_path.read_text(encoding="utf-8"))
    hot_groups = _hotpot_groups("validation")
    wiki_groups = _wiki_groups("dev", args.allow_download)
    musique_groups = _musique_groups("train", args.allow_download)
    excluded = {
        dataset: all_ids(original, dataset) | all_ids(test_b, dataset)
        for dataset in ("hotpotqa", "2wiki", "musique")
    }
    hot_remaining = remaining(hot_groups, excluded["hotpotqa"])
    wiki_remaining = remaining(wiki_groups, excluded["2wiki"])
    musique_remaining = remaining(musique_groups, excluded["musique"])
    hot_ids, _ = _sample_test(
        hot_remaining, 2000, args.seed, "test-c:hotpotqa"
    )
    wiki_ids, _ = _sample_test(
        wiki_remaining, 2000, args.seed, "test-c:2wiki"
    )
    musique_ids, _ = _sample_test(
        musique_remaining, 2000, args.seed, "test-c:musique"
    )
    musique_gpu0, musique_gpu1 = split_balanced(
        musique_ids, musique_remaining, args.seed
    )
    chosen = {
        "hotpotqa": set(hot_ids),
        "2wiki": set(wiki_ids),
        "musique": set(musique_ids),
    }
    overlap = {
        dataset: len(chosen[dataset] & excluded[dataset])
        for dataset in chosen
    }
    if any(overlap.values()):
        raise RuntimeError(f"Test-C overlap detected: {overlap}")
    manifest: dict[str, Any] = {
        "name": "FARR-EVA-TEST-C",
        "version": 1,
        "seed": args.seed,
        "created_at_utc": datetime.now(timezone.utc).replace(
            microsecond=0
        ).isoformat(),
        "protocol": {
            "test": 6000,
            "role": (
                "reserved untouched final confirmation after the failed "
                "portable Test-B; Test-C must not be read during development"
            ),
        },
        "excluded_manifests": [
            {
                "path": str(original_path.relative_to(ROOT)),
                "sha256": sha256(original_path),
            },
            {
                "path": str(test_b_path.relative_to(ROOT)),
                "sha256": sha256(test_b_path),
            },
        ],
        "datasets": {
            "hotpotqa": {
                "test_source": "validation",
                "train": [],
                "validation": [],
                "test": hot_ids,
                "distributions": {"test": distribution(hot_ids, hot_groups)},
            },
            "2wiki": {
                "test_source": "dev",
                "train": [],
                "validation": [],
                "test": wiki_ids,
                "distributions": {"test": distribution(wiki_ids, wiki_groups)},
            },
            "musique": {
                "test_source": "train",
                "train": [],
                "validation": [],
                "test": musique_ids,
                "distributions": {
                    "test": distribution(musique_ids, musique_groups)
                },
                "note": (
                    "Question-disjoint unused MuSiQue train IDs are used "
                    "because the dev split cannot supply another unused 2K."
                ),
            },
        },
        "test_shards": {
            "gpu0": [
                {
                    "dataset": "hotpotqa",
                    "source_split": "validation",
                    "ids": hot_ids,
                },
                {
                    "dataset": "musique",
                    "source_split": "train",
                    "ids": musique_gpu0,
                },
            ],
            "gpu1": [
                {
                    "dataset": "2wiki",
                    "source_split": "dev",
                    "ids": wiki_ids,
                },
                {
                    "dataset": "musique",
                    "source_split": "train",
                    "ids": musique_gpu1,
                },
            ],
        },
        "overlap_audit": {
            "prior_question_id_overlap": overlap,
            "gpu0_questions": 3000,
            "gpu1_questions": 3000,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lock = {
        "schema": "farr-eva-test-c-reservation-v1",
        "manifest": str(output.relative_to(ROOT)),
        "manifest_sha256": sha256(output),
        "question_count": 6000,
        "per_dataset": {dataset: 2000 for dataset in chosen},
        "status": "RESERVED_BEFORE_EVA_VALIDATION",
        "prohibition": (
            "Do not inspect, replace, or use Test-C questions during model "
            "development. Bind the validation-frozen artifact in a separate "
            "final lock before inference."
        ),
    }
    reservation_lock.write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Reserved Test-C: {output.relative_to(ROOT)}")
    print(f"Reservation lock: {reservation_lock.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
