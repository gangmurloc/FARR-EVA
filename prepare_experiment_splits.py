from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from datasets import load_dataset
from huggingface_hub import snapshot_download

from farr_star import compat  # noqa: F401


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the fixed FARR-EPR 9K/3K/6K split manifest."
    )
    parser.add_argument(
        "--output",
        default="data/farr_epr_9k3k6k_splits.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _shuffle(values: list[str], seed: int, name: str) -> list[str]:
    result = list(values)
    random.Random(f"{seed}:{name}").shuffle(result)
    return result


def _balanced_counts(
    groups: dict[str, list[str]],
    total: int,
) -> dict[str, int]:
    names = sorted(groups)
    base, remainder = divmod(total, len(names))
    counts = {
        name: base + int(index < remainder)
        for index, name in enumerate(names)
    }
    for name, count in counts.items():
        if count > len(groups[name]):
            raise ValueError(
                f"Not enough examples for {name}: {len(groups[name])} < {count}"
            )
    return counts


def _proportional_counts(
    groups: dict[str, list[str]],
    total: int,
) -> dict[str, int]:
    available = sum(len(values) for values in groups.values())
    if total > available:
        raise ValueError(f"Requested {total} from only {available} examples.")
    raw = {
        name: total * len(values) / available
        for name, values in groups.items()
    }
    counts = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(
        groups,
        key=lambda name: (raw[name] - counts[name], name),
        reverse=True,
    )
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def _partition(
    groups: dict[str, list[str]],
    train_counts: dict[str, int],
    validation_counts: dict[str, int],
    seed: int,
    name: str,
) -> tuple[
    list[str],
    list[str],
    dict[str, list[str]],
    dict[str, list[str]],
]:
    train_by_group = {}
    validation_by_group = {}
    for group in sorted(groups):
        values = _shuffle(groups[group], seed, f"{name}:{group}")
        train_end = train_counts[group]
        validation_end = train_end + validation_counts[group]
        if validation_end > len(values):
            raise ValueError(
                f"Not enough {name}/{group}: need {validation_end}, "
                f"have {len(values)}"
            )
        train_by_group[group] = values[:train_end]
        validation_by_group[group] = values[train_end:validation_end]
    train = _shuffle(
        [value for values in train_by_group.values() for value in values],
        seed,
        f"{name}:train",
    )
    validation = _shuffle(
        [
            value
            for values in validation_by_group.values()
            for value in values
        ],
        seed,
        f"{name}:validation",
    )
    return train, validation, train_by_group, validation_by_group


def _sample_test(
    groups: dict[str, list[str]],
    total: int,
    seed: int,
    name: str,
) -> tuple[list[str], dict[str, int]]:
    counts = _proportional_counts(groups, total)
    selected = []
    for group in sorted(groups):
        values = _shuffle(groups[group], seed, f"{name}:test:{group}")
        selected.extend(values[: counts[group]])
    return _shuffle(selected, seed, f"{name}:test"), counts


def _hotpot_groups(split: str) -> dict[str, list[str]]:
    dataset = load_dataset(
        "hotpot_qa",
        "distractor",
        split=split,
    )
    groups: dict[str, list[str]] = defaultdict(list)
    for row in dataset:
        if str(row.get("level", "")).lower() != "hard":
            continue
        groups[str(row.get("type", "other"))].append(str(row["id"]))
    return dict(groups)


def _wiki_groups(
    split: str,
    allow_download: bool,
) -> dict[str, list[str]]:
    snapshot = Path(
        snapshot_download(
            repo_id="xanhho/2WikiMultihopQA",
            repo_type="dataset",
            allow_patterns=[f"{split}.parquet"],
            local_files_only=not allow_download,
        )
    )
    path = snapshot / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run once with --allow-download."
        )
    dataset = load_dataset(
        "parquet",
        data_files={split: str(path)},
        split=split,
    )
    groups: dict[str, list[str]] = defaultdict(list)
    for row in dataset:
        groups[str(row.get("type", "other"))].append(str(row["_id"]))
    return dict(groups)


def _musique_groups(
    split: str,
    allow_download: bool,
) -> dict[str, list[str]]:
    snapshot = Path(
        snapshot_download(
            repo_id="dgslibisey/MuSiQue",
            repo_type="dataset",
            allow_patterns=[f"musique_ans_v1.0_{split}.jsonl"],
            local_files_only=not allow_download,
        )
    )
    path = snapshot / f"musique_ans_v1.0_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run once with --allow-download."
        )
    groups: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not bool(row.get("answerable", True)):
                continue
            match = re.match(r"([234])hop", str(row.get("id", "")))
            if match:
                groups[f"{match.group(1)}hop"].append(str(row["id"]))
    return dict(groups)


def _distribution(
    ids: list[str],
    groups: dict[str, list[str]],
) -> dict[str, int]:
    lookup = {
        value: group for group, values in groups.items() for value in values
    }
    return dict(sorted(Counter(lookup[value] for value in ids).items()))


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite.")

    hot_train_groups = _hotpot_groups("train")
    hot_test_groups = _hotpot_groups("validation")
    wiki_train_groups = _wiki_groups("train", args.allow_download)
    wiki_test_groups = _wiki_groups("dev", args.allow_download)
    musique_train_groups = _musique_groups("train", args.allow_download)
    musique_test_groups = _musique_groups("dev", args.allow_download)

    hot_train_counts = _balanced_counts(hot_train_groups, 3000)
    hot_validation_counts = _balanced_counts(hot_train_groups, 1000)
    hot_train, hot_validation, _, _ = _partition(
        hot_train_groups,
        hot_train_counts,
        hot_validation_counts,
        args.seed,
        "hotpotqa",
    )

    wiki_train_counts = _balanced_counts(wiki_train_groups, 3000)
    wiki_validation_counts = _balanced_counts(wiki_train_groups, 1000)
    wiki_train, wiki_validation, _, _ = _partition(
        wiki_train_groups,
        wiki_train_counts,
        wiki_validation_counts,
        args.seed,
        "2wiki",
    )

    musique_train_counts = {"2hop": 1080, "3hop": 1080, "4hop": 840}
    musique_validation_counts = {
        "2hop": 334,
        "3hop": 333,
        "4hop": 333,
    }
    (
        musique_train,
        musique_validation,
        musique_train_by_hop,
        musique_validation_by_hop,
    ) = _partition(
        musique_train_groups,
        musique_train_counts,
        musique_validation_counts,
        args.seed,
        "musique",
    )

    hot_test, hot_test_counts = _sample_test(
        hot_test_groups,
        2000,
        args.seed,
        "hotpotqa",
    )
    wiki_test, wiki_test_counts = _sample_test(
        wiki_test_groups,
        2000,
        args.seed,
        "2wiki",
    )
    musique_test, musique_test_counts = _sample_test(
        musique_test_groups,
        2000,
        args.seed,
        "musique",
    )

    musique_gpu0 = []
    musique_gpu1 = []
    for hop in ("2hop", "3hop", "4hop"):
        values = musique_train_by_hop[hop]
        midpoint = len(values) // 2
        musique_gpu0.extend(values[:midpoint])
        musique_gpu1.extend(values[midpoint:])
    musique_gpu0 = _shuffle(
        musique_gpu0, args.seed, "musique:gpu0"
    )
    musique_gpu1 = _shuffle(
        musique_gpu1, args.seed, "musique:gpu1"
    )
    musique_validation_gpu0 = []
    musique_validation_gpu1 = []
    validation_gpu0_counts = {"2hop": 167, "3hop": 167, "4hop": 166}
    for hop in ("2hop", "3hop", "4hop"):
        values = musique_validation_by_hop[hop]
        first = validation_gpu0_counts[hop]
        musique_validation_gpu0.extend(values[:first])
        musique_validation_gpu1.extend(values[first:])
    musique_validation_gpu0 = _shuffle(
        musique_validation_gpu0,
        args.seed,
        "musique:validation:gpu0",
    )
    musique_validation_gpu1 = _shuffle(
        musique_validation_gpu1,
        args.seed,
        "musique:validation:gpu1",
    )
    musique_test_by_hop = {
        hop: [
            question_id
            for question_id in musique_test
            if question_id in set(musique_test_groups[hop])
        ]
        for hop in ("2hop", "3hop", "4hop")
    }
    test_gpu0_counts = {"2hop": 518, "3hop": 315, "4hop": 167}
    musique_test_gpu0 = []
    musique_test_gpu1 = []
    for hop in ("2hop", "3hop", "4hop"):
        values = musique_test_by_hop[hop]
        first = test_gpu0_counts[hop]
        musique_test_gpu0.extend(values[:first])
        musique_test_gpu1.extend(values[first:])
    musique_test_gpu0 = _shuffle(
        musique_test_gpu0,
        args.seed,
        "musique:test:gpu0",
    )
    musique_test_gpu1 = _shuffle(
        musique_test_gpu1,
        args.seed,
        "musique:test:gpu1",
    )

    manifest: dict[str, Any] = {
        "name": "FARR-EPR-9K3K6K",
        "version": 1,
        "seed": args.seed,
        "protocol": {
            "train": 9000,
            "validation": 3000,
            "test": 6000,
        },
        "datasets": {
            "hotpotqa": {
                "train_source": "train",
                "test_source": "validation",
                "train": hot_train,
                "validation": hot_validation,
                "test": hot_test,
                "distributions": {
                    "train": _distribution(hot_train, hot_train_groups),
                    "validation": _distribution(
                        hot_validation, hot_train_groups
                    ),
                    "test": hot_test_counts,
                },
            },
            "2wiki": {
                "train_source": "train",
                "test_source": "dev",
                "train": wiki_train,
                "validation": wiki_validation,
                "test": wiki_test,
                "distributions": {
                    "train": _distribution(wiki_train, wiki_train_groups),
                    "validation": _distribution(
                        wiki_validation, wiki_train_groups
                    ),
                    "test": wiki_test_counts,
                },
            },
            "musique": {
                "train_source": "train",
                "test_source": "dev",
                "train": musique_train,
                "validation": musique_validation,
                "test": musique_test,
                "distributions": {
                    "train": musique_train_counts,
                    "validation": musique_validation_counts,
                    "test": musique_test_counts,
                },
            },
        },
        "training_shards": {
            "gpu0": [
                {
                    "dataset": "hotpotqa",
                    "source_split": "train",
                    "ids": hot_train,
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
                    "source_split": "train",
                    "ids": wiki_train,
                },
                {
                    "dataset": "musique",
                    "source_split": "train",
                    "ids": musique_gpu1,
                },
            ],
        },
        "validation_shards": {
            "gpu0": [
                {
                    "dataset": "hotpotqa",
                    "source_split": "train",
                    "ids": hot_validation,
                },
                {
                    "dataset": "musique",
                    "source_split": "train",
                    "ids": musique_validation_gpu0,
                },
            ],
            "gpu1": [
                {
                    "dataset": "2wiki",
                    "source_split": "train",
                    "ids": wiki_validation,
                },
                {
                    "dataset": "musique",
                    "source_split": "train",
                    "ids": musique_validation_gpu1,
                },
            ],
        },
        "test_shards": {
            "gpu0": [
                {
                    "dataset": "hotpotqa",
                    "source_split": "validation",
                    "ids": hot_test,
                },
                {
                    "dataset": "musique",
                    "source_split": "dev",
                    "ids": musique_test_gpu0,
                },
            ],
            "gpu1": [
                {
                    "dataset": "2wiki",
                    "source_split": "dev",
                    "ids": wiki_test,
                },
                {
                    "dataset": "musique",
                    "source_split": "dev",
                    "ids": musique_test_gpu1,
                },
            ],
        },
    }
    for shard, segments in manifest["training_shards"].items():
        count = sum(len(segment["ids"]) for segment in segments)
        if count != 4500:
            raise AssertionError(f"{shard} has {count}, expected 4500.")
    for shard, segments in manifest["validation_shards"].items():
        count = sum(len(segment["ids"]) for segment in segments)
        if count != 1500:
            raise AssertionError(f"{shard} has {count}, expected 1500.")
    for shard, segments in manifest["test_shards"].items():
        count = sum(len(segment["ids"]) for segment in segments)
        if count != 3000:
            raise AssertionError(f"{shard} has {count}, expected 3000.")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {output}")
    for dataset, values in manifest["datasets"].items():
        print(dataset, values["distributions"])
    for shard, segments in manifest["training_shards"].items():
        print(
            f"train/{shard}",
            [(segment["dataset"], len(segment["ids"])) for segment in segments],
        )
    for shard, segments in manifest["validation_shards"].items():
        print(
            f"validation/{shard}",
            [(segment["dataset"], len(segment["ids"])) for segment in segments],
        )
    for shard, segments in manifest["test_shards"].items():
        print(
            f"test/{shard}",
            [(segment["dataset"], len(segment["ids"])) for segment in segments],
        )


if __name__ == "__main__":
    main()
