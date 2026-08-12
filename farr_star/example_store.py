from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from datasets import load_dataset
from huggingface_hub import snapshot_download

from . import compat  # noqa: F401
from .datasets import _json_value, _standard_context


PREPARED_DATASETS = {
    "triviaqa",
    "squad",
    "nq_open",
    "webquestions",
}


def _prepared(
    dataset: str,
    ids: set[str],
    split: str,
) -> dict[str, dict[str, Any]]:
    path = Path("data/easyqa_prepared") / dataset / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Prepared easy-QA split not found: {path}. "
            "Run prepare_easyqa_splits.py first."
        )
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row.get("id", ""))
            if question_id in ids:
                result[question_id] = row
    return result


def _hotpot(ids: set[str], split: str) -> dict[str, dict[str, Any]]:
    dataset = load_dataset(
        "hotpot_qa",
        "distractor",
        split=split,
    )
    return {
        str(row["id"]): dict(row)
        for row in dataset
        if str(row["id"]) in ids
    }


def _wiki(
    ids: set[str],
    split: str,
    allow_download: bool,
) -> dict[str, dict[str, Any]]:
    snapshot = Path(
        snapshot_download(
            repo_id="xanhho/2WikiMultihopQA",
            repo_type="dataset",
            allow_patterns=[f"{split}.parquet"],
            local_files_only=not allow_download,
        )
    )
    path = snapshot / f"{split}.parquet"
    dataset = load_dataset(
        "parquet",
        data_files={split: str(path)},
        split=split,
    )
    result = {}
    for raw in dataset:
        question_id = str(raw.get("_id", ""))
        if question_id not in ids:
            continue
        facts = _json_value(raw.get("supporting_facts") or [])
        result[question_id] = {
            "id": question_id,
            "question": str(raw.get("question", "")),
            "answer": str(raw.get("answer", "")),
            "answer_aliases": [],
            "level": "multi-hop",
            "type": str(raw.get("type", "")),
            "context": _standard_context(
                _json_value(raw.get("context") or [])
            ),
            "supporting_facts": {
                "title": [
                    str(item[0])
                    for item in facts
                    if isinstance(item, (list, tuple)) and item
                ],
                "sent_id": [
                    int(item[1])
                    for item in facts
                    if isinstance(item, (list, tuple)) and len(item) > 1
                ],
            },
        }
    return result


def _musique(
    ids: set[str],
    split: str,
    allow_download: bool,
) -> dict[str, dict[str, Any]]:
    snapshot = Path(
        snapshot_download(
            repo_id="dgslibisey/MuSiQue",
            repo_type="dataset",
            allow_patterns=[f"musique_ans_v1.0_{split}.jsonl"],
            local_files_only=not allow_download,
        )
    )
    path = snapshot / f"musique_ans_v1.0_{split}.jsonl"
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            question_id = str(raw.get("id", ""))
            if question_id not in ids:
                continue
            paragraphs = list(raw.get("paragraphs") or [])
            supporting = [
                paragraph
                for paragraph in paragraphs
                if bool(paragraph.get("is_supporting", False))
            ]
            result[question_id] = {
                "id": question_id,
                "question": str(raw.get("question", "")),
                "answer": str(raw.get("answer", "")),
                "answer_aliases": [
                    str(value)
                    for value in raw.get("answer_aliases") or []
                ],
                "level": question_id.split("__", 1)[0],
                "type": "compositional",
                "context": {
                    "title": [
                        str(paragraph.get("title", ""))
                        for paragraph in paragraphs
                    ],
                    "sentences": [
                        [str(paragraph.get("paragraph_text", ""))]
                        for paragraph in paragraphs
                    ],
                },
                "supporting_facts": {
                    "title": [
                        str(paragraph.get("title", ""))
                        for paragraph in supporting
                    ],
                    "sent_id": [0] * len(supporting),
                },
            }
    return result


def load_examples_by_ids(
    dataset: str,
    split: str,
    ids: Sequence[str],
    allow_download: bool = False,
) -> list[dict[str, Any]]:
    order = [str(value) for value in ids]
    required = set(order)
    if len(required) != len(order):
        raise ValueError(f"Duplicate IDs requested for {dataset}/{split}.")
    if dataset == "hotpotqa":
        found = _hotpot(required, split)
    elif dataset == "2wiki":
        found = _wiki(required, split, allow_download)
    elif dataset == "musique":
        found = _musique(required, split, allow_download)
    elif dataset in PREPARED_DATASETS:
        found = _prepared(dataset, required, split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    missing = required - set(found)
    if missing:
        raise KeyError(
            f"Missing {len(missing)} {dataset}/{split} examples; "
            f"sample={sorted(missing)[:5]}"
        )
    return [found[question_id] for question_id in order]
