from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from huggingface_hub import snapshot_download

from . import compat  # noqa: F401
from farr.types import Document
from run_hotpotqa import context_documents


ContextKey = tuple[str, str]


def _documents(
    question_id: str,
    titles: list[Any],
    sentence_groups: list[Any],
) -> list[Document]:
    example = {
        "id": question_id,
        "context": {
            "title": [str(value) for value in titles],
            "sentences": sentence_groups,
        },
    }
    return context_documents(example)


def _load_hotpot(required: set[str]) -> dict[ContextKey, list[Document]]:
    dataset = load_dataset(
        "hotpot_qa",
        "distractor",
        split="validation",
    )
    result = {}
    for raw in dataset:
        question_id = str(raw.get("id", ""))
        if question_id not in required:
            continue
        context = raw.get("context") or {}
        result[("hotpotqa", question_id)] = _documents(
            question_id,
            list(context.get("title") or []),
            list(context.get("sentences") or []),
        )
    return result


def _load_2wiki(required: set[str]) -> dict[ContextKey, list[Document]]:
    snapshot = Path(
        snapshot_download(
            repo_id="xanhho/2WikiMultihopQA",
            repo_type="dataset",
            local_files_only=True,
        )
    )
    dataset = load_dataset(
        "parquet",
        data_files={"validation": str(snapshot / "dev.parquet")},
        split="validation",
    )
    result = {}
    for raw in dataset:
        question_id = str(raw.get("_id", ""))
        if question_id not in required:
            continue
        context = raw.get("context") or []
        if isinstance(context, str):
            context = json.loads(context)
        titles = []
        groups = []
        for item in context:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            titles.append(str(item[0]))
            sentences = item[1]
            groups.append(
                [str(value) for value in sentences]
                if isinstance(sentences, list)
                else [str(sentences)]
            )
        result[("2wiki", question_id)] = _documents(
            question_id,
            titles,
            groups,
        )
    return result


def _load_musique(required: set[str]) -> dict[ContextKey, list[Document]]:
    snapshot = Path(
        snapshot_download(
            repo_id="dgslibisey/MuSiQue",
            repo_type="dataset",
            local_files_only=True,
        )
    )
    path = snapshot / "musique_ans_v1.0_dev.jsonl"
    result = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            question_id = str(raw.get("id", ""))
            if question_id not in required:
                continue
            paragraphs = list(raw.get("paragraphs") or [])
            result[("musique", question_id)] = _documents(
                question_id,
                [
                    str(paragraph.get("title", ""))
                    for paragraph in paragraphs
                ],
                [
                    [str(paragraph.get("paragraph_text", ""))]
                    for paragraph in paragraphs
                ],
            )
    return result


def load_context_index(
    keys: Iterable[ContextKey],
) -> dict[ContextKey, list[Document]]:
    required: dict[str, set[str]] = {
        "hotpotqa": set(),
        "2wiki": set(),
        "musique": set(),
    }
    for dataset, question_id in keys:
        if dataset not in required:
            raise ValueError(f"Unsupported context dataset: {dataset}")
        required[dataset].add(question_id)

    result: dict[ContextKey, list[Document]] = {}
    if required["hotpotqa"]:
        result.update(_load_hotpot(required["hotpotqa"]))
    if required["2wiki"]:
        result.update(_load_2wiki(required["2wiki"]))
    if required["musique"]:
        result.update(_load_musique(required["musique"]))

    requested = {
        (dataset, question_id)
        for dataset, values in required.items()
        for question_id in values
    }
    missing = requested - set(result)
    if missing:
        examples = sorted(missing)[:5]
        raise KeyError(
            f"Missing {len(missing)} contexts; examples={examples}"
        )
    return result

