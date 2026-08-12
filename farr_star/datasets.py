from __future__ import annotations

import json
import re
from argparse import Namespace
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import snapshot_download

from . import compat  # noqa: F401
from run_hotpotqa import load_examples as load_hotpot_examples


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _standard_context(
    titles_and_sentences: list[Any],
) -> dict[str, list[Any]]:
    titles = []
    sentence_groups = []
    for item in titles_and_sentences:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        titles.append(str(item[0]))
        sentences = item[1]
        sentence_groups.append(
            [str(value) for value in sentences]
            if isinstance(sentences, list)
            else [str(sentences)]
        )
    return {"title": titles, "sentences": sentence_groups}


def _slice(rows: list[dict[str, Any]], args: Namespace) -> list[dict[str, Any]]:
    start = int(args.sample_offset)
    return rows[start : start + int(args.num_samples)]


def stratified_hop_slice(
    rows: list[dict[str, Any]],
    num_samples: int,
    sample_offset: int,
) -> list[dict[str, Any]]:
    """Select an approximately equal number of 2-, 3-, and 4-hop examples.

    ``sample_offset`` is applied within each hop bucket so successive pilot
    runs can remain disjoint without depending on the dataset's file order.
    """

    groups: dict[int, list[dict[str, Any]]] = {2: [], 3: [], 4: []}
    for row in rows:
        match = re.match(r"([234])hop", str(row.get("id", "")))
        if match:
            groups[int(match.group(1))].append(row)
    base, remainder = divmod(int(num_samples), 3)
    counts = {
        hop: base + int(index < remainder)
        for index, hop in enumerate((2, 3, 4))
    }
    selected = []
    for hop in (2, 3, 4):
        start = int(sample_offset)
        end = start + counts[hop]
        bucket = groups[hop]
        if end > len(bucket):
            raise ValueError(
                f"Not enough {hop}-hop MuSiQue examples for "
                f"offset={start}, count={counts[hop]}."
            )
        selected.extend(bucket[start:end])
    return selected


def _load_2wiki(args: Namespace) -> list[dict[str, Any]]:
    snapshot = Path(
        snapshot_download(
            repo_id="xanhho/2WikiMultihopQA",
            repo_type="dataset",
            local_files_only=not args.allow_download,
        )
    )
    split_name = "dev" if args.split in {"validation", "dev"} else args.split
    path = snapshot / f"{split_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"2Wiki split is not cached: {path}. "
            "Use --allow-download once or select validation."
        )
    dataset = load_dataset(
        "parquet",
        data_files={args.split: str(path)},
        split=args.split,
    )
    rows = []
    for raw in dataset:
        facts = _json_value(raw.get("supporting_facts") or [])
        rows.append(
            {
                "id": str(raw.get("_id", "")),
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
                        if isinstance(item, (list, tuple))
                        and len(item) > 1
                    ],
                },
            }
        )
    return _slice(rows, args)


def _load_musique(args: Namespace) -> list[dict[str, Any]]:
    snapshot = Path(
        snapshot_download(
            repo_id="dgslibisey/MuSiQue",
            repo_type="dataset",
            local_files_only=not args.allow_download,
        )
    )
    split_name = "dev" if args.split in {"validation", "dev"} else args.split
    path = snapshot / f"musique_ans_v1.0_{split_name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"MuSiQue split is not cached: {path}. "
            "Use --allow-download once or select validation."
        )
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if not bool(raw.get("answerable", True)):
                continue
            paragraphs = list(raw.get("paragraphs") or [])
            supporting = [
                paragraph
                for paragraph in paragraphs
                if bool(paragraph.get("is_supporting", False))
            ]
            rows.append(
                {
                    "id": str(raw.get("id", "")),
                    "question": str(raw.get("question", "")),
                    "answer": str(raw.get("answer", "")),
                    "answer_aliases": [
                        str(value)
                        for value in raw.get("answer_aliases") or []
                    ],
                    "level": str(raw.get("id", "")).split("__", 1)[0],
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
            )
    if bool(getattr(args, "stratify_hops", False)):
        return stratified_hop_slice(
            rows,
            args.num_samples,
            args.sample_offset,
        )
    return _slice(rows, args)


def load_benchmark_examples(args: Namespace) -> list[dict[str, Any]]:
    if args.dataset == "hotpotqa":
        return load_hotpot_examples(args)
    if args.dataset == "2wiki":
        return _load_2wiki(args)
    if args.dataset == "musique":
        return _load_musique(args)
    raise ValueError(f"Unsupported dataset: {args.dataset}")
