#!/usr/bin/env python3
"""GPU extraction of candidate-specific reranking and NLI proof features."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from tqdm import tqdm

from farr_star import compat  # noqa: F401
from farr_star.evidence_verifier import (
    TransformerEvidenceScorer,
    VerifierConfig,
    extract_question_features,
)
from farr_star.example_store import load_examples_by_ids
from run_hotpotqa import context_documents


ROOT = Path(__file__).resolve().parent
CANDIDATES = ("flare-embedded", "ircot", "farr")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_shard(dataset: str, question_id: str, count: int) -> int:
    value = hashlib.sha256(
        f"{dataset}:{question_id}".encode("utf-8")
    ).hexdigest()
    return int(value[:16], 16) % count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-files", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-v2-m3",
    )
    parser.add_argument(
        "--nli-model",
        default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    )
    parser.add_argument("--allow-download", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_candidate_groups(
    paths: Sequence[Path],
    *,
    shard_index: int,
    num_shards: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    order: list[tuple[str, str]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            method = str(row.get("method", ""))
            if method not in CANDIDATES:
                continue
            key = (str(row["dataset"]), str(row["question_id"]))
            if stable_shard(*key, num_shards) != shard_index:
                continue
            if key not in groups:
                order.append(key)
            if method in groups[key]:
                raise ValueError(f"Duplicate candidate row: {key}/{method}")
            groups[key][method] = row
    result = []
    for key in order:
        methods = groups[key]
        if set(methods) != set(CANDIDATES):
            raise ValueError(
                f"Incomplete exact candidate pool: {key} has {sorted(methods)}"
            )
        source_splits = {
            str(value.get("source_split", "")) for value in methods.values()
        }
        questions = {
            str(value.get("question", "")) for value in methods.values()
        }
        if len(source_splits) != 1 or not next(iter(source_splits)):
            raise ValueError(f"Missing/inconsistent source split: {key}")
        if len(questions) != 1:
            raise ValueError(f"Inconsistent question text: {key}")
        result.append(
            {
                "dataset": key[0],
                "question_id": key[1],
                "source_split": next(iter(source_splits)),
                "question": next(iter(questions)),
                "candidates": [methods[method] for method in CANDIDATES],
            }
        )
    return result


def load_contexts(
    groups: Sequence[dict[str, Any]],
    *,
    allow_download: bool,
) -> dict[tuple[str, str], list[Any]]:
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for group in groups:
        buckets[
            (str(group["dataset"]), str(group["source_split"]))
        ].append(str(group["question_id"]))
    result = {}
    for (dataset, split), ids in buckets.items():
        print(f"Loading {len(ids)} contexts from {dataset}/{split}...")
        examples = load_examples_by_ids(
            dataset,
            split,
            ids,
            allow_download=allow_download,
        )
        for example in examples:
            result[(dataset, str(example["id"]))] = context_documents(example)
    if len(result) != len(groups):
        raise RuntimeError(
            f"Context coverage mismatch: {len(result)} != {len(groups)}"
        )
    return result


def resume_rows(
    path: Path,
) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    if not path.exists():
        return [], set()
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    order: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["dataset"]), str(row["question_id"]))
        if key not in grouped:
            order.append(key)
        grouped[key][str(row["method"])] = row
    complete = {
        key for key, values in grouped.items() if set(values) == set(CANDIDATES)
    }
    rows = [
        grouped[key][method]
        for key in order
        if key in complete
        for method in CANDIDATES
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows, complete


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards).")
    candidate_paths = [ROOT / value for value in args.candidate_files]
    output = ROOT / args.output
    metadata_output = ROOT / (
        args.metadata_output or f"{args.output}.meta.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        output.unlink(missing_ok=True)
        metadata_output.unlink(missing_ok=True)
        existing_rows, complete = [], set()
    elif args.resume:
        existing_rows, complete = resume_rows(output)
    elif output.exists() or metadata_output.exists():
        raise FileExistsError("Feature output exists; use --resume/--overwrite.")
    else:
        existing_rows, complete = [], set()

    config = VerifierConfig(
        reranker_model=args.reranker_model,
        nli_model=args.nli_model,
    )
    expected_metadata = {
        "schema": "farr-eva-candidate-evidence-features-v1",
        "candidate_files": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
            }
            for path in candidate_paths
        ],
        "candidate_methods": list(CANDIDATES),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "verifier_config": config.to_dict(),
        "inference_exclusions": [
            "gold_answer",
            "target_f1",
            "dataset_name_as_feature",
            "expert_identity_as_feature",
            "execution_metadata",
            "internal_selection_label",
            "raw_trace_text",
        ],
    }
    if metadata_output.exists():
        existing_metadata = json.loads(
            metadata_output.read_text(encoding="utf-8")
        )
        for key in (
            "candidate_files",
            "candidate_methods",
            "shard_index",
            "num_shards",
            "verifier_config",
        ):
            if existing_metadata.get(key) != expected_metadata[key]:
                raise RuntimeError(f"Resume metadata mismatch: {key}")

    groups = load_candidate_groups(
        candidate_paths,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    pending = [
        group
        for group in groups
        if (group["dataset"], group["question_id"]) not in complete
    ]
    print(
        f"Evidence feature shard {args.shard_index}/{args.num_shards}: "
        f"questions={len(groups)}, complete={len(complete)}, "
        f"pending={len(pending)}"
    )
    contexts = load_contexts(pending, allow_download=args.allow_download)
    scorer = TransformerEvidenceScorer(
        config,
        device=args.device,
        local_files_only=not args.allow_download,
    )
    mode = "a" if output.exists() and existing_rows else "w"
    with output.open(mode, encoding="utf-8") as handle:
        progress = tqdm(pending, desc=f"evidence-gpu{args.shard_index}")
        for group in progress:
            key = (str(group["dataset"]), str(group["question_id"]))
            rows = extract_question_features(
                question=str(group["question"]),
                candidates=group["candidates"],
                documents=contexts[key],
                scorer=scorer,
                config=config,
            )
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            progress.set_postfix(dataset=key[0])

    final_rows, final_complete = resume_rows(output)
    if len(final_complete) != len(groups):
        raise RuntimeError(
            f"Feature shard incomplete: {len(final_complete)} != {len(groups)}"
        )
    metadata = {
        **expected_metadata,
        "question_count": len(final_complete),
        "row_count": len(final_rows),
        "output": str(output.relative_to(ROOT)),
        "output_sha256": sha256(output),
        "status": "COMPLETE",
    }
    metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Features: {output.relative_to(ROOT)}")
    print(f"Metadata: {metadata_output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
