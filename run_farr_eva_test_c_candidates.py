#!/usr/bin/env python3
"""Generate the locked exact candidate pool and baselines for one Test-C shard."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from farr_star import compat  # noqa: F401
from farr_star.example_store import load_examples_by_ids
from farr.adapters import DEFAULT_LOCAL_MODEL, LocalHFLLM
from farr.baselines import flare, ircot, rag, rarr
from farr.config import FARRConfig
from farr.pipeline_v2 import FARRV2, clean_short_answer
from farr.retrievers import TfidfRetriever
from farr.types import FARRResult, FARRStats
from run_benchmark import summarize
from run_hotpotqa import (
    context_documents,
    exact_match,
    result_row,
    token_f1,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parent
METHODS = (
    "rag",
    "flare",
    "rarr",
    "ircot",
    "farr",
    "flare-embedded",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-lock",
        default="data/farr_eva_test_c.lock.json",
    )
    parser.add_argument("--shard", required=True, choices=("gpu0", "gpu1"))
    parser.add_argument("--model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="outputs/farr_eva_test_c",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser.parse_args()


def verify_test_lock(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("status") != "LOCKED_BEFORE_TEST_C_INFERENCE":
        raise RuntimeError("Test-C is not in locked pre-inference state.")
    manifest_path = ROOT / str(lock["manifest"])
    artifact_path = ROOT / str(lock["selector_artifact"])
    if sha256(manifest_path) != lock["manifest_sha256"]:
        raise RuntimeError("Test-C manifest hash mismatch.")
    if sha256(artifact_path) != lock["selector_artifact_sha256"]:
        raise RuntimeError("EVA selector artifact hash mismatch.")
    return lock, json.loads(manifest_path.read_text(encoding="utf-8"))


def load_resume(path: Path) -> tuple[list[dict[str, Any]], set[tuple[str, str]]]:
    if not path.exists():
        return [], set()
    groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    order: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["dataset"]), str(row["question_id"]))
        if key not in groups:
            order.append(key)
        groups[key][str(row["method"])] = row
    complete = {
        key for key, values in groups.items() if set(values) == set(METHODS)
    }
    rows = [
        groups[key][method]
        for key in order
        if key in complete
        for method in METHODS
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows, complete


def score_aliases(row: dict[str, Any], example: dict[str, Any]) -> None:
    answers = [
        str(example.get("answer", "")),
        *[
            str(value)
            for value in example.get("answer_aliases") or []
            if str(value).strip()
        ],
    ]
    row["exact_match"] = max(
        exact_match(str(row["prediction"]), answer) for answer in answers
    )
    row["f1"] = max(
        token_f1(str(row["prediction"]), answer) for answer in answers
    )


def embedded_flare_result(question: str, farr_result: Any) -> FARRResult:
    traces = list(farr_result.stats.hop_traces)
    completed = len(traces)
    answer = clean_short_answer(
        question,
        farr_result.stats.candidate_answers.get("flare", farr_result.answer),
        fallback=farr_result.answer,
    )
    stats = FARRStats(
        llm_calls=2 * completed + 1,
        retrieval_calls=completed + 1,
        planned_hops=completed,
        completed_hops=completed,
        revision_count=0,
        final_verification_label="UNCERTAIN",
        hop_traces=traces,
    )
    return FARRResult(
        answer=answer,
        evidence=list(farr_result.evidence),
        stats=stats,
    )


def main() -> None:
    args = parse_args()
    _, manifest = verify_test_lock(ROOT / args.test_lock)
    segments = list(manifest["test_shards"][args.shard])
    expected = sum(len(segment["ids"]) for segment in segments)
    output_dir = ROOT / args.output_dir / args.shard
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "candidates.jsonl"
    summary_path = output_dir / "candidate_summary.json"
    if args.overwrite:
        result_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
        rows, complete = [], set()
    elif args.resume:
        rows, complete = load_resume(result_path)
        summary_path.unlink(missing_ok=True)
        print(f"Resume: {len(complete)}/{expected} questions complete.")
    elif result_path.exists():
        raise FileExistsError("Candidate output exists; use --resume/--overwrite.")
    else:
        rows, complete = [], set()

    llm = LocalHFLLM(
        model_name=args.model,
        max_input_tokens=args.max_input_tokens,
        local_files_only=not args.allow_download,
    )
    config = FARRConfig(
        initial_top_k=6,
        max_hops=4,
        max_queries_per_hop=3,
        per_query_top_k=4,
        hop_evidence_top_k=5,
        max_evidence_docs=16,
        verification_top_k=5,
        max_revision_rounds=2,
        max_chars_per_doc=1000,
        revise_on_labels=("UNSUPPORTED", "UNCERTAIN"),
    )
    global_index = 0
    for segment in segments:
        dataset = str(segment["dataset"])
        source_split = str(segment["source_split"])
        examples = load_examples_by_ids(
            dataset,
            source_split,
            [str(value) for value in segment["ids"]],
            allow_download=args.allow_download,
        )
        progress = tqdm(
            examples,
            desc=f"{args.shard}:{dataset}:Test-C-candidates",
            total=len(examples),
        )
        for example in progress:
            global_index += 1
            key = (dataset, str(example["id"]))
            if key in complete:
                continue
            question = " ".join(str(example["question"]).split())
            retriever = TfidfRetriever(context_documents(example))
            farr_result = FARRV2(retriever, llm, config).answer(question)
            ircot_result = ircot(llm, retriever, question, config)
            embedded = embedded_flare_result(question, farr_result)
            results = {
                "rag": rag(
                    llm,
                    retriever,
                    question,
                    config.initial_top_k,
                    config.max_chars_per_doc,
                ),
                "flare": flare(llm, retriever, question, config),
                "rarr": rarr(llm, retriever, question, config),
                "ircot": ircot_result,
                "farr": farr_result,
                "flare-embedded": embedded,
            }
            question_rows = []
            for method in METHODS:
                row = result_row(
                    global_index,
                    method,
                    example,
                    results[method],
                )
                score_aliases(row, example)
                row.update(
                    {
                        "dataset": dataset,
                        "source_split": source_split,
                        "experiment_split": "test-c",
                        "shard": args.shard,
                        "candidate_source": (
                            "embedded_farr_flare"
                            if method == "flare-embedded"
                            else method
                        ),
                    }
                )
                question_rows.append(row)
                write_jsonl(result_path, row)
            rows.extend(question_rows)
            complete.add(key)
            progress.set_postfix(
                done=f"{len(complete)}/{expected}",
                farr=farr_result.answer,
            )
    summary = summarize(rows, list(METHODS))
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Candidates: {result_path.relative_to(ROOT)}")
    print(f"Summary: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
