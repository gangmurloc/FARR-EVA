from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from farr_star import (
    FARREPR,
    FARRODR,
    FARRSTAR,
    EvidencePairwiseRouter,
    OracleDistilledRouter,
    STARConfig,
)
from farr_star import compat  # noqa: F401
from farr_star.datasets import load_benchmark_examples
from farr.adapters import DEFAULT_LOCAL_MODEL, LocalHFLLM
from farr.baselines import flare, rag, rarr
from farr.config import FARRConfig
from farr.retrievers import TfidfRetriever
from run_hotpotqa import (
    context_documents,
    exact_match,
    result_row,
    token_f1,
    write_jsonl,
)


SUPPORTED_METHODS = {
    "rag",
    "flare",
    "rarr",
    "ircot",
    "farr",
    "farr-v2",
    "farr-star",
    "farr-odr",
    "farr-epr",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate dataset-independent FARR-STAR and baselines."
    )
    parser.add_argument("--model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument(
        "--dataset",
        default="hotpotqa",
        choices=["hotpotqa", "2wiki", "musique"],
    )
    parser.add_argument(
        "--methods",
        default="rag,flare,rarr,ircot,farr,farr-star",
    )
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--config", default="distractor")
    parser.add_argument(
        "--level",
        default="hard",
        choices=["", "easy", "medium", "hard"],
    )
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--router-path",
        default="artifacts/farr_odr_qwen.joblib",
        help="Oracle-distilled router artifact for method=farr-odr.",
    )
    parser.add_argument(
        "--epr-path",
        default="artifacts/farr_epr_qwen.joblib",
        help="Evidence pairwise ranker artifact for method=farr-epr.",
    )
    parser.add_argument(
        "--stratify-hops",
        action="store_true",
        help=(
            "For MuSiQue, balance samples across 2-, 3-, and 4-hop "
            "questions; sample-offset is applied inside each bucket."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/qwen_farr_star_test",
    )
    parser.add_argument(
        "--disable-contract-retrieval",
        action="store_true",
        help="Ablation: do not retrieve the contract's atomic queries.",
    )
    parser.add_argument(
        "--disable-repair",
        action="store_true",
        help="Ablation: select experts without evidence repair.",
    )
    parser.add_argument(
        "--disable-repair-verification",
        action="store_true",
        help="Ablation: accept locally validated repairs without an LLM verifier.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true")
    mode.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_resume_rows(
    result_path: Path,
    methods: list[str],
) -> tuple[list[dict[str, Any]], set[str]]:
    if not result_path.exists():
        return [], set()
    raw_rows = []
    for line in result_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw_rows.append(json.loads(line))
    expected = set(methods)
    unexpected = {
        str(row.get("method", "")) for row in raw_rows
    } - expected
    if unexpected:
        raise ValueError(
            f"Resume output contains unexpected methods: {sorted(unexpected)}"
        )

    by_question: dict[str, dict[str, dict[str, Any]]] = {}
    order = []
    for row in raw_rows:
        question_id = str(row.get("question_id", ""))
        method = str(row.get("method", ""))
        if question_id not in by_question:
            by_question[question_id] = {}
            order.append(question_id)
        by_question[question_id].setdefault(method, row)
    completed = {
        question_id
        for question_id, rows in by_question.items()
        if set(rows) == expected
    }
    kept = [
        by_question[question_id][method]
        for question_id in order
        if question_id in completed
        for method in methods
    ]
    result_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in kept
        ),
        encoding="utf-8",
    )
    return kept, completed


def summarize(
    rows: list[dict[str, Any]],
    methods: list[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        if not selected:
            continue
        summary[method] = {
            "count": len(selected),
            "exact_match": sum(
                float(row["exact_match"]) for row in selected
            ) / len(selected),
            "f1": sum(float(row["f1"]) for row in selected) / len(selected),
            "supporting_title_recall": sum(
                float(row["supporting_title_recall"]) for row in selected
            ) / len(selected),
            "avg_llm_calls": sum(
                float(row["llm_calls"]) for row in selected
            ) / len(selected),
            "avg_retrieval_calls": sum(
                float(row["retrieval_calls"]) for row in selected
            ) / len(selected),
        }
        if method in {"farr-star", "farr-odr", "farr-epr"}:
            summary[method]["routes"] = {
                route: sum(row.get("route") == route for row in selected)
                for route in sorted(
                    {
                        str(row.get("route", ""))
                        for row in selected
                        if row.get("route")
                    }
                )
            }
            summary[method]["reasoning_types"] = {
                kind: sum(
                    row.get("reasoning_type") == kind for row in selected
                )
                for kind in sorted(
                    {
                        str(row.get("reasoning_type", ""))
                        for row in selected
                        if row.get("reasoning_type")
                    }
                )
            }
    return summary


def main() -> None:
    args = parse_args()
    requested_methods = [
        method.strip().lower()
        for method in args.methods.split(",")
        if method.strip()
    ]
    unknown = set(requested_methods) - SUPPORTED_METHODS
    if unknown:
        raise ValueError(f"Unsupported methods: {sorted(unknown)}")
    # ``farr-v2`` remains a CLI compatibility alias. New result files and
    # paper-facing output consistently use the final method name ``farr``.
    methods = [
        "farr" if method == "farr-v2" else method
        for method in requested_methods
    ]
    if len(methods) != len(set(methods)):
        raise ValueError("Do not request both farr and its farr-v2 alias.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    if args.overwrite:
        result_path.unlink(missing_ok=True)
        summary_path.unlink(missing_ok=True)
    elif result_path.exists() and not args.resume:
        raise FileExistsError(
            "Output exists. Use --resume, --overwrite, or another output dir."
        )

    examples = load_benchmark_examples(args)
    if not examples:
        raise RuntimeError("No examples matched.")
    llm = LocalHFLLM(
        model_name=args.model,
        max_input_tokens=args.max_input_tokens,
        local_files_only=not args.allow_download,
    )
    farr_config = FARRConfig(
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
    star_config = STARConfig(
        enable_contract_retrieval=not args.disable_contract_retrieval,
        enable_repair=not args.disable_repair,
        verify_repair=not args.disable_repair_verification,
    )

    if args.resume:
        rows, completed_ids = load_resume_rows(result_path, methods)
        summary_path.unlink(missing_ok=True)
        print(
            f"Resume: {len(completed_ids)} complete questions and "
            f"{len(rows)} rows preserved."
        )
    else:
        rows = []
        completed_ids = set()

    combined_methods = {
        "farr-star",
        "farr-odr",
        "farr-epr",
    } & set(methods)
    if len(combined_methods) > 1:
        raise ValueError(
            "Evaluate farr-star, farr-odr, and farr-epr in separate runs "
            "to avoid duplicating the same experts."
        )
    internal = {
        "ircot",
        "farr",
        "farr-star",
        "farr-odr",
        "farr-epr",
    }
    need_star = bool(set(methods) & internal)
    router = (
        OracleDistilledRouter(args.router_path)
        if "farr-odr" in methods
        else None
    )
    epr_router = (
        EvidencePairwiseRouter(args.epr_path)
        if "farr-epr" in methods
        else None
    )
    for index, example in enumerate(
        tqdm(examples, desc=f"{args.dataset}-FARR-STAR"),
        1,
    ):
        if str(example.get("id", "")) in completed_ids:
            continue
        question = str(example["question"])
        retriever = TfidfRetriever(context_documents(example))
        results: dict[str, Any] = {}

        if need_star:
            if epr_router is not None:
                combined = FARREPR(
                    retriever,
                    llm,
                    epr_router,
                    farr_config,
                ).answer(question)
                results["farr-epr"] = combined
            elif router is not None:
                combined = FARRODR(
                    retriever,
                    llm,
                    router,
                    farr_config,
                ).answer(question)
                results["farr-odr"] = combined
            else:
                combined = FARRSTAR(
                    retriever,
                    llm,
                    farr_config,
                    star_config,
                ).answer(question)
                results["farr-star"] = combined
            results["farr"] = combined.farr_result
            results["ircot"] = combined.ircot_result
        if "rag" in methods:
            results["rag"] = rag(
                llm,
                retriever,
                question,
                farr_config.initial_top_k,
                farr_config.max_chars_per_doc,
            )
        if "flare" in methods:
            results["flare"] = flare(
                llm,
                retriever,
                question,
                farr_config,
            )
        if "rarr" in methods:
            results["rarr"] = rarr(
                llm,
                retriever,
                question,
                farr_config,
            )

        for method in methods:
            row = result_row(
                index,
                method,
                example,
                results[method],
            )
            aliases = [
                str(value)
                for value in example.get("answer_aliases") or []
                if str(value).strip()
            ]
            if aliases:
                answers = [str(example.get("answer", "")), *aliases]
                row["exact_match"] = max(
                    exact_match(row["prediction"], answer)
                    for answer in answers
                )
                row["f1"] = max(
                    token_f1(row["prediction"], answer)
                    for answer in answers
                )
            row["dataset"] = args.dataset
            rows.append(row)
            write_jsonl(result_path, row)

        star_row = next(
            (
                row
                for row in reversed(rows)
                if row["method"]
                in {"farr-star", "farr-odr", "farr-epr"}
            ),
            None,
        )
        if star_row is not None:
            print(
                f"\n[{index}/{len(examples)}] {star_row['method']}: "
                f"{star_row['prediction']!r} "
                f"(gold={star_row['gold_answer']!r}, "
                f"F1={star_row['f1']:.3f}, "
                f"type={star_row['reasoning_type']}, "
                f"route={star_row['route']}, "
                f"LLM={star_row['llm_calls']})"
            )

    summary = summarize(rows, methods)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Results: {result_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
