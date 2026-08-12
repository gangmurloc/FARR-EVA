from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import load_dataset
from tqdm import tqdm

from farr import FARR, FARRConfig, FARRV1, FARRV2
from farr.adapters import DEFAULT_LOCAL_MODEL, LocalHFLLM, OpenAIChatLLM
from farr.baselines import closed_book, flare, full_context, ircot, rag, rarr
from farr.documents import doc_title
from farr.retrievers import TfidfRetriever
from farr.types import FARRResult


def normalize_answer(text: str) -> str:
    text = str(text).lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(gold).split()
    if not predicted or not expected:
        return float(predicted == expected)
    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def context_documents(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    context = example.get("context") or {}
    titles = context.get("title") or []
    sentence_groups = context.get("sentences") or []
    documents = []
    for index, sentences in enumerate(sentence_groups):
        title = titles[index] if index < len(titles) else f"context_{index}"
        text = " ".join(map(str, sentences)) if isinstance(sentences, list) else str(sentences)
        if text.strip():
            documents.append(
                {
                    "page_content": text.strip(),
                    "metadata": {
                        "title": str(title),
                        "question_id": str(example.get("id", "")),
                    },
                }
            )
    return documents


def supporting_titles(example: Dict[str, Any]) -> set[str]:
    facts = example.get("supporting_facts") or {}
    return {str(title) for title in facts.get("title", [])}


def evidence_title_recall(result: FARRResult, example: Dict[str, Any]) -> float:
    gold = supporting_titles(example)
    if not gold:
        return 0.0
    retrieved = {doc_title(doc) for doc in result.evidence}
    return len(gold & retrieved) / len(gold)


def load_examples(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.sample_offset:
        split = args.split
    else:
        load_count = max(args.num_samples * (10 if args.level else 1), args.num_samples)
        split = f"{args.split}[:{load_count}]"
    print(f"Loading HotpotQA {args.config} / {split}")
    dataset = load_dataset("hotpot_qa", args.config, split=split)
    examples = list(dataset)
    if args.level:
        examples = [
            example for example in examples
            if str(example.get("level", "")).lower() == args.level.lower()
        ]
    start = args.sample_offset
    end = start + args.num_samples
    return examples[start:end]


def write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, row: Dict[str, Any]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def build_llm(args: argparse.Namespace) -> Any:
    if args.llm == "openai":
        return OpenAIChatLLM(args.model)
    return LocalHFLLM(
        model_name=args.model,
        max_input_tokens=args.max_input_tokens,
        local_files_only=not args.allow_download,
    )


def result_row(
    index: int,
    method: str,
    example: Dict[str, Any],
    result: FARRResult,
) -> Dict[str, Any]:
    gold = str(example.get("answer", ""))
    stats = result.stats.to_dict()
    return {
        "index": index,
        "question_id": example.get("id", ""),
        "level": example.get("level", ""),
        "type": example.get("type", ""),
        "method": method,
        "question": example.get("question", ""),
        "prediction": result.answer,
        "gold_answer": gold,
        "exact_match": exact_match(result.answer, gold),
        "f1": token_f1(result.answer, gold),
        "supporting_title_recall": evidence_title_recall(result, example),
        **stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full FARR on HotpotQA.")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--sample-offset",
        type=int,
        default=0,
        help="Skip this many examples after level filtering.",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--config", default="distractor")
    parser.add_argument("--level", default="hard", choices=["", "easy", "medium", "hard"])
    parser.add_argument(
        "--methods",
        default="closedbook,rag,flare,rarr,ircot,farr,fullcontext",
        help="Comma-separated comparison methods.",
    )
    parser.add_argument("--llm", default="local", choices=["local", "openai"])
    parser.add_argument("--model", default=DEFAULT_LOCAL_MODEL)
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument(
        "--selector-path",
        default=None,
        help="Joblib artifact required by method=farr-learned.",
    )
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = [method.strip().lower() for method in args.methods.split(",") if method.strip()]
    supported = {
        "closedbook",
        "rag",
        "flare",
        "rarr",
        "ircot",
        "farr",
        "farr-v1",
        "farr-v2",
        "farr-learned",
        "farr-no-lookahead",
        "farr-no-revision",
        "farr-no-decomposition",
        "farr-conservative",
        "fullcontext",
    }
    unknown = set(methods) - supported
    if unknown:
        raise ValueError(f"Unsupported methods: {sorted(unknown)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "hotpotqa_results.jsonl"
    csv_path = output_dir / "hotpotqa_results.csv"
    if args.overwrite:
        jsonl_path.unlink(missing_ok=True)
        csv_path.unlink(missing_ok=True)
    elif jsonl_path.exists() or csv_path.exists():
        raise FileExistsError("Output exists. Use --overwrite or choose another --output-dir.")

    examples = load_examples(args)
    if not examples:
        raise RuntimeError("No examples matched the requested filters.")
    print(f"Selected examples: {len(examples)}")
    llm = build_llm(args)

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
        verbose=args.verbose,
    )

    rows = []
    for index, example in enumerate(tqdm(examples, desc="HotpotQA"), 1):
        documents = context_documents(example)
        retriever = TfidfRetriever(documents)
        question = str(example.get("question", ""))

        for method in methods:
            if method == "closedbook":
                result = closed_book(llm, question)
            elif method == "rag":
                result = rag(
                    llm,
                    retriever,
                    question,
                    config.initial_top_k,
                    config.max_chars_per_doc,
                )
            elif method == "flare":
                result = flare(llm, retriever, question, config)
            elif method == "rarr":
                result = rarr(llm, retriever, question, config)
            elif method == "ircot":
                result = ircot(llm, retriever, question, config)
            elif method == "farr":
                result = FARR(retriever, llm, config).answer(question)
            elif method == "farr-v1":
                result = FARRV1(retriever, llm, config).answer(question)
            elif method == "farr-v2":
                result = FARRV2(retriever, llm, config).answer(question)
            elif method == "farr-learned":
                if not args.selector_path:
                    raise ValueError(
                        "--selector-path is required for method=farr-learned"
                    )
                result = FARRV2(
                    retriever,
                    llm,
                    replace(
                        config,
                        candidate_selector_path=args.selector_path,
                    ),
                ).answer(question)
            elif method == "farr-no-lookahead":
                result = FARR(
                    retriever,
                    llm,
                    replace(config, enable_adaptive_queries=False),
                ).answer(question)
            elif method == "farr-no-revision":
                result = FARR(
                    retriever,
                    llm,
                    replace(config, enable_verification=False),
                ).answer(question)
            elif method == "farr-no-decomposition":
                result = FARR(
                    retriever,
                    llm,
                    replace(config, enable_decomposition=False),
                ).answer(question)
            elif method == "farr-conservative":
                result = FARR(
                    retriever,
                    llm,
                    replace(config, revise_on_labels=("UNSUPPORTED",)),
                ).answer(question)
            else:
                result = full_context(
                    llm,
                    question,
                    documents,
                    config.max_chars_per_doc,
                )

            row = result_row(index, method, example, result)
            rows.append(row)
            write_jsonl(jsonl_path, row)
            write_csv(csv_path, row)
            print(
                f"\n[{index}/{len(examples)}] {method}: {result.answer!r} "
                f"(gold={row['gold_answer']!r}, F1={row['f1']:.3f})"
            )

    summary = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            "count": len(selected),
            "exact_match": sum(row["exact_match"] for row in selected) / len(selected),
            "f1": sum(row["f1"] for row in selected) / len(selected),
            "supporting_title_recall": (
                sum(row["supporting_title_recall"] for row in selected) / len(selected)
            ),
            "avg_llm_calls": sum(row["llm_calls"] for row in selected) / len(selected),
            "avg_retrieval_calls": (
                sum(row["retrieval_calls"] for row in selected) / len(selected)
            ),
        }
    summary_path = output_dir / "hotpotqa_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Results: {jsonl_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
