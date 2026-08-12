from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from . import compat  # noqa: F401
from .contracts import answer_contract_compliance, heuristic_contract
from .oracle_router import answer_shape, normalize_answer
from farr.documents import content_tokens, doc_text, doc_title


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")
TRACE_BOUNDARY = re.compile(r"\s*\|\|\|\s*")
WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class VerifierConfig:
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    nli_model: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    lexical_top_k: int = 8
    neural_top_k: int = 3
    max_trace_claims: int = 3
    max_context_units: int = 96
    max_unit_chars: int = 850
    max_claim_chars: int = 520
    reranker_max_length: int = 512
    nli_max_length: int = 512
    reranker_batch_size: int = 32
    nli_batch_size: int = 64

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceUnit:
    title: str
    text: str

    @property
    def passage(self) -> str:
        return f"{self.title}: {self.text}" if self.title else self.text


@dataclass(frozen=True)
class Claim:
    kind: str
    text: str


def _clean(value: str, limit: int) -> str:
    return WHITESPACE.sub(" ", str(value)).strip()[:limit]


def _extract_json_answer(value: str) -> str:
    text = str(value).strip()
    if not text.startswith("{"):
        return text
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(
            r'\"answer\"\s*:\s*\"(.+?)(?:\"\s*,\s*\"|\"\s*})',
            text,
        )
        return match.group(1) if match else text
    if isinstance(parsed, dict) and parsed.get("answer"):
        return str(parsed["answer"])
    return text


def split_trace(value: str, max_chars: int = 520) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for block in TRACE_BOUNDARY.split(str(value or "")):
        block = _extract_json_answer(block)
        for sentence in SENTENCE_BOUNDARY.split(block):
            sentence = _clean(sentence, max_chars)
            normalized = normalize_answer(sentence)
            if len(normalized.split()) < 3 or normalized in seen:
                continue
            seen.add(normalized)
            result.append(sentence)
    return result


def evidence_units(
    documents: Sequence[Any],
    *,
    max_units: int = 96,
    max_chars: int = 850,
) -> list[EvidenceUnit]:
    result: list[EvidenceUnit] = []
    seen: set[tuple[str, str]] = set()
    for document in documents:
        title = _clean(doc_title(document), 180)
        text = _clean(doc_text(document), max_chars * 8)
        sentences = SENTENCE_BOUNDARY.split(text)
        if not sentences:
            sentences = [text]
        for sentence in sentences:
            sentence = _clean(sentence, max_chars)
            key = (normalize_answer(title), normalize_answer(sentence))
            if not sentence or key in seen:
                continue
            seen.add(key)
            result.append(EvidenceUnit(title=title, text=sentence))
            if len(result) >= max_units:
                return result
    return result


def _overlap(left: str, right: str) -> float:
    left_terms = content_tokens(left)
    right_terms = content_tokens(right)
    return len(left_terms & right_terms) / max(len(left_terms), 1)


def candidate_claims(
    *,
    question: str,
    answer: str,
    trace: str,
    max_trace_claims: int = 3,
    max_claim_chars: int = 520,
) -> list[Claim]:
    trace_sentences = split_trace(trace, max_claim_chars)
    answer_tokens = content_tokens(answer)
    question_tokens = content_tokens(question)

    def score(sentence: str) -> tuple[float, float, int]:
        terms = content_tokens(sentence)
        answer_overlap = len(answer_tokens & terms) / max(
            len(answer_tokens), 1
        )
        question_overlap = len(question_tokens & terms) / max(
            len(question_tokens), 1
        )
        exact = int(
            bool(normalize_answer(answer))
            and normalize_answer(answer) in normalize_answer(sentence)
        )
        return exact + answer_overlap, question_overlap, -len(sentence)

    if trace_sentences:
        answer_sentence = max(trace_sentences, key=score)
        if score(answer_sentence)[0] <= 0.0:
            answer_sentence = (
                f'The answer to the question "{_clean(question, 280)}" '
                f'is "{_clean(answer, 160)}".'
            )
    else:
        answer_sentence = (
            f'The answer to the question "{_clean(question, 280)}" '
            f'is "{_clean(answer, 160)}".'
        )

    ranked = sorted(
        (
            sentence
            for sentence in trace_sentences
            if normalize_answer(sentence)
            != normalize_answer(answer_sentence)
        ),
        key=lambda sentence: (
            _overlap(f"{question} {answer}", sentence),
            score(sentence),
        ),
        reverse=True,
    )[:max_trace_claims]
    return [
        Claim("answer", _clean(answer_sentence, max_claim_chars)),
        *[
            Claim("trace", _clean(sentence, max_claim_chars))
            for sentence in ranked
        ],
    ]


def lexical_prefilter(
    *,
    question: str,
    answer: str,
    claim: str,
    units: Sequence[EvidenceUnit],
    top_k: int,
) -> list[int]:
    claim_terms = content_tokens(f"{claim} {answer}")
    question_terms = content_tokens(question)
    normalized_answer = normalize_answer(answer)
    scored = []
    for index, unit in enumerate(units):
        passage = unit.passage
        terms = content_tokens(passage)
        claim_overlap = len(claim_terms & terms) / max(
            len(claim_terms), 1
        )
        question_overlap = len(question_terms & terms) / max(
            len(question_terms), 1
        )
        exact = float(
            bool(normalized_answer)
            and normalized_answer in normalize_answer(passage)
        )
        title_overlap = len(
            question_terms & content_tokens(unit.title)
        ) / max(len(question_terms), 1)
        score = (
            0.50 * claim_overlap
            + 0.25 * question_overlap
            + 0.15 * exact
            + 0.10 * title_overlap
        )
        scored.append((score, -index, index))
    return [value[2] for value in sorted(scored, reverse=True)[:top_k]]


class TransformerEvidenceScorer:
    def __init__(
        self,
        config: VerifierConfig,
        *,
        device: str = "cuda",
        local_files_only: bool = True,
    ) -> None:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        self.config = config
        self.device = torch.device(device)
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            config.reranker_model,
            local_files_only=local_files_only,
        )
        self.reranker = AutoModelForSequenceClassification.from_pretrained(
            config.reranker_model,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device).eval()
        self.nli_tokenizer = AutoTokenizer.from_pretrained(
            config.nli_model,
            local_files_only=local_files_only,
        )
        self.nli = AutoModelForSequenceClassification.from_pretrained(
            config.nli_model,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device).eval()
        labels = {
            str(value).lower(): int(index)
            for index, value in self.nli.config.id2label.items()
        }
        self.entailment_index = labels.get("entailment")
        self.contradiction_index = labels.get("contradiction")
        if self.entailment_index is None or self.contradiction_index is None:
            raise ValueError(
                f"NLI labels are not recognized: {self.nli.config.id2label}"
            )

    def rerank(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> np.ndarray:
        import torch

        values: list[np.ndarray] = []
        for start in range(0, len(pairs), self.config.reranker_batch_size):
            batch = pairs[start:start + self.config.reranker_batch_size]
            encoded = self.reranker_tokenizer(
                [value[0] for value in batch],
                [value[1] for value in batch],
                padding=True,
                truncation=True,
                max_length=self.config.reranker_max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                logits = self.reranker(**encoded).logits.reshape(-1)
                scores = torch.sigmoid(logits)
            values.append(scores.float().cpu().numpy())
        return np.concatenate(values) if values else np.empty(0)

    def nli_scores(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> np.ndarray:
        import torch

        values: list[np.ndarray] = []
        for start in range(0, len(pairs), self.config.nli_batch_size):
            batch = pairs[start:start + self.config.nli_batch_size]
            encoded = self.nli_tokenizer(
                [value[0] for value in batch],
                [value[1] for value in batch],
                padding=True,
                truncation=True,
                max_length=self.config.nli_max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.inference_mode():
                probabilities = torch.softmax(
                    self.nli(**encoded).logits,
                    dim=-1,
                )
            values.append(probabilities.float().cpu().numpy())
        return np.vstack(values) if values else np.empty((0, 3))


def _aggregate_candidate(
    *,
    question: str,
    answer: str,
    trace: str,
    units: Sequence[EvidenceUnit],
    claims: Sequence[Claim],
    claim_results: Sequence[dict[str, Any]],
) -> dict[str, float]:
    contract = heuristic_contract(question)
    contract_ok, _ = answer_contract_compliance(answer, contract)
    normalized_answer = normalize_answer(answer)
    answer_tokens = content_tokens(answer)
    exact_units = sum(
        bool(normalized_answer)
        and normalized_answer in normalize_answer(unit.passage)
        for unit in units
    )
    coverage = [
        len(answer_tokens & content_tokens(unit.passage))
        / max(len(answer_tokens), 1)
        for unit in units
    ]
    answer_result = claim_results[0]
    trace_results = list(claim_results[1:])
    all_results = list(claim_results)
    margins = [float(value["margin_max"]) for value in all_results]
    entails = [float(value["entail_max"]) for value in all_results]
    contradictions = [
        float(value["contradiction_max"]) for value in all_results
    ]
    trace_margins = [
        float(value["margin_max"]) for value in trace_results
    ]
    trace_entails = [
        float(value["entail_max"]) for value in trace_results
    ]
    trace_contradictions = [
        float(value["contradiction_max"]) for value in trace_results
    ]
    best_titles = {
        str(value["best_title"])
        for value in all_results
        if str(value.get("best_title", ""))
    }
    trace_text = " ".join(value.text for value in claims)
    trace_alignment = (
        len(answer_tokens & content_tokens(trace_text))
        / max(len(answer_tokens), 1)
    )
    shape = answer_shape(answer)
    features = {
        "contract_ok": float(contract_ok),
        f"shape={shape}": 1.0,
        "answer_tokens_log": math.log1p(len(normalized_answer.split())),
        "answer_exact_evidence_units_log": math.log1p(exact_units),
        "answer_lexical_coverage_max": max(coverage, default=0.0),
        "answer_trace_alignment": trace_alignment,
        "answer_relevance_max": float(answer_result["rerank_max"]),
        "answer_relevance_mean": float(answer_result["rerank_mean"]),
        "answer_entail_max": float(answer_result["entail_max"]),
        "answer_entail_mean": float(answer_result["entail_mean"]),
        "answer_contradiction_max": float(
            answer_result["contradiction_max"]
        ),
        "answer_margin_max": float(answer_result["margin_max"]),
        "answer_weighted_support": float(
            answer_result["weighted_support"]
        ),
        "proof_entail_mean": float(np.mean(entails)),
        "proof_entail_min": float(np.min(entails)),
        "proof_margin_mean": float(np.mean(margins)),
        "proof_margin_min": float(np.min(margins)),
        "proof_contradiction_max": float(np.max(contradictions)),
        "proof_supported_fraction": float(
            np.mean(np.asarray(margins) > 0.15)
        ),
        "proof_evidence_title_diversity": float(len(best_titles)),
        "trace_entail_mean": float(
            np.mean(trace_entails) if trace_entails else 0.0
        ),
        "trace_entail_min": float(
            np.min(trace_entails) if trace_entails else 0.0
        ),
        "trace_margin_mean": float(
            np.mean(trace_margins) if trace_margins else 0.0
        ),
        "trace_margin_min": float(
            np.min(trace_margins) if trace_margins else 0.0
        ),
        "trace_contradiction_max": float(
            np.max(trace_contradictions)
            if trace_contradictions
            else 0.0
        ),
    }
    return features


def extract_question_features(
    *,
    question: str,
    candidates: Sequence[dict[str, Any]],
    documents: Sequence[Any],
    scorer: TransformerEvidenceScorer,
    config: VerifierConfig,
) -> list[dict[str, Any]]:
    units = evidence_units(
        documents,
        max_units=config.max_context_units,
        max_chars=config.max_unit_chars,
    )
    if not units:
        raise ValueError("Question has no usable evidence units.")

    candidate_claim_sets = [
        candidate_claims(
            question=question,
            answer=str(candidate["prediction"]),
            trace=str(candidate.get("intermediate_answers", "")),
            max_trace_claims=config.max_trace_claims,
            max_claim_chars=config.max_claim_chars,
        )
        for candidate in candidates
    ]
    rerank_pairs: list[tuple[str, str]] = []
    rerank_meta: list[tuple[int, int, int]] = []
    for candidate_index, (candidate, claims) in enumerate(
        zip(candidates, candidate_claim_sets)
    ):
        answer = str(candidate["prediction"])
        for claim_index, claim in enumerate(claims):
            indexes = lexical_prefilter(
                question=question,
                answer=answer,
                claim=claim.text,
                units=units,
                top_k=config.lexical_top_k,
            )
            query = (
                f"Question: {question}\nCandidate answer: {answer}\n"
                f"Claim: {claim.text}"
            )
            for unit_index in indexes:
                rerank_pairs.append((query, units[unit_index].passage))
                rerank_meta.append(
                    (candidate_index, claim_index, unit_index)
                )
    rerank_scores = scorer.rerank(rerank_pairs)
    grouped_rerank: dict[tuple[int, int], list[tuple[float, int]]] = {}
    for meta, score in zip(rerank_meta, rerank_scores):
        candidate_index, claim_index, unit_index = meta
        grouped_rerank.setdefault(
            (candidate_index, claim_index), []
        ).append((float(score), unit_index))

    nli_pairs: list[tuple[str, str]] = []
    nli_meta: list[tuple[int, int, float, int]] = []
    for candidate_index, claims in enumerate(candidate_claim_sets):
        for claim_index, claim in enumerate(claims):
            ranked = sorted(
                grouped_rerank[(candidate_index, claim_index)],
                reverse=True,
            )[:config.neural_top_k]
            for rerank_score, unit_index in ranked:
                nli_pairs.append((units[unit_index].passage, claim.text))
                nli_meta.append(
                    (
                        candidate_index,
                        claim_index,
                        rerank_score,
                        unit_index,
                    )
                )
    nli_scores = scorer.nli_scores(nli_pairs)
    grouped_nli: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for meta, probabilities in zip(nli_meta, nli_scores):
        candidate_index, claim_index, rerank_score, unit_index = meta
        entailment = float(probabilities[scorer.entailment_index])
        contradiction = float(
            probabilities[scorer.contradiction_index]
        )
        grouped_nli.setdefault(
            (candidate_index, claim_index), []
        ).append(
            {
                "rerank": rerank_score,
                "entail": entailment,
                "contradiction": contradiction,
                "margin": entailment - contradiction,
                "unit_index": unit_index,
            }
        )

    result = []
    for candidate_index, (candidate, claims) in enumerate(
        zip(candidates, candidate_claim_sets)
    ):
        claim_results = []
        claim_audit = []
        for claim_index, claim in enumerate(claims):
            values = grouped_nli[(candidate_index, claim_index)]
            best = max(values, key=lambda value: value["margin"])
            aggregated = {
                "rerank_max": max(value["rerank"] for value in values),
                "rerank_mean": float(
                    np.mean([value["rerank"] for value in values])
                ),
                "entail_max": max(value["entail"] for value in values),
                "entail_mean": float(
                    np.mean([value["entail"] for value in values])
                ),
                "contradiction_max": max(
                    value["contradiction"] for value in values
                ),
                "margin_max": max(value["margin"] for value in values),
                "weighted_support": max(
                    value["rerank"] * value["entail"]
                    for value in values
                ),
                "best_title": units[int(best["unit_index"])].title,
            }
            claim_results.append(aggregated)
            claim_audit.append(
                {
                    "kind": claim.kind,
                    "claim": claim.text,
                    "best_evidence_title": aggregated["best_title"],
                    "entailment": aggregated["entail_max"],
                    "contradiction": aggregated["contradiction_max"],
                    "support_margin": aggregated["margin_max"],
                }
            )
        features = _aggregate_candidate(
            question=question,
            answer=str(candidate["prediction"]),
            trace=str(candidate.get("intermediate_answers", "")),
            units=units,
            claims=claims,
            claim_results=claim_results,
        )
        result.append(
            {
                "dataset": str(candidate["dataset"]),
                "question_id": str(candidate["question_id"]),
                "source_split": str(candidate.get("source_split", "")),
                "experiment_split": str(
                    candidate.get("experiment_split", "")
                ),
                "method": str(candidate["method"]),
                "prediction": str(candidate["prediction"]),
                "features": features,
                "claim_audit": claim_audit,
            }
        )
    return result


def feature_schema(rows: Iterable[dict[str, Any]]) -> list[str]:
    names = {
        str(name)
        for row in rows
        for name in dict(row["features"]).keys()
    }
    return sorted(names)


def feature_vector(
    row: dict[str, Any],
    names: Sequence[str],
) -> np.ndarray:
    values = dict(row["features"])
    return np.asarray(
        [float(values.get(name, 0.0)) for name in names],
        dtype=np.float64,
    )


def duplicate_answer_groups(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for candidate in candidates:
        grouped.setdefault(
            normalize_answer(str(candidate["prediction"])), []
        ).append(str(candidate["method"]))
    return grouped


def candidate_answer_distribution(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, int]:
    return dict(
        Counter(
            normalize_answer(str(candidate["prediction"]))
            for candidate in candidates
        )
    )
