from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from .baselines import flare, rag
from .config import FARRConfig
from .documents import dedupe_documents, doc_text, normalize, rerank_documents
from .parsing import _json_value, parse_queries, parse_revised_answer
from .prompts import answer_type_instruction
from .prompts_v2 import (
    candidate_audit_prompt,
    corrected_answer_verify_prompt,
    disagreement_query_prompt,
)
from .selector import CandidateSelector
from .types import Document, FARRResult, FARRStats, VerificationTrace


def _is_yes_no(question: str) -> bool:
    q = " ".join(str(question).lower().split())
    prefixes = (
        "is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ",
        "could ", "has ", "have ", "had ", "will ", "would ",
    )
    return q.startswith(prefixes) or any(
        marker in f" {q} "
        for marker in (" determine if ", " determine whether ", " decide if ")
    )


def clean_short_answer(question: str, raw: str, fallback: str = "") -> str:
    answer = parse_revised_answer(raw).strip()
    answer = re.sub(r"^\s*(?:final\s+answer|answer)\s*:\s*", "", answer, flags=re.I)
    answer = answer.strip().strip("`").strip().strip("\"'")

    if _is_yes_no(question):
        match = re.match(r"^\s*(yes|no)\b", answer, flags=re.I)
        if match:
            return match.group(1).lower()
        fallback_match = re.match(r"^\s*(yes|no)\b", fallback, flags=re.I)
        if fallback_match:
            return fallback_match.group(1).lower()

    first_line = answer.splitlines()[0].strip() if answer else ""
    if not first_line:
        return fallback.strip()
    if len(first_line.split()) > 15 and fallback:
        return fallback.strip()
    return first_line


def _merge_stats(target: FARRStats, source: FARRStats) -> None:
    target.llm_calls += source.llm_calls
    target.retrieval_calls += source.retrieval_calls
    target.planned_hops += source.planned_hops
    target.completed_hops += source.completed_hops
    target.hop_traces.extend(source.hop_traces)


def _audit_value(raw: str) -> Dict[str, str]:
    value = _json_value(raw)
    if not isinstance(value, dict):
        return {
            "candidate_a_label": "UNCERTAIN",
            "candidate_b_label": "UNCERTAIN",
            "selected": "A",
            "answer": "",
            "reason": str(raw).strip(),
        }
    selected = str(value.get("selected") or "A").upper()
    if selected not in {"A", "B", "REVISE"}:
        selected = "A"
    result = {
        "candidate_a_label": str(
            value.get("candidate_a_label") or "UNCERTAIN"
        ).upper(),
        "candidate_b_label": str(
            value.get("candidate_b_label") or "UNCERTAIN"
        ).upper(),
        "selected": selected,
        "answer": str(value.get("answer") or "").strip(),
        "reason": str(value.get("reason") or "").strip(),
    }
    valid = {"SUPPORTED", "UNSUPPORTED", "UNCERTAIN"}
    for key in ("candidate_a_label", "candidate_b_label"):
        if result[key] not in valid:
            result[key] = "UNCERTAIN"
    return result


class FARRV2:
    """Candidate-preserving FLARE + retrieval-augmented revision.

    V2 uses RAG as a stable candidate, FLARE as the active-retrieval candidate,
    and invokes RARR-style research/auditing only when the candidates disagree.
    Revision is bounded to one evidence-grounded correction.
    """

    def __init__(self, retriever: Any, llm: Any, config: FARRConfig | None = None):
        self.retriever = retriever
        self.llm = llm
        self.config = config or FARRConfig()
        self.candidate_selector = (
            CandidateSelector(self.config.candidate_selector_path)
            if self.config.candidate_selector_path
            else None
        )

    def _llm(self, prompt: str, stats: FARRStats) -> str:
        stats.llm_calls += 1
        return str(self.llm(prompt)).strip()

    def _retrieve(self, query: str, top_k: int, stats: FARRStats) -> List[Document]:
        stats.retrieval_calls += 1
        return list(self.retriever(query, top_k) or [])

    def _evidence_score(self, answer: str, evidence: Sequence[Document]) -> float:
        candidate = normalize(answer)
        if not candidate:
            return -1.0
        corpus = normalize(" ".join(doc_text(doc) for doc in evidence))
        score = 2.0 if candidate in corpus else 0.0
        words = candidate.split()
        if words:
            score += sum(word in corpus.split() for word in words) / len(words)
        score -= max(0, len(words) - 8) * 0.05
        return score

    def _fallback_selection(
        self,
        rag_answer: str,
        flare_answer: str,
        evidence: Sequence[Document],
    ) -> tuple[str, str]:
        rag_score = self._evidence_score(rag_answer, evidence)
        flare_score = self._evidence_score(flare_answer, evidence)
        if flare_score > rag_score:
            return "B", flare_answer
        return "A", rag_answer

    def answer(self, question: str) -> FARRResult:
        question = " ".join(str(question).split())
        if not question:
            raise ValueError("question cannot be empty")

        stats = FARRStats()

        rag_result = rag(
            self.llm,
            self.retriever,
            question,
            self.config.initial_top_k,
            self.config.max_chars_per_doc,
        )
        _merge_stats(stats, rag_result.stats)

        if self.config.enable_adaptive_queries:
            flare_result = flare(
                self.llm,
                self.retriever,
                question,
                self.config,
                confidence_threshold=self.config.flare_confidence_threshold,
                max_steps=self.config.flare_max_steps,
            )
            _merge_stats(stats, flare_result.stats)
        else:
            flare_result = rag_result

        rag_answer = clean_short_answer(question, rag_result.answer)
        flare_answer = clean_short_answer(question, flare_result.answer, rag_answer)
        stats.candidate_answers = {"rag": rag_answer, "flare": flare_answer}
        stats.draft_answer = flare_answer

        combined_evidence = rerank_documents(
            question,
            [*rag_result.evidence, *flare_result.evidence],
            self.config.fusion_evidence_top_k,
        )

        if normalize(rag_answer) == normalize(flare_answer):
            stats.selected_candidate = "CONSENSUS"
            stats.final_verification_label = "CONSENSUS"
            return FARRResult(
                answer=rag_answer if len(rag_answer) <= len(flare_answer) else flare_answer,
                evidence=combined_evidence,
                stats=stats,
            )

        if not self.config.enable_verification:
            stats.selected_candidate = "FLARE"
            stats.final_verification_label = "NOT_RUN"
            return FARRResult(
                answer=flare_answer,
                evidence=combined_evidence,
                stats=stats,
            )

        raw_queries = self._llm(
            disagreement_query_prompt(
                question,
                rag_answer,
                flare_answer,
                min(2, self.config.max_queries_per_hop),
            ),
            stats,
        )
        queries = parse_queries(
            raw_queries,
            [
                f"{question} {rag_answer}",
                f"{question} {flare_answer}",
            ],
            min(2, self.config.max_queries_per_hop),
        )
        fresh: List[Document] = []
        for query in queries:
            fresh.extend(
                self._retrieve(query, self.config.verification_top_k, stats)
            )
        audit_evidence = rerank_documents(
            f"{question} {rag_answer} {flare_answer}",
            [*fresh, *combined_evidence],
            self.config.fusion_evidence_top_k,
        )

        audit = _audit_value(
            self._llm(
                candidate_audit_prompt(
                    question,
                    rag_answer,
                    flare_answer,
                    audit_evidence,
                    self.config.max_chars_per_doc,
                ),
                stats,
            )
        )
        stats.verification_traces.append(
            VerificationTrace(
                round=1,
                answer=f"A={rag_answer} || B={flare_answer}",
                label=(
                    f"A:{audit['candidate_a_label']}|"
                    f"B:{audit['candidate_b_label']}"
                ),
                rationale=audit["reason"],
                queries=queries,
            )
        )

        fallback_key, fallback_answer = self._fallback_selection(
            rag_answer,
            flare_answer,
            audit_evidence,
        )
        selected = audit["selected"]
        if selected == "A":
            answer = rag_answer
            label = audit["candidate_a_label"]
        elif selected == "B":
            answer = flare_answer
            label = audit["candidate_b_label"]
        else:
            answer = clean_short_answer(
                question,
                audit["answer"],
                fallback=fallback_answer,
            )
            label = "REVISED"
            stats.revision_count = 1

        # Guard against a malformed or unsupported selection.
        selected_label = (
            audit["candidate_a_label"] if selected == "A"
            else audit["candidate_b_label"] if selected == "B"
            else "SUPPORTED"
        )
        if selected in {"A", "B"} and selected_label != "SUPPORTED":
            other_key = "B" if selected == "A" else "A"
            other_label = (
                audit["candidate_b_label"] if other_key == "B"
                else audit["candidate_a_label"]
            )
            if other_label == "SUPPORTED":
                selected = other_key
                answer = flare_answer if other_key == "B" else rag_answer
                label = other_label
            else:
                selected = fallback_key
                answer = fallback_answer
                label = "UNCERTAIN"

        if stats.revision_count:
            verify_raw = self._llm(
                corrected_answer_verify_prompt(
                    question,
                    answer,
                    audit_evidence,
                    self.config.max_chars_per_doc,
                ),
                stats,
            )
            verify_value = _json_value(verify_raw)
            verify_label = (
                str(verify_value.get("label", "UNCERTAIN")).upper()
                if isinstance(verify_value, dict)
                else "UNCERTAIN"
            )
            if verify_label != "SUPPORTED":
                selected = fallback_key
                answer = fallback_answer
                label = verify_label
                stats.revision_count = 0

        # A selector trained only on development data may override the v2
        # decision when its confidence exceeds the cross-validated threshold.
        # Low-confidence cases retain the original audit/revision result.
        if self.candidate_selector is not None:
            selector_choice, selector_confidence = self.candidate_selector.choose(
                model_name=getattr(self.llm, "model_name", "other"),
                question=question,
                rag_answer=rag_answer,
                flare_answer=flare_answer,
            )
            stats.selector_confidence = selector_confidence
            if selector_confidence >= self.candidate_selector.threshold:
                selected = selector_choice
                answer = rag_answer if selected == "A" else flare_answer
                label = "LEARNED_SELECTOR"
                stats.revision_count = 0
                stats.selector_used = True

        stats.selected_candidate = {
            "A": "RAG",
            "B": "FLARE",
            "REVISE": "REVISED",
        }.get(selected, selected)
        stats.final_verification_label = label
        return FARRResult(
            answer=clean_short_answer(question, answer, fallback_answer),
            evidence=dedupe_documents(audit_evidence),
            stats=stats,
        )

    __call__ = answer
