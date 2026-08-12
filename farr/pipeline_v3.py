from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from .baselines import flare, rag
from .config import FARRConfig
from .documents import doc_text, normalize, rerank_documents
from .parsing import _json_value, parse_queries
from .pipeline_v2 import (
    FARRV2,
    _audit_value,
    _merge_stats,
    clean_short_answer,
)
from .prompts_v2 import (
    candidate_audit_prompt,
    corrected_answer_verify_prompt,
    disagreement_query_prompt,
)
from .prompts_v3 import evidence_graph_prompt
from .types import Document, FARRResult, FARRStats, VerificationTrace


VALID_LABELS = {"SUPPORTED", "UNSUPPORTED", "UNCERTAIN"}


def parse_evidence_graph(raw: str) -> Dict[str, Any]:
    value = _json_value(raw)
    if not isinstance(value, dict):
        return {"question_type": "other", "hops": [], "complete": False, "answer": ""}

    parsed_hops = []
    for item in value.get("hops") or []:
        if not isinstance(item, dict):
            continue
        doc_id_raw = item.get("doc_id")
        match = re.search(r"\d+", str(doc_id_raw))
        doc_id = int(match.group()) if match else 0
        parsed_hops.append(
            {
                "relation": str(item.get("relation") or "").strip(),
                "value": str(item.get("value") or "").strip(),
                "doc_id": doc_id,
                "quote": str(item.get("quote") or "").strip(),
            }
        )

    return {
        "question_type": str(value.get("question_type") or "other").lower(),
        "hops": parsed_hops,
        "complete": bool(value.get("complete", False)),
        "answer": str(value.get("answer") or "").strip(),
    }


def validate_evidence_graph(
    graph: Dict[str, Any],
    evidence: Sequence[Document],
) -> Tuple[bool, List[Dict[str, Any]], str]:
    if not graph.get("complete"):
        return False, [], "graph marked incomplete"

    validated = []
    for hop in graph.get("hops") or []:
        doc_id = int(hop.get("doc_id") or 0)
        quote = normalize(hop.get("quote") or "")
        relation = str(hop.get("relation") or "").strip()
        value = str(hop.get("value") or "").strip()
        if not 1 <= doc_id <= len(evidence):
            continue
        if len(quote.split()) < 3 or not relation or not value:
            continue
        document = normalize(doc_text(evidence[doc_id - 1]))
        if quote not in document:
            continue
        validated.append(hop)

    unique_docs = {hop["doc_id"] for hop in validated}
    answer = str(graph.get("answer") or "").strip()
    valid = (
        len(validated) >= 2
        and len(unique_docs) >= 2
        and bool(answer)
        and len(answer.split()) <= 15
    )
    reason = (
        f"{len(validated)}/{len(graph.get('hops') or [])} quotes validated; "
        f"{len(unique_docs)} distinct documents"
    )
    return valid, validated, reason


def parse_selector(raw: str) -> Dict[str, str]:
    value = _json_value(raw)
    if not isinstance(value, dict):
        return {
            "candidate_a_label": "UNCERTAIN",
            "candidate_b_label": "UNCERTAIN",
            "candidate_c_label": "UNCERTAIN",
            "selected": "A",
            "reason": str(raw).strip(),
        }
    result = {
        "candidate_a_label": str(value.get("candidate_a_label") or "UNCERTAIN").upper(),
        "candidate_b_label": str(value.get("candidate_b_label") or "UNCERTAIN").upper(),
        "candidate_c_label": str(value.get("candidate_c_label") or "UNCERTAIN").upper(),
        "selected": str(value.get("selected") or "A").upper(),
        "reason": str(value.get("reason") or "").strip(),
    }
    for key in ("candidate_a_label", "candidate_b_label", "candidate_c_label"):
        if result[key] not in VALID_LABELS:
            result[key] = "UNCERTAIN"
    if result["selected"] not in {"A", "B", "C"}:
        result["selected"] = "A"
    return result


class FARRV3(FARRV2):
    """Evidence-graph FARR with locally validated citations."""

    def _best_supported(
        self,
        question: str,
        candidates: Dict[str, str],
        labels: Dict[str, str],
        selected: str,
        evidence: Sequence[Document],
        graph_valid: bool,
    ) -> str:
        allowed = ["A", "B", "C"] if graph_valid else ["A", "B"]
        supported = [key for key in allowed if labels[key] == "SUPPORTED"]
        if selected in supported:
            return selected
        if supported:
            return max(
                supported,
                key=lambda key: self._evidence_score(candidates[key], evidence),
            )

        # A stable fallback: RAG is stronger on comparison disagreements;
        # otherwise use evidence overlap without allowing a free-form rewrite.
        if "comparison" in question.lower() or question.lower().startswith(
            ("are ", "were ", "is ", "was ")
        ):
            return "A"
        return max(
            ["A", "B"],
            key=lambda key: self._evidence_score(candidates[key], evidence),
        )

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

        flare_result = flare(
            self.llm,
            self.retriever,
            question,
            self.config,
            confidence_threshold=self.config.flare_confidence_threshold,
            max_steps=self.config.flare_max_steps,
        )
        _merge_stats(stats, flare_result.stats)

        rag_answer = clean_short_answer(question, rag_result.answer)
        flare_answer = clean_short_answer(question, flare_result.answer, rag_answer)
        stats.candidate_answers = {"rag": rag_answer, "flare": flare_answer}
        stats.draft_answer = flare_answer

        evidence = rerank_documents(
            question,
            [*rag_result.evidence, *flare_result.evidence],
            self.config.fusion_evidence_top_k,
        )
        if normalize(rag_answer) == normalize(flare_answer):
            stats.selected_candidate = "CONSENSUS"
            stats.final_verification_label = "CONSENSUS"
            return FARRResult(
                answer=rag_answer if len(rag_answer) <= len(flare_answer) else flare_answer,
                evidence=evidence,
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
            [f"{question} {rag_answer}", f"{question} {flare_answer}"],
            min(2, self.config.max_queries_per_hop),
        )
        fresh: List[Document] = []
        for query in queries:
            fresh.extend(self._retrieve(query, self.config.verification_top_k, stats))
        evidence = rerank_documents(
            f"{question} {rag_answer} {flare_answer}",
            [*fresh, *evidence],
            self.config.fusion_evidence_top_k,
        )

        # Preserve the proven v2 A/B selector. The graph is not allowed to
        # influence ordinary A/B disagreements because quote existence alone
        # does not prove that the inferred relation is correct.
        audit = _audit_value(
            self._llm(
                candidate_audit_prompt(
                    question,
                    rag_answer,
                    flare_answer,
                    evidence,
                    self.config.max_chars_per_doc,
                ),
                stats,
            )
        )
        both_refuted = (
            audit["candidate_a_label"]
            == audit["candidate_b_label"]
            == "UNSUPPORTED"
        )

        # Evidence-graph generation is expensive and useful only when the
        # ordinary v2 audit explicitly refutes both candidates.
        graph: Dict[str, Any] = {
            "question_type": "other",
            "hops": [],
            "complete": False,
            "answer": "",
        }
        graph_valid = False
        validated_hops: List[Dict[str, Any]] = []
        graph_answer = ""
        validation_reason = "graph skipped: original candidates not both refuted"
        if both_refuted:
            graph = parse_evidence_graph(
                self._llm(
                    evidence_graph_prompt(
                        question,
                        rag_answer,
                        flare_answer,
                        evidence,
                        self.config.max_chars_per_doc,
                    ),
                    stats,
                )
            )
            graph_valid, validated_hops, validation_reason = validate_evidence_graph(
                graph,
                evidence,
            )
            graph["hops"] = validated_hops
            graph_answer = clean_short_answer(question, graph.get("answer", ""))
            if not graph_answer:
                graph_valid = False

        stats.evidence_graph_valid = graph_valid
        stats.evidence_graph_hops = len(validated_hops)
        stats.candidate_answers["graph"] = graph_answer

        candidates = {"A": rag_answer, "B": flare_answer, "C": graph_answer}
        labels = {
            "A": audit["candidate_a_label"],
            "B": audit["candidate_b_label"],
            "C": "UNCERTAIN" if graph_valid else "UNSUPPORTED",
        }

        fallback_key, fallback_answer = self._fallback_selection(
            rag_answer,
            flare_answer,
            evidence,
        )
        chosen = audit["selected"]
        answer = (
            rag_answer if chosen == "A"
            else flare_answer if chosen == "B"
            else clean_short_answer(question, audit["answer"], fallback_answer)
        )
        final_label = (
            labels["A"] if chosen == "A"
            else labels["B"] if chosen == "B"
            else "REVISED"
        )
        revision_count = int(chosen == "REVISE")

        # V2 guard: never keep an A/B choice that the audit itself did not
        # support when the other candidate is supported.
        if chosen in {"A", "B"} and labels[chosen] != "SUPPORTED":
            other = "B" if chosen == "A" else "A"
            if labels[other] == "SUPPORTED":
                chosen = other
                answer = candidates[other]
                final_label = labels[other]
            else:
                chosen = fallback_key
                answer = fallback_answer
                final_label = "UNCERTAIN"

        # Graph correction is an exception, not the default selector. It must
        # satisfy all of: valid citations, both original candidates explicitly
        # contradicted, a genuinely new answer, and independent verification.
        graph_is_new = (
            graph_valid
            and normalize(graph_answer) not in {
                normalize(rag_answer),
                normalize(flare_answer),
            }
        )
        if graph_is_new and both_refuted:
            verify_value = _json_value(
                self._llm(
                    corrected_answer_verify_prompt(
                        question,
                        graph_answer,
                        evidence,
                        self.config.max_chars_per_doc,
                    ),
                    stats,
                )
            )
            graph_label = (
                str(verify_value.get("label") or "UNCERTAIN").upper()
                if isinstance(verify_value, dict)
                else "UNCERTAIN"
            )
            labels["C"] = graph_label
            if graph_label == "SUPPORTED":
                chosen = "C"
                answer = graph_answer
                final_label = "SUPPORTED"
                revision_count = 1

        # A free revision proposed by v2 is still bounded to one round and must
        # pass independent verification; otherwise return to the stable
        # evidence-overlap fallback.
        if chosen == "REVISE":
            verify_value = _json_value(
                self._llm(
                    corrected_answer_verify_prompt(
                        question,
                        answer,
                        evidence,
                        self.config.max_chars_per_doc,
                    ),
                    stats,
                )
            )
            revise_label = (
                str(verify_value.get("label") or "UNCERTAIN").upper()
                if isinstance(verify_value, dict)
                else "UNCERTAIN"
            )
            if revise_label == "SUPPORTED":
                final_label = "REVISED"
            else:
                chosen = fallback_key
                answer = fallback_answer
                final_label = revise_label
                revision_count = 0

        stats.selected_candidate = {
            "A": "RAG",
            "B": "FLARE",
            "C": "GRAPH",
            "REVISE": "REVISED",
        }[chosen]
        stats.revision_count = revision_count
        stats.final_verification_label = final_label
        stats.selection_reason = (
            f"{audit['reason']} | {validation_reason}"
        ).strip()
        stats.verification_traces.append(
            VerificationTrace(
                round=1,
                answer=f"A={rag_answer} || B={flare_answer} || C={graph_answer}",
                label=f"A:{labels['A']}|B:{labels['B']}|C:{labels['C']}",
                rationale=stats.selection_reason,
                queries=queries,
            )
        )

        return FARRResult(
            answer=clean_short_answer(question, answer, rag_answer),
            evidence=evidence,
            stats=stats,
        )

    __call__ = answer
