from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from . import compat  # noqa: F401
from .contracts import (
    QuestionContract,
    answer_contract_compliance,
    parse_contract,
)
from .prompts import (
    adjudication_prompt,
    contract_prompt,
    repair_prompt,
    verification_prompt,
)
from farr.baselines import ircot
from farr.config import FARRConfig
from farr.documents import (
    dedupe_documents,
    doc_text,
    normalize,
    rerank_documents,
)
from farr.parsing import _json_value
from farr.pipeline_v2 import FARRV2, clean_short_answer


VALID_LABELS = {"SUPPORTED", "UNSUPPORTED", "UNCERTAIN"}


def _as_bool(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return fallback


@dataclass(frozen=True)
class STARConfig:
    evidence_top_k: int = 18
    targeted_top_k: int = 4
    max_targeted_queries: int = 4
    max_chars_per_doc: int = 900
    min_quote_words: int = 3
    enable_contract_retrieval: bool = True
    enable_repair: bool = True
    verify_repair: bool = True

    def __post_init__(self) -> None:
        for name in (
            "evidence_top_k",
            "targeted_top_k",
            "max_targeted_queries",
            "max_chars_per_doc",
            "min_quote_words",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1")


@dataclass
class STARStats:
    llm_calls: int = 0
    retrieval_calls: int = 0
    farr_llm_calls: int = 0
    ircot_llm_calls: int = 0
    star_llm_calls: int = 0
    farr_retrieval_calls: int = 0
    ircot_retrieval_calls: int = 0
    star_retrieval_calls: int = 0
    planned_hops: int = 0
    completed_hops: int = 0
    revision_count: int = 0
    final_verification_label: str = "UNCERTAIN"
    reasoning_type: str = "other"
    answer_type: str = "entity"
    answer_target: str = ""
    requires_comparison: bool = False
    return_compared_entity: bool = False
    allowed_answers: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    selected_expert: str = ""
    route: str = ""
    selection_reason: str = ""
    farr_answer: str = ""
    ircot_answer: str = ""
    flare_answer: str = ""
    farr_contract_ok: bool = False
    ircot_contract_ok: bool = False
    repair_attempted: bool = False
    router_confidence: float = 0.0
    router_probabilities: str = ""
    adjudicator_selected: str = ""
    farr_adjudication_label: str = ""
    ircot_adjudication_label: str = ""
    farr_adjudicated_target_match: bool = False
    ircot_adjudicated_target_match: bool = False
    evidence_graph_valid: bool = False
    evidence_graph_hops: int = 0
    contract_queries: list[str] = field(default_factory=list)
    missing_queries: list[str] = field(default_factory=list)
    intermediate_answers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "retrieval_calls": self.retrieval_calls,
            "planned_hops": self.planned_hops,
            "completed_hops": self.completed_hops,
            "revision_count": self.revision_count,
            "revision_applied": int(self.revision_count > 0),
            "final_verification_label": self.final_verification_label,
            "rag_candidate": "",
            "flare_candidate": self.flare_answer,
            "selected_candidate": self.selected_expert,
            "draft_answer": self.ircot_answer,
            "evidence_graph_valid": int(self.evidence_graph_valid),
            "evidence_graph_hops": self.evidence_graph_hops,
            "selection_reason": self.selection_reason,
            "selector_used": int(
                self.route
                in {
                    "oracle_distilled_router",
                    "evidence_pairwise_ranker",
                }
            ),
            "selector_confidence": self.router_confidence,
            "router_probabilities": self.router_probabilities,
            "hop_queries": " ||| ".join(
                [*self.contract_queries, *self.missing_queries]
            ),
            "intermediate_answers": " ||| ".join(self.intermediate_answers),
            "reasoning_type": self.reasoning_type,
            "answer_type": self.answer_type,
            "answer_target": self.answer_target,
            "requires_comparison": int(self.requires_comparison),
            "return_compared_entity": int(self.return_compared_entity),
            "allowed_answers": " ||| ".join(self.allowed_answers),
            "contract_constraints": " ||| ".join(self.constraints),
            "selected_expert": self.selected_expert,
            "route": self.route,
            "farr_answer": self.farr_answer,
            "ircot_answer": self.ircot_answer,
            "farr_contract_ok": int(self.farr_contract_ok),
            "ircot_contract_ok": int(self.ircot_contract_ok),
            "repair_attempted": int(self.repair_attempted),
            "adjudicator_selected": self.adjudicator_selected,
            "farr_adjudication_label": self.farr_adjudication_label,
            "ircot_adjudication_label": self.ircot_adjudication_label,
            "farr_adjudicated_target_match": int(
                self.farr_adjudicated_target_match
            ),
            "ircot_adjudicated_target_match": int(
                self.ircot_adjudicated_target_match
            ),
            "farr_llm_calls": self.farr_llm_calls,
            "ircot_llm_calls": self.ircot_llm_calls,
            "star_llm_calls": self.star_llm_calls,
            "farr_retrieval_calls": self.farr_retrieval_calls,
            "ircot_retrieval_calls": self.ircot_retrieval_calls,
            "star_retrieval_calls": self.star_retrieval_calls,
        }


@dataclass
class STARResult:
    answer: str
    evidence: list[Any]
    stats: STARStats
    contract: QuestionContract
    farr_result: Any
    ircot_result: Any


def _label(value: Any) -> str:
    result = str(value or "UNCERTAIN").upper()
    return result if result in VALID_LABELS else "UNCERTAIN"


def parse_adjudication(raw: str) -> dict[str, Any]:
    value = _json_value(raw)
    if not isinstance(value, dict):
        return {
            "candidate_a_label": "UNCERTAIN",
            "candidate_a_target_match": False,
            "candidate_b_label": "UNCERTAIN",
            "candidate_b_target_match": False,
            "selected": "UNCERTAIN",
            "answer": "",
            "missing_queries": [],
            "reason": str(raw).strip(),
        }
    selected = str(value.get("selected") or "UNCERTAIN").upper()
    if selected not in {"A", "B", "SYNTHESIZE", "UNCERTAIN"}:
        selected = "UNCERTAIN"
    missing = value.get("missing_queries") or []
    if not isinstance(missing, list):
        missing = []
    return {
        "candidate_a_label": _label(value.get("candidate_a_label")),
        "candidate_a_target_match": _as_bool(
            value.get("candidate_a_target_match"),
        ),
        "candidate_b_label": _label(value.get("candidate_b_label")),
        "candidate_b_target_match": _as_bool(
            value.get("candidate_b_target_match"),
        ),
        "selected": selected,
        "answer": str(value.get("answer") or "").strip(),
        "missing_queries": [
            " ".join(str(query).split())
            for query in missing[:4]
            if str(query).strip()
        ],
        "reason": str(value.get("reason") or "").strip(),
    }


def parse_solution(raw: str) -> dict[str, Any]:
    value = _json_value(raw)
    if not isinstance(value, dict):
        return {
            "answer": "",
            "target_match": False,
            "complete": False,
            "steps": [],
        }
    steps = []
    for item in value.get("steps") or []:
        if not isinstance(item, dict):
            continue
        try:
            doc_id = int(item.get("doc_id") or 0)
        except (TypeError, ValueError):
            doc_id = 0
        steps.append(
            {
                "claim": str(item.get("claim") or "").strip(),
                "doc_id": doc_id,
                "quote": str(item.get("quote") or "").strip(),
            }
        )
    return {
        "answer": str(value.get("answer") or "").strip(),
        "target_match": _as_bool(value.get("target_match")),
        "complete": _as_bool(value.get("complete")),
        "steps": steps,
    }


def validate_solution(
    solution: dict[str, Any],
    contract: QuestionContract,
    evidence: Sequence[Any],
    min_quote_words: int,
) -> tuple[bool, list[dict[str, Any]], str]:
    answer_ok, answer_reason = answer_contract_compliance(
        solution.get("answer", ""),
        contract,
    )
    validated = []
    for step in solution.get("steps") or []:
        doc_id = int(step.get("doc_id") or 0)
        quote = normalize(step.get("quote") or "")
        claim = str(step.get("claim") or "").strip()
        if not 1 <= doc_id <= len(evidence):
            continue
        if len(quote.split()) < min_quote_words or not claim:
            continue
        if quote not in normalize(doc_text(evidence[doc_id - 1])):
            continue
        validated.append(step)

    required = min(2, max(1, contract.expected_hops))
    distinct_docs = {step["doc_id"] for step in validated}
    graph_ok = (
        bool(solution.get("complete"))
        and bool(solution.get("target_match"))
        and answer_ok
        and len(validated) >= required
        and len(distinct_docs) >= required
    )
    reason = (
        f"answer={answer_reason}; validated_steps={len(validated)}; "
        f"distinct_docs={len(distinct_docs)}; required={required}"
    )
    return graph_ok, validated, reason


class FARRSTAR:
    """Contract-gated dual reasoning with verified evidence repair.

    The decision rule has three stages:
    1. build a question/answer contract;
    2. keep IRCoT unless it violates the contract or evidence explicitly
       supports FARR while refuting IRCoT;
    3. accept a synthesized answer only after cited evidence repair verifies.
    """

    def __init__(
        self,
        retriever: Any,
        llm: Any,
        farr_config: FARRConfig | None = None,
        star_config: STARConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.farr_config = farr_config or FARRConfig()
        self.star_config = star_config or STARConfig()

    def _llm(self, prompt: str, stats: STARStats) -> str:
        stats.star_llm_calls += 1
        return str(self.llm(prompt)).strip()

    def _retrieve(
        self,
        query: str,
        top_k: int,
        stats: STARStats,
    ) -> list[Any]:
        stats.star_retrieval_calls += 1
        return list(self.retriever(query, top_k) or [])

    def _finish_stats(
        self,
        stats: STARStats,
        farr_result: Any,
        ircot_result: Any,
    ) -> None:
        stats.farr_llm_calls = int(farr_result.stats.llm_calls)
        stats.ircot_llm_calls = int(ircot_result.stats.llm_calls)
        stats.farr_retrieval_calls = int(farr_result.stats.retrieval_calls)
        stats.ircot_retrieval_calls = int(
            ircot_result.stats.retrieval_calls
        )
        stats.llm_calls = (
            stats.farr_llm_calls
            + stats.ircot_llm_calls
            + stats.star_llm_calls
        )
        stats.retrieval_calls = (
            stats.farr_retrieval_calls
            + stats.ircot_retrieval_calls
            + stats.star_retrieval_calls
        )
        if stats.completed_hops == 0:
            stats.completed_hops = max(
                int(farr_result.stats.completed_hops),
                int(ircot_result.stats.completed_hops),
            )
        stats.intermediate_answers = [
            *[
                str(hop.intermediate_answer)
                for hop in farr_result.stats.hop_traces
                if str(hop.intermediate_answer).strip()
            ],
            *[
                str(hop.intermediate_answer)
                for hop in ircot_result.stats.hop_traces
                if str(hop.intermediate_answer).strip()
            ],
        ]

    def answer(self, question: str) -> STARResult:
        question = " ".join(str(question).split())
        if not question:
            raise ValueError("question cannot be empty")
        stats = STARStats()

        contract = parse_contract(
            self._llm(contract_prompt(question), stats),
            question,
        )
        stats.reasoning_type = contract.reasoning_type
        stats.answer_type = contract.answer_type
        stats.answer_target = contract.target
        stats.requires_comparison = contract.requires_comparison
        stats.return_compared_entity = contract.return_compared_entity
        stats.allowed_answers = list(contract.allowed_answers)
        stats.constraints = list(contract.constraints)
        stats.planned_hops = contract.expected_hops
        stats.contract_queries = contract.retrieval_queries[
            : self.star_config.max_targeted_queries
        ]

        farr_result = FARRV2(
            self.retriever,
            self.llm,
            self.farr_config,
        ).answer(question)
        ircot_result = ircot(
            self.llm,
            self.retriever,
            question,
            self.farr_config,
        )
        stats.farr_answer = clean_short_answer(
            question,
            farr_result.answer,
        )
        stats.ircot_answer = clean_short_answer(
            question,
            ircot_result.answer,
        )
        stats.intermediate_answers = [
            *[
                str(hop.intermediate_answer)
                for hop in farr_result.stats.hop_traces
                if str(hop.intermediate_answer).strip()
            ],
            *[
                str(hop.intermediate_answer)
                for hop in ircot_result.stats.hop_traces
                if str(hop.intermediate_answer).strip()
            ],
        ]

        targeted = []
        if self.star_config.enable_contract_retrieval:
            for query in stats.contract_queries:
                targeted.extend(
                    self._retrieve(
                        query,
                        self.star_config.targeted_top_k,
                        stats,
                    )
                )
        evidence = rerank_documents(
            f"{question} {contract.target}",
            [
                *farr_result.evidence,
                *ircot_result.evidence,
                *targeted,
            ],
            self.star_config.evidence_top_k,
            bridge_context=" ".join(stats.intermediate_answers),
        )

        farr_check = answer_contract_compliance(
            stats.farr_answer,
            contract,
        )
        ircot_check = answer_contract_compliance(
            stats.ircot_answer,
            contract,
        )
        stats.farr_contract_ok = farr_check[0]
        stats.ircot_contract_ok = ircot_check[0]

        if (
            normalize(stats.farr_answer) == normalize(stats.ircot_answer)
            and stats.farr_contract_ok
        ):
            answer = (
                stats.farr_answer
                if len(stats.farr_answer) <= len(stats.ircot_answer)
                else stats.ircot_answer
            )
            stats.selected_expert = "consensus"
            stats.route = "contract_consensus"
            stats.selection_reason = "Experts agreed and the answer obeyed the contract."
            stats.final_verification_label = "CONSENSUS"
            self._finish_stats(stats, farr_result, ircot_result)
            return STARResult(
                answer=answer,
                evidence=dedupe_documents(evidence),
                stats=stats,
                contract=contract,
                farr_result=farr_result,
                ircot_result=ircot_result,
            )

        adjudication = parse_adjudication(
            self._llm(
                adjudication_prompt(
                    question,
                    contract,
                    farr_result,
                    ircot_result,
                    farr_check,
                    ircot_check,
                    evidence,
                    self.star_config.max_chars_per_doc,
                ),
                stats,
            )
        )
        stats.adjudicator_selected = adjudication["selected"]
        stats.farr_adjudication_label = adjudication[
            "candidate_a_label"
        ]
        stats.ircot_adjudication_label = adjudication[
            "candidate_b_label"
        ]
        stats.farr_adjudicated_target_match = adjudication[
            "candidate_a_target_match"
        ]
        stats.ircot_adjudicated_target_match = adjudication[
            "candidate_b_target_match"
        ]
        stats.missing_queries = adjudication["missing_queries"][
            : self.star_config.max_targeted_queries
        ]

        chosen_key = adjudication["selected"]
        chosen_result = (
            farr_result if chosen_key == "A"
            else ircot_result if chosen_key == "B"
            else None
        )
        chosen_ok = (
            stats.farr_contract_ok if chosen_key == "A"
            else stats.ircot_contract_ok if chosen_key == "B"
            else False
        )
        chosen_label = (
            adjudication["candidate_a_label"] if chosen_key == "A"
            else adjudication["candidate_b_label"] if chosen_key == "B"
            else "UNCERTAIN"
        )
        chosen_target = (
            adjudication["candidate_a_target_match"] if chosen_key == "A"
            else adjudication["candidate_b_target_match"] if chosen_key == "B"
            else False
        )
        force_chain_verification = (
            contract.reasoning_type == "comparison_chain"
            and normalize(stats.farr_answer) != normalize(stats.ircot_answer)
        )
        repair_needed = force_chain_verification or not (
            chosen_result is not None
            and chosen_ok
            and chosen_label == "SUPPORTED"
            and chosen_target
        )

        solution = {
            "answer": "",
            "target_match": False,
            "complete": False,
            "steps": [],
        }
        graph_ok = False
        graph_reason = (
            "repair required but disabled"
            if repair_needed
            else "repair not required"
        )
        verifier = {
            "label": "UNCERTAIN",
            "target_match": False,
            "reason": "",
        }
        if self.star_config.enable_repair and repair_needed:
            stats.repair_attempted = True
            fresh = []
            for query in stats.missing_queries:
                fresh.extend(
                    self._retrieve(
                        query,
                        self.star_config.targeted_top_k,
                        stats,
                    )
                )
            evidence = rerank_documents(
                f"{question} {contract.target}",
                [*fresh, *evidence],
                self.star_config.evidence_top_k,
                bridge_context=" ".join(stats.intermediate_answers),
            )
            solution = parse_solution(
                self._llm(
                    repair_prompt(
                        question,
                        contract,
                        stats.farr_answer,
                        stats.ircot_answer,
                        evidence,
                        self.star_config.max_chars_per_doc,
                    ),
                    stats,
                )
            )
            graph_ok, validated, graph_reason = validate_solution(
                solution,
                contract,
                evidence,
                self.star_config.min_quote_words,
            )
            solution["steps"] = validated
            stats.evidence_graph_valid = graph_ok
            stats.evidence_graph_hops = len(validated)
            stats.completed_hops = len(validated)

            if graph_ok and self.star_config.verify_repair:
                raw_verify = _json_value(
                    self._llm(
                        verification_prompt(
                            question,
                            contract,
                            solution,
                            evidence,
                            self.star_config.max_chars_per_doc,
                        ),
                        stats,
                    )
                )
                if isinstance(raw_verify, dict):
                    verifier = {
                        "label": _label(raw_verify.get("label")),
                        "target_match": _as_bool(
                            raw_verify.get("target_match"),
                        ),
                        "reason": str(
                            raw_verify.get("reason") or ""
                        ).strip(),
                    }
            elif graph_ok:
                verifier = {
                    "label": "SUPPORTED",
                    "target_match": True,
                    "reason": "independent verification disabled",
                }

        if (
            graph_ok
            and verifier["label"] == "SUPPORTED"
            and verifier["target_match"]
        ):
            answer = clean_short_answer(
                question,
                solution["answer"],
            )
            stats.selected_expert = "synthesized"
            stats.route = "evidence_repair"
            stats.selection_reason = (
                f"{adjudication['reason']} | {graph_reason} | "
                f"{verifier['reason']}"
            ).strip(" |")
            stats.final_verification_label = verifier["label"]
            stats.revision_count = 1
        else:
            farr_verified = (
                stats.farr_contract_ok
                and adjudication["candidate_a_label"] == "SUPPORTED"
                and adjudication["candidate_a_target_match"]
            )
            ircot_refuted = (
                adjudication["candidate_b_label"] == "UNSUPPORTED"
                or not adjudication["candidate_b_target_match"]
            )
            if not stats.ircot_contract_ok and stats.farr_contract_ok:
                result = farr_result
                stats.selected_expert = "farr"
                stats.route = "contract_gate"
            elif farr_verified and ircot_refuted:
                result = farr_result
                stats.selected_expert = "farr"
                stats.route = "verified_farr"
            else:
                result = ircot_result
                stats.selected_expert = "ircot"
                stats.route = "ircot_default"
            answer = clean_short_answer(question, result.answer)
            stats.selection_reason = (
                f"{adjudication['reason']} | {graph_reason} | "
                f"{verifier['reason']}"
            ).strip(" |")
            stats.final_verification_label = adjudication[
                "candidate_a_label"
                if stats.selected_expert == "farr"
                else "candidate_b_label"
            ]

        self._finish_stats(stats, farr_result, ircot_result)
        return STARResult(
            answer=answer,
            evidence=dedupe_documents(evidence),
            stats=stats,
            contract=contract,
            farr_result=farr_result,
            ircot_result=ircot_result,
        )

    __call__ = answer
