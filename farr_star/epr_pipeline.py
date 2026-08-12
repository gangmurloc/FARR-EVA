from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import compat  # noqa: F401
from .contracts import heuristic_contract
from .evidence_ranker import CandidateObservation, EvidencePairwiseRouter
from .oracle_router import CANDIDATES
from .pipeline import STARStats
from farr.baselines import ircot
from farr.config import FARRConfig
from farr.documents import dedupe_documents
from farr.pipeline_v2 import FARRV2, clean_short_answer


@dataclass
class EPRResult:
    answer: str
    evidence: list[Any]
    stats: STARStats
    farr_result: Any
    ircot_result: Any


def _runtime_evidence(
    retriever: Any,
    farr_result: Any,
    ircot_result: Any,
) -> list[Any]:
    documents = getattr(retriever, "documents", None)
    if documents:
        return list(documents)
    return dedupe_documents(
        [*farr_result.evidence, *ircot_result.evidence]
    )


def _flare_row_from_farr(farr_result: Any) -> dict[str, Any]:
    """Reconstruct the standalone FLARE counters embedded in FARR."""
    traces = list(farr_result.stats.hop_traces)
    completed = len(traces)
    return {
        "llm_calls": 2 * completed + 1,
        "retrieval_calls": completed + 1,
        "planned_hops": completed,
        "completed_hops": completed,
        "revision_applied": 0,
        "final_verification_label": "UNCERTAIN",
        "selected_candidate": "",
        "intermediate_answers": " ||| ".join(
            str(trace.intermediate_answer) for trace in traces
        ),
    }


class FARREPR:
    """Shared evidence-aware pairwise ranking over three RAG experts."""

    def __init__(
        self,
        retriever: Any,
        llm: Any,
        router: EvidencePairwiseRouter,
        config: FARRConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.router = router
        self.config = config or FARRConfig()

    def answer(self, question: str) -> EPRResult:
        question = " ".join(str(question).split())
        if not question:
            raise ValueError("question cannot be empty")

        farr_result = FARRV2(
            self.retriever,
            self.llm,
            self.config,
        ).answer(question)
        ircot_result = ircot(
            self.llm,
            self.retriever,
            question,
            self.config,
        )
        answers = {
            "flare": clean_short_answer(
                question,
                farr_result.stats.candidate_answers.get(
                    "flare",
                    farr_result.answer,
                ),
                fallback=farr_result.answer,
            ),
            "ircot": clean_short_answer(
                question,
                ircot_result.answer,
            ),
            "farr-v2": clean_short_answer(
                question,
                farr_result.answer,
            ),
        }
        rows = {
            "flare": _flare_row_from_farr(farr_result),
            "ircot": ircot_result.stats.to_dict(),
            "farr-v2": farr_result.stats.to_dict(),
        }
        evidence = _runtime_evidence(
            self.retriever,
            farr_result,
            ircot_result,
        )
        observations = [
            CandidateObservation(
                question_id="runtime",
                dataset="runtime",
                question=question,
                method=method,
                answer=answers[method],
                row=rows[method],
                peer_answers=answers,
                evidence=evidence,
            )
            for method in CANDIDATES
        ]
        selected, confidence, utilities = self.router.choose(observations)
        public_selected = "farr" if selected == "farr-v2" else selected
        public_utilities = {
            ("farr" if name == "farr-v2" else name): value
            for name, value in utilities.items()
        }

        contract = heuristic_contract(question)
        stats = STARStats(
            llm_calls=(
                int(farr_result.stats.llm_calls)
                + int(ircot_result.stats.llm_calls)
            ),
            retrieval_calls=(
                int(farr_result.stats.retrieval_calls)
                + int(ircot_result.stats.retrieval_calls)
            ),
            farr_llm_calls=int(farr_result.stats.llm_calls),
            ircot_llm_calls=int(ircot_result.stats.llm_calls),
            farr_retrieval_calls=int(farr_result.stats.retrieval_calls),
            ircot_retrieval_calls=int(
                ircot_result.stats.retrieval_calls
            ),
            planned_hops=max(
                int(farr_result.stats.planned_hops),
                int(ircot_result.stats.planned_hops),
            ),
            completed_hops=max(
                int(farr_result.stats.completed_hops),
                int(ircot_result.stats.completed_hops),
            ),
            final_verification_label="EVIDENCE_PAIRED",
            reasoning_type=contract.reasoning_type,
            answer_type=contract.answer_type,
            answer_target=contract.target,
            requires_comparison=contract.requires_comparison,
            return_compared_entity=contract.return_compared_entity,
            allowed_answers=list(contract.allowed_answers),
            constraints=list(contract.constraints),
            selected_expert=public_selected,
            route="evidence_pairwise_ranker",
            router_confidence=confidence,
            router_probabilities=" ||| ".join(
                f"{name}={public_utilities.get(name, 0.0):.6f}"
                for name in ("flare", "ircot", "farr")
            ),
            selection_reason=(
                "Shared evidence-aware pairwise utility: "
                + ", ".join(
                    f"{name}={public_utilities.get(name, 0.0):.4f}"
                    for name in ("flare", "ircot", "farr")
                )
            ),
            farr_answer=answers["farr-v2"],
            ircot_answer=answers["ircot"],
            flare_answer=answers["flare"],
            intermediate_answers=[
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
            ],
        )
        return EPRResult(
            answer=answers[selected],
            evidence=dedupe_documents(
                [*farr_result.evidence, *ircot_result.evidence]
            ),
            stats=stats,
            farr_result=farr_result,
            ircot_result=ircot_result,
        )

    __call__ = answer
