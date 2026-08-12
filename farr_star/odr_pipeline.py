from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import compat  # noqa: F401
from .contracts import heuristic_contract
from .oracle_router import (
    OracleDistilledRouter,
    feature_from_runtime,
)
from .pipeline import STARStats
from farr.baselines import ircot
from farr.config import FARRConfig
from farr.documents import dedupe_documents
from farr.pipeline_v2 import FARRV2, clean_short_answer


@dataclass
class ODRResult:
    answer: str
    evidence: list[Any]
    stats: STARStats
    farr_result: Any
    ircot_result: Any


class FARRODR:
    """Oracle-distilled selection over FLARE, IRCoT, and FARR."""

    def __init__(
        self,
        retriever: Any,
        llm: Any,
        router: OracleDistilledRouter,
        config: FARRConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.router = router
        self.config = config or FARRConfig()

    def answer(self, question: str) -> ODRResult:
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
        flare_answer = clean_short_answer(
            question,
            farr_result.stats.candidate_answers.get(
                "flare",
                farr_result.answer,
            ),
            fallback=farr_result.answer,
        )
        feature = feature_from_runtime(
            question,
            flare_answer,
            farr_result,
            ircot_result,
        )
        selected, confidence, probabilities = self.router.choose(feature)
        answers = {
            "flare": flare_answer,
            "ircot": clean_short_answer(
                question,
                ircot_result.answer,
            ),
            "farr-v2": clean_short_answer(
                question,
                farr_result.answer,
            ),
        }
        contract = heuristic_contract(question)
        public_selected = "farr" if selected == "farr-v2" else selected
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
            final_verification_label="ORACLE_DISTILLED",
            reasoning_type=contract.reasoning_type,
            answer_type=contract.answer_type,
            answer_target=contract.target,
            requires_comparison=contract.requires_comparison,
            return_compared_entity=contract.return_compared_entity,
            allowed_answers=list(contract.allowed_answers),
            constraints=list(contract.constraints),
            selected_expert=public_selected,
            route="oracle_distilled_router",
            router_confidence=confidence,
            router_probabilities=" ||| ".join(
                (
                    f"{'farr' if name == 'farr-v2' else name}="
                    f"{probabilities.get(name, 0.0):.6f}"
                )
                for name in ("flare", "ircot", "farr-v2")
            ),
            selection_reason=(
                "Dataset-balanced oracle-distilled linear router: "
                + ", ".join(
                    (
                        f"{'farr' if name == 'farr-v2' else name}="
                        f"{probabilities.get(name, 0.0):.4f}"
                    )
                    for name in ("flare", "ircot", "farr-v2")
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
        return ODRResult(
            answer=answers[selected],
            evidence=dedupe_documents(
                [*farr_result.evidence, *ircot_result.evidence]
            ),
            stats=stats,
            farr_result=farr_result,
            ircot_result=ircot_result,
        )

    __call__ = answer
