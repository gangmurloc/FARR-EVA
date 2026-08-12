from __future__ import annotations

from typing import Callable, List, Sequence

from .config import FARRConfig
from .documents import (
    dedupe_documents,
    doc_text,
    rerank_documents,
)
from .parsing import (
    parse_hop_answer,
    parse_plan,
    parse_queries,
    parse_revised_answer,
    parse_verification,
)
from .prompts import (
    answer_prompt,
    hop_prompt,
    plan_prompt,
    query_prompt,
    revise_prompt,
    verification_query_prompt,
    verify_prompt,
)
from .types import Document, FARRResult, FARRStats, HopTrace, VerificationTrace


Retriever = Callable[[str, int], List[Document]]
LanguageModel = Callable[[str], str]


class FARR:
    """Full FLARE-inspired active retrieval + RARR revision pipeline.

    The model first constructs a dependency-aware multi-hop plan, performs
    forward-looking retrieval at every hop, composes a draft, and then runs
    answer-aware retrieval, verification, revision, and re-verification.
    """

    def __init__(
        self,
        retriever: Retriever,
        llm: LanguageModel,
        config: FARRConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.config = config or FARRConfig()

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(f"[FARR] {message}")

    def _llm(self, prompt: str, stats: FARRStats) -> str:
        stats.llm_calls += 1
        return str(self.llm(prompt))

    def _retrieve(self, query: str, top_k: int, stats: FARRStats) -> List[Document]:
        query = " ".join(str(query).split())
        if not query:
            return []
        stats.retrieval_calls += 1
        return list(self.retriever(query, top_k) or [])

    @staticmethod
    def _bridge_context(hops: Sequence[HopTrace]) -> str:
        return " ".join(
            f"{hop.subquestion} {hop.intermediate_answer}" for hop in hops
        )

    def _initial_retrieval(self, question: str, stats: FARRStats) -> List[Document]:
        documents = self._retrieve(question, self.config.initial_top_k, stats)
        return dedupe_documents(documents)

    def _plan(
        self,
        question: str,
        initial_evidence: Sequence[Document],
        stats: FARRStats,
    ) -> List[str]:
        if not self.config.enable_decomposition:
            stats.planned_hops = 1
            return [question]
        raw = self._llm(
            plan_prompt(
                question,
                initial_evidence,
                self.config.max_hops,
                self.config.max_chars_per_doc,
            ),
            stats,
        )
        plan = parse_plan(raw, question, self.config.max_hops)
        stats.planned_hops = len(plan)
        self._log(f"plan={plan}")
        return plan

    def _run_hop(
        self,
        question: str,
        subquestion: str,
        hop_number: int,
        prior_hops: Sequence[HopTrace],
        initial_evidence: Sequence[Document],
        stats: FARRStats,
    ) -> tuple[HopTrace, List[Document]]:
        bridge = self._bridge_context(prior_hops)
        fallback_queries = [f"{bridge} {subquestion}".strip(), subquestion]
        if self.config.enable_adaptive_queries:
            raw_queries = self._llm(
                query_prompt(
                    question,
                    subquestion,
                    prior_hops,
                    self.config.max_queries_per_hop,
                ),
                stats,
            )
            queries = parse_queries(
                raw_queries,
                fallback_queries,
                self.config.max_queries_per_hop,
            )
        else:
            queries = [fallback_queries[0]]

        candidates: List[Document] = list(initial_evidence)
        for query in queries:
            candidates.extend(
                self._retrieve(query, self.config.per_query_top_k, stats)
            )

        evidence = rerank_documents(
            subquestion,
            candidates,
            self.config.hop_evidence_top_k,
            bridge_context=bridge,
        )
        raw_answer = self._llm(
            hop_prompt(
                question,
                subquestion,
                prior_hops,
                evidence,
                self.config.max_chars_per_doc,
            ),
            stats,
        )
        parsed = parse_hop_answer(raw_answer)
        trace = HopTrace(
            hop=hop_number,
            subquestion=subquestion,
            queries=queries,
            intermediate_answer=parsed["answer"],
            missing_information=parsed["missing_information"],
            evidence_count=len(evidence),
        )
        self._log(f"hop={hop_number} answer={trace.intermediate_answer!r}")
        return trace, evidence

    def _verification_queries(
        self,
        question: str,
        answer: str,
        hops: Sequence[HopTrace],
        stats: FARRStats,
    ) -> List[str]:
        raw = self._llm(
            verification_query_prompt(
                question,
                answer,
                hops,
                self.config.max_queries_per_hop,
            ),
            stats,
        )
        fallbacks = [
            f"{question} {answer}",
            f"{self._bridge_context(hops)} {answer}",
        ]
        return parse_queries(raw, fallbacks, self.config.max_queries_per_hop)

    def _verify(
        self,
        question: str,
        answer: str,
        hops: Sequence[HopTrace],
        accumulated_evidence: Sequence[Document],
        round_number: int,
        stats: FARRStats,
    ) -> tuple[dict[str, str], List[Document], List[str]]:
        queries = self._verification_queries(question, answer, hops, stats)
        fresh_evidence: List[Document] = []
        for query in queries:
            fresh_evidence.extend(
                self._retrieve(query, self.config.verification_top_k, stats)
            )

        verification_evidence = rerank_documents(
            f"{question} {answer}",
            [*fresh_evidence, *accumulated_evidence],
            self.config.max_evidence_docs,
            bridge_context=self._bridge_context(hops),
        )
        raw = self._llm(
            verify_prompt(
                question,
                answer,
                hops,
                verification_evidence,
                self.config.max_chars_per_doc,
            ),
            stats,
        )
        verdict = parse_verification(raw)
        stats.verification_traces.append(
            VerificationTrace(
                round=round_number,
                answer=answer,
                label=verdict["label"],
                rationale=verdict["rationale"],
                queries=queries,
            )
        )
        stats.final_verification_label = verdict["label"]
        self._log(f"verification={verdict['label']}")
        return verdict, verification_evidence, queries

    def answer(self, question: str) -> FARRResult:
        question = " ".join(str(question).split())
        if not question:
            raise ValueError("question cannot be empty")

        stats = FARRStats()
        initial_evidence = self._initial_retrieval(question, stats)
        subquestions = self._plan(question, initial_evidence, stats)

        hops: List[HopTrace] = []
        all_evidence: List[Document] = list(initial_evidence)
        for hop_number, subquestion in enumerate(subquestions, 1):
            trace, evidence = self._run_hop(
                question,
                subquestion,
                hop_number,
                hops,
                initial_evidence,
                stats,
            )
            hops.append(trace)
            all_evidence.extend(evidence)
            stats.completed_hops += 1
        stats.hop_traces = hops

        all_evidence = rerank_documents(
            question,
            all_evidence,
            self.config.max_evidence_docs,
            bridge_context=self._bridge_context(hops),
        )
        draft = self._llm(
            answer_prompt(
                question,
                hops,
                all_evidence,
                self.config.max_chars_per_doc,
            ),
            stats,
        ).strip()
        answer = draft

        if not self.config.enable_verification:
            stats.final_verification_label = "NOT_RUN"
            return FARRResult(answer=answer, evidence=all_evidence, stats=stats)

        total_verifications = self.config.max_revision_rounds + 1
        for round_number in range(1, total_verifications + 1):
            verdict, verification_evidence, _ = self._verify(
                question,
                answer,
                hops,
                all_evidence,
                round_number,
                stats,
            )
            all_evidence = dedupe_documents([*all_evidence, *verification_evidence])

            if verdict["label"] not in self.config.revise_on_labels:
                break
            if stats.revision_count >= self.config.max_revision_rounds:
                break

            raw_revision = self._llm(
                revise_prompt(
                    question,
                    answer,
                    verdict["label"],
                    verdict["rationale"],
                    verdict["correction"],
                    hops,
                    all_evidence,
                    self.config.max_chars_per_doc,
                ),
                stats,
            )
            revised = parse_revised_answer(raw_revision)
            if not revised:
                break
            answer = revised
            stats.revision_count += 1

        final_evidence = rerank_documents(
            f"{question} {answer}",
            all_evidence,
            self.config.max_evidence_docs,
            bridge_context=self._bridge_context(hops),
        )
        return FARRResult(answer=answer, evidence=final_evidence, stats=stats)

    __call__ = answer
