from __future__ import annotations

import re
from typing import Any, List, Sequence

from .config import FARRConfig
from .documents import dedupe_documents, format_documents, rerank_documents
from .parsing import (
    parse_hop_answer,
    parse_queries,
    parse_revised_answer,
    parse_verification,
)
from .prompts import (
    answer_prompt,
    answer_type_instruction,
    revise_prompt,
    verification_query_prompt,
    verify_prompt,
)
from .types import Document, FARRResult, FARRStats, HopTrace, VerificationTrace


def _call(llm: Any, prompt: str, stats: FARRStats) -> str:
    stats.llm_calls += 1
    return str(llm(prompt)).strip()


def _retrieve(retriever: Any, query: str, top_k: int, stats: FARRStats) -> List[Document]:
    stats.retrieval_calls += 1
    return list(retriever(query, top_k) or [])


def _direct_answer_prompt(
    tag: str,
    question: str,
    documents: Sequence[Document] | None,
    max_chars: int,
) -> str:
    evidence = (
        format_documents(documents or [], max_chars)
        if documents is not None
        else "(No external evidence is available.)"
    )
    return f"""[{tag}]
Answer the question concisely.
{answer_type_instruction(question)}
Return only the final answer without explanation.

Question:
{question}

Evidence:
{evidence}

Final answer:
"""


def closed_book(llm: Any, question: str) -> FARRResult:
    stats = FARRStats()
    answer = _call(
        llm,
        _direct_answer_prompt("CLOSED_BOOK:ANSWER", question, None, 1),
        stats,
    )
    return FARRResult(answer=answer, evidence=[], stats=stats)


def rag(
    llm: Any,
    retriever: Any,
    question: str,
    top_k: int = 6,
    max_chars: int = 1000,
) -> FARRResult:
    stats = FARRStats()
    documents = _retrieve(retriever, question, top_k, stats)
    answer = _call(
        llm,
        _direct_answer_prompt("RAG:ANSWER", question, documents, max_chars),
        stats,
    )
    return FARRResult(answer=answer, evidence=documents, stats=stats)


def full_context(
    llm: Any,
    question: str,
    documents: Sequence[Document],
    max_chars: int = 1000,
) -> FARRResult:
    stats = FARRStats()
    answer = _call(
        llm,
        _direct_answer_prompt(
            "FULL_CONTEXT:ANSWER",
            question,
            list(documents),
            max_chars,
        ),
        stats,
    )
    return FARRResult(answer=answer, evidence=list(documents), stats=stats)


def _flare_query(
    question: str,
    generated: dict[str, Any],
    threshold: float,
) -> str:
    pieces = []
    for token, probability in zip(
        generated.get("tokens", []),
        generated.get("probabilities", []),
    ):
        clean = token.strip()
        if not clean:
            continue
        pieces.append(clean if probability >= threshold else "[MASK]")
    prediction = " ".join(pieces)
    prediction = re.sub(r"(?:\[MASK\]\s*){2,}", "[MASK] ", prediction).strip()
    return f"{question} {prediction}".strip()


def flare(
    llm: Any,
    retriever: Any,
    question: str,
    config: FARRConfig,
    confidence_threshold: float = 0.20,
    max_steps: int = 3,
) -> FARRResult:
    """Token-confidence FLARE adaptation for short multi-hop QA."""
    stats = FARRStats()
    evidence = _retrieve(retriever, question, config.initial_top_k, stats)
    hops: List[HopTrace] = []

    for step in range(1, max_steps + 1):
        trace = "\n".join(
            f"- {hop.intermediate_answer}" for hop in hops
        ) or "(none)"
        prospective_prompt = f"""[FLARE:PREDICT]
Predict exactly one short factual sentence that should come next when solving
the multi-hop question. Use resolved facts, but do not give a bare final answer.

Question:
{question}

Resolved facts:
{trace}

Next factual sentence:
"""
        stats.llm_calls += 1
        if hasattr(llm, "generate_with_confidence"):
            generated = llm.generate_with_confidence(prospective_prompt, 64)
        else:
            text = str(llm(prospective_prompt)).strip()
            generated = {
                "text": text,
                "tokens": text.split(),
                "probabilities": [0.0] * len(text.split()),
            }

        query = _flare_query(question, generated, confidence_threshold)
        fresh = _retrieve(retriever, query, config.per_query_top_k, stats)
        evidence = rerank_documents(
            question,
            [*evidence, *fresh],
            config.max_evidence_docs,
            bridge_context=" ".join(hop.intermediate_answer for hop in hops),
        )
        grounded_prompt = f"""[FLARE:GROUND]
Ground the predicted next fact in the retrieved evidence. Return JSON only:
{{"answer":"one short supported factual sentence","missing_information":""}}

Question:
{question}

Predicted sentence:
{generated.get("text", "")}

Evidence:
{format_documents(evidence, config.max_chars_per_doc)}
"""
        parsed = parse_hop_answer(_call(llm, grounded_prompt, stats))
        hops.append(
            HopTrace(
                hop=step,
                subquestion="Forward-looking next fact",
                queries=[query],
                intermediate_answer=parsed["answer"],
                missing_information=parsed["missing_information"],
                evidence_count=len(evidence),
            )
        )

    stats.hop_traces = hops
    stats.planned_hops = max_steps
    stats.completed_hops = len(hops)
    answer = _call(
        llm,
        answer_prompt(
            question,
            hops,
            evidence,
            config.max_chars_per_doc,
        ),
        stats,
    )
    return FARRResult(answer=answer, evidence=evidence, stats=stats)


def rarr(
    llm: Any,
    retriever: Any,
    question: str,
    config: FARRConfig,
) -> FARRResult:
    """Retrieve, answer, then research/revise without conservative retention."""
    base = rag(
        llm,
        retriever,
        question,
        config.initial_top_k,
        config.max_chars_per_doc,
    )
    stats = base.stats
    answer = base.answer
    evidence = list(base.evidence)

    for round_number in range(1, config.max_revision_rounds + 2):
        raw_queries = _call(
            llm,
            verification_query_prompt(
                question,
                answer,
                [],
                config.max_queries_per_hop,
            ),
            stats,
        )
        queries = parse_queries(
            raw_queries,
            [f"{question} {answer}"],
            config.max_queries_per_hop,
        )
        fresh = []
        for query in queries:
            fresh.extend(
                _retrieve(retriever, query, config.verification_top_k, stats)
            )
        evidence = rerank_documents(
            f"{question} {answer}",
            [*fresh, *evidence],
            config.max_evidence_docs,
        )
        verdict = parse_verification(
            _call(
                llm,
                verify_prompt(
                    question,
                    answer,
                    [],
                    evidence,
                    config.max_chars_per_doc,
                ),
                stats,
            )
        )
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
        if verdict["label"] not in config.revise_on_labels:
            break
        if stats.revision_count >= config.max_revision_rounds:
            break

        revised = parse_revised_answer(
            _call(
                llm,
                revise_prompt(
                    question,
                    answer,
                    verdict["label"],
                    verdict["rationale"],
                    verdict["correction"],
                    [],
                    evidence,
                    config.max_chars_per_doc,
                ),
                stats,
            )
        )
        if not revised:
            break
        answer = revised
        stats.revision_count += 1

    return FARRResult(answer=answer, evidence=evidence, stats=stats)


def ircot(
    llm: Any,
    retriever: Any,
    question: str,
    config: FARRConfig,
    max_steps: int = 4,
) -> FARRResult:
    """IRCoT-style interleaving of one evidence-derived step and retrieval."""
    stats = FARRStats(planned_hops=max_steps)
    evidence = _retrieve(retriever, question, config.initial_top_k, stats)
    hops: List[HopTrace] = []

    for step in range(1, max_steps + 1):
        trace = "\n".join(
            f"{hop.hop}. {hop.intermediate_answer}" for hop in hops
        ) or "(none)"
        prompt = f"""[IRCOT:STEP]
Produce the next concise evidence-derived fact for this multi-hop question and
a retrieval query needed after that fact. If the answer is now resolved, also
fill final_answer. Do not include hidden reasoning.

Return JSON only:
{{"fact":"...", "query":"...", "final_answer":""}}

Question:
{question}

Facts so far:
{trace}

Evidence:
{format_documents(evidence, config.max_chars_per_doc)}
"""
        raw = _call(llm, prompt, stats)
        from .parsing import _json_value

        parsed = _json_value(raw)
        if not isinstance(parsed, dict):
            parsed = {"fact": raw, "query": f"{question} {raw}", "final_answer": ""}
        fact = str(parsed.get("fact") or parsed.get("thought") or "").strip()
        query = str(parsed.get("query") or f"{question} {fact}").strip()
        final_answer = str(parsed.get("final_answer") or "").strip()
        hops.append(
            HopTrace(
                hop=step,
                subquestion="Interleaved retrieval step",
                queries=[query],
                intermediate_answer=fact,
                evidence_count=len(evidence),
            )
        )
        stats.completed_hops += 1
        if final_answer:
            hops[-1].intermediate_answer = (
                f"{fact} Proposed final answer: {final_answer}".strip()
            )
            break
        fresh = _retrieve(retriever, query, config.per_query_top_k, stats)
        evidence = rerank_documents(
            question,
            [*evidence, *fresh],
            config.max_evidence_docs,
            bridge_context=" ".join(hop.intermediate_answer for hop in hops),
        )

    stats.hop_traces = hops
    answer = _call(
        llm,
        answer_prompt(question, hops, evidence, config.max_chars_per_doc),
        stats,
    )
    return FARRResult(answer=answer, evidence=evidence, stats=stats)
