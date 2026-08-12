from __future__ import annotations

from typing import Sequence

from .documents import format_documents
from .types import Document, HopTrace


def answer_type_instruction(question: str) -> str:
    normalized = " ".join(str(question).lower().split())
    yes_no_prefixes = (
        "is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ",
        "could ", "has ", "have ", "had ", "will ", "would ",
    )
    if normalized.startswith(yes_no_prefixes) or any(
        marker in f" {normalized} "
        for marker in (" determine if ", " determine whether ", " decide if ")
    ):
        return 'This is a yes/no question. Return exactly "yes" or "no".'
    if "how many" in normalized or "how much" in normalized:
        return "Return only the requested number or quantity."
    if normalized.startswith("who "):
        return "Return only the requested person or organization name."
    if normalized.startswith("where "):
        return "Return only the requested place name."
    if normalized.startswith("when "):
        return "Return only the requested date, year, or time period."
    return "Return only the shortest phrase that matches the requested answer type."


def plan_prompt(question: str, evidence: Sequence[Document], max_hops: int, max_chars: int) -> str:
    return f"""[FARR:PLAN]
Decompose the multi-hop question into an ordered chain of at most {max_hops}
answerable subquestions. Each hop must resolve one missing fact, and later hops
must use entities or facts discovered earlier. The LAST subquestion must combine
the preceding bridge facts and directly resolve the answer type requested by the
original question. For a comparison or yes/no question, include a final
comparison/decision hop. Do not assume the final answer.

Return JSON only:
{{"subquestions": ["...", "..."]}}

Question:
{question}

Initial evidence:
{format_documents(evidence, max_chars)}
"""


def query_prompt(
    question: str,
    subquestion: str,
    prior_hops: Sequence[HopTrace],
    max_queries: int,
) -> str:
    trace = "\n".join(
        f"- Hop {hop.hop}: {hop.subquestion} -> {hop.intermediate_answer}"
        for hop in prior_hops
    ) or "(none)"
    return f"""[FARR:QUERY]
Generate up to {max_queries} focused retrieval queries for the current reasoning
hop. This is forward-looking active retrieval: use facts already resolved,
replace ambiguous references with discovered entities, and target information
still missing. Do not answer the overall question.

Return JSON only:
{{"queries": ["...", "..."]}}

Overall question:
{question}

Current subquestion:
{subquestion}

Resolved reasoning trace:
{trace}
"""


def hop_prompt(
    question: str,
    subquestion: str,
    prior_hops: Sequence[HopTrace],
    evidence: Sequence[Document],
    max_chars: int,
) -> str:
    trace = "\n".join(
        f"- {hop.subquestion} -> {hop.intermediate_answer}"
        for hop in prior_hops
    ) or "(none)"
    return f"""[FARR:HOP]
Resolve only the current subquestion from the evidence. The intermediate answer
will become bridge context for the next retrieval hop. State missing information
when the evidence is incomplete; do not invent a fact.

Return JSON only:
{{"answer": "...", "missing_information": "..."}}

Overall question:
{question}

Current subquestion:
{subquestion}

Prior resolved facts:
{trace}

Evidence:
{format_documents(evidence, max_chars)}
"""


def answer_prompt(
    question: str,
    hops: Sequence[HopTrace],
    evidence: Sequence[Document],
    max_chars: int,
) -> str:
    trace = "\n".join(
        f"{hop.hop}. {hop.subquestion}\n   Intermediate answer: {hop.intermediate_answer}"
        for hop in hops
    )
    type_rule = answer_type_instruction(question)
    return f"""[FARR:ANSWER]
Answer the original question by composing the resolved multi-hop chain. Use the
evidence, preserve the answer type requested by the question, and return only
the concise final answer. Do not expose chain-of-thought or add an explanation.
If evidence conflicts, choose the answer supported across the complete chain.

Mandatory output rule:
{type_rule}

Original question:
{question}

Resolved fact chain:
{trace}

Evidence:
{format_documents(evidence, max_chars)}

Final answer:
"""


def verification_query_prompt(
    question: str,
    answer: str,
    hops: Sequence[HopTrace],
    max_queries: int,
) -> str:
    trace = "\n".join(
        f"- {hop.subquestion} -> {hop.intermediate_answer}" for hop in hops
    )
    return f"""[FARR:VERIFY_QUERY]
Generate up to {max_queries} independent search queries that can verify or
falsify the proposed answer and its bridge facts. Include the proposed answer
where useful. Seek counterevidence as well as supporting evidence.

Return JSON only:
{{"queries": ["...", "..."]}}

Question:
{question}

Proposed answer:
{answer}

Fact chain:
{trace}
"""


def verify_prompt(
    question: str,
    answer: str,
    hops: Sequence[HopTrace],
    evidence: Sequence[Document],
    max_chars: int,
) -> str:
    trace = "\n".join(
        f"- {hop.subquestion} -> {hop.intermediate_answer}" for hop in hops
    )
    return f"""[FARR:VERIFY]
Verify the question-conditioned claim "the answer to the question is the
proposed answer." Check every required bridge in the fact chain against the
evidence. A plausible entity mention alone is not sufficient.

Labels:
- SUPPORTED: the answer and all necessary bridges are supported.
- UNSUPPORTED: evidence contradicts the answer or a necessary bridge.
- UNCERTAIN: at least one necessary bridge lacks enough evidence.

Return JSON only:
{{"label": "SUPPORTED|UNSUPPORTED|UNCERTAIN",
  "rationale": "brief evidence-grounded reason",
  "correction": "better answer if directly supported, otherwise empty"}}

Question:
{question}

Proposed answer:
{answer}

Fact chain:
{trace}

Verification evidence:
{format_documents(evidence, max_chars)}
"""


def revise_prompt(
    question: str,
    answer: str,
    verification_label: str,
    rationale: str,
    correction: str,
    hops: Sequence[HopTrace],
    evidence: Sequence[Document],
    max_chars: int,
) -> str:
    trace = "\n".join(
        f"- {hop.subquestion} -> {hop.intermediate_answer}" for hop in hops
    )
    return f"""[FARR:REVISE]
Revise the proposed answer after retrieval-based verification. Resolve the
identified unsupported or uncertain bridge using the complete evidence. You
may keep, replace, narrow, or expand the answer as the evidence warrants.
There is no default preference for preserving the original answer.

Return JSON only:
{{"answer": "..."}}

Question:
{question}

Current answer:
{answer}

Verification label:
{verification_label}

Verification rationale:
{rationale}

Suggested correction:
{correction}

Fact chain:
{trace}

Complete evidence:
{format_documents(evidence, max_chars)}
"""
