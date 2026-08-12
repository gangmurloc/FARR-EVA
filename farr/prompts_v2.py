from __future__ import annotations

from typing import Sequence

from .documents import format_documents
from .prompts import answer_type_instruction
from .types import Document


def disagreement_query_prompt(
    question: str,
    rag_answer: str,
    flare_answer: str,
    max_queries: int,
) -> str:
    return f"""[FARR_V2:AUDIT_QUERY]
Two answer candidates disagree. Generate at most {max_queries} independent
search queries that directly test the disputed bridge, attribute, comparison,
or quantity. Search for the relation asked by the question, not merely mentions
of either candidate.

Return JSON only:
{{"queries":["...", "..."]}}

Question:
{question}

Candidate A (single-pass RAG):
{rag_answer}

Candidate B (active FLARE):
{flare_answer}
"""


def candidate_audit_prompt(
    question: str,
    rag_answer: str,
    flare_answer: str,
    evidence: Sequence[Document],
    max_chars: int,
) -> str:
    type_rule = answer_type_instruction(question)
    return f"""[FARR_V2:AUDIT]
Act as an evidence auditor for two competing answers to a multi-hop question.
Evaluate the complete question-conditioned relation. An entity appearing in a
passage is not enough: every necessary bridge and the final relation must hold.

For comparison questions:
1. extract the requested attribute for each entity,
2. compare those two values explicitly,
3. apply the mandatory answer format.

For bridge questions:
1. resolve the intermediate entity,
2. verify that the final attribute belongs to that entity.

For numeric questions, distinguish the exact requested quantity (for example,
seated capacity versus total capacity). Prefer the exact wording in evidence.

Candidate labels:
- SUPPORTED: all required relations are supported.
- UNSUPPORTED: a required relation is contradicted.
- UNCERTAIN: a required relation is missing.

Selection:
- Select A or B only when that candidate is better supported.
- Select REVISE only when neither candidate gives the exact supported answer.
- Do not preserve a candidate merely because it appeared first.

Mandatory final-answer rule:
{type_rule}

Return JSON only:
{{
  "candidate_a_label":"SUPPORTED|UNSUPPORTED|UNCERTAIN",
  "candidate_b_label":"SUPPORTED|UNSUPPORTED|UNCERTAIN",
  "selected":"A|B|REVISE",
  "answer":"exact concise final answer",
  "reason":"brief evidence-grounded reason"
}}

Question:
{question}

Candidate A (single-pass RAG):
{rag_answer}

Candidate B (active FLARE):
{flare_answer}

Evidence:
{format_documents(evidence, max_chars)}
"""


def corrected_answer_verify_prompt(
    question: str,
    answer: str,
    evidence: Sequence[Document],
    max_chars: int,
) -> str:
    return f"""[FARR_V2:VERIFY_CORRECTION]
Verify the corrected answer against the complete relation asked by the question.
Return JSON only:
{{"label":"SUPPORTED|UNSUPPORTED|UNCERTAIN","reason":"brief reason"}}

Question:
{question}

Corrected answer:
{answer}

Evidence:
{format_documents(evidence, max_chars)}
"""
