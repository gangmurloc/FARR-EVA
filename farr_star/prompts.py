from __future__ import annotations

import json
from typing import Any, Sequence

from . import compat  # noqa: F401
from .contracts import QuestionContract
from farr.documents import format_documents
from farr.types import Document


def contract_prompt(question: str) -> str:
    return f"""[FARR_STAR:CONTRACT]
Convert the question into a dataset-independent reasoning contract. Do not
answer the question and do not use benchmark-specific labels.

Reasoning types:
- bridge: resolve an intermediate entity, then its requested property
- comparison: compare two directly available attributes
- comparison_chain: resolve attributes through one or more entities, compare
  them, and return the originally requested entity rather than an intermediate
  person, date, or attribute
- compositional: follow a sequential A -> B -> C relation chain
- inference: combine facts to infer a relation not copied directly
- temporal, numeric, boolean, or other

For explicit forced-choice questions, copy every possible final answer from
the question into allowed_answers. Example: "Which film ..., A or B?" must
return A or B, never a director or a year.

Create up to four self-contained retrieval queries. They must seek atomic
facts and must not assume an unknown intermediate answer.

Return JSON only:
{{
  "reasoning_type":"bridge|comparison|comparison_chain|compositional|inference|temporal|numeric|boolean|other",
  "answer_type":"person|organization|location|work|date|number|boolean|cause|entity|text",
  "target":"what the final answer must denote",
  "expected_hops":2,
  "requires_comparison":false,
  "return_compared_entity":false,
  "allowed_answers":[],
  "constraints":["relation or comparison that must hold"],
  "retrieval_queries":["self-contained atomic query"]
}}

Question:
{question}
"""


def _trace(result: Any) -> str:
    hops = getattr(getattr(result, "stats", None), "hop_traces", []) or []
    if not hops:
        return "(none)"
    return "\n".join(
        f"{hop.hop}. {hop.intermediate_answer}"
        for hop in hops
        if str(hop.intermediate_answer).strip()
    ) or "(none)"


def adjudication_prompt(
    question: str,
    contract: QuestionContract,
    farr_result: Any,
    ircot_result: Any,
    deterministic_a: tuple[bool, str],
    deterministic_b: tuple[bool, str],
    evidence: Sequence[Document],
    max_chars: int,
) -> str:
    return f"""[FARR_STAR:ADJUDICATE]
Judge two answers to a multi-hop question against the complete reasoning
contract and evidence. Evidence mention alone is insufficient: every relation,
comparison, temporal operation, and requested return target must be satisfied.

Critical rules:
- The final answer must denote contract.target.
- In a forced-choice question, the answer must be one of allowed_answers.
- Never return an intermediate entity, person, date, number, or attribute when
  the question requests the original compared object.
- A comparison_chain must resolve both branches, compare their attributes in
  the requested direction, then map the winner back to the requested object.
- Prefer a supported candidate. Use SYNTHESIZE only when neither candidate is
  fully correct but the evidence determines a different concise answer.

Return JSON only:
{{
  "candidate_a_label":"SUPPORTED|UNSUPPORTED|UNCERTAIN",
  "candidate_a_target_match":true,
  "candidate_b_label":"SUPPORTED|UNSUPPORTED|UNCERTAIN",
  "candidate_b_target_match":true,
  "selected":"A|B|SYNTHESIZE|UNCERTAIN",
  "answer":"candidate or corrected concise answer",
  "missing_queries":["query for a genuinely missing relation"],
  "reason":"brief relation-level explanation"
}}

Question:
{question}

Reasoning contract:
{json.dumps(contract.to_dict(), ensure_ascii=False)}

Candidate A (FARR):
{farr_result.answer}
Deterministic target check: {deterministic_a}
FARR trace:
{_trace(farr_result)}

Candidate B (IRCoT):
{ircot_result.answer}
Deterministic target check: {deterministic_b}
IRCoT trace:
{_trace(ircot_result)}

Evidence:
{format_documents(evidence, max_chars)}
"""


def repair_prompt(
    question: str,
    contract: QuestionContract,
    farr_answer: str,
    ircot_answer: str,
    evidence: Sequence[Document],
    max_chars: int,
) -> str:
    return f"""[FARR_STAR:REPAIR]
Resolve the question independently using the reasoning contract and numbered
evidence. Correct candidate errors, especially answers that stop at an
intermediate entity or attribute.

Each step must contain one concise supported fact and a short quote copied
verbatim from its cited document. For comparisons, include both attributes,
the comparison operation, and the mapping from the winning attribute back to
the requested entity. The answer must obey allowed_answers when non-empty.

Return JSON only:
{{
  "answer":"exact concise final answer",
  "target_match":true,
  "complete":true,
  "steps":[
    {{"claim":"atomic relation", "doc_id":1, "quote":"verbatim evidence quote"}}
  ]
}}

Question:
{question}

Reasoning contract:
{json.dumps(contract.to_dict(), ensure_ascii=False)}

FARR candidate:
{farr_answer}

IRCoT candidate:
{ircot_answer}

Numbered evidence:
{format_documents(evidence, max_chars)}
"""


def verification_prompt(
    question: str,
    contract: QuestionContract,
    solution: dict[str, Any],
    evidence: Sequence[Document],
    max_chars: int,
) -> str:
    return f"""[FARR_STAR:VERIFY]
Act as an independent verifier. Check the proposed answer against the question
contract and evidence. Reject it if a hop is missing, the comparison direction
is wrong, or the answer denotes an intermediate value instead of the requested
target. Do not replace a supported answer merely for stylistic reasons.

Return JSON only:
{{
  "label":"SUPPORTED|UNSUPPORTED|UNCERTAIN",
  "target_match":true,
  "reason":"brief failed or satisfied constraint",
  "corrected_answer":"only when directly established by the evidence"
}}

Question:
{question}

Contract:
{json.dumps(contract.to_dict(), ensure_ascii=False)}

Proposed solution:
{json.dumps(solution, ensure_ascii=False)}

Evidence:
{format_documents(evidence, max_chars)}
"""
