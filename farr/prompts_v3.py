from __future__ import annotations

import json
from typing import Any, Dict, Sequence

from .documents import format_documents
from .prompts import answer_type_instruction
from .types import Document


def evidence_graph_prompt(
    question: str,
    rag_answer: str,
    flare_answer: str,
    evidence: Sequence[Document],
    max_chars: int,
) -> str:
    return f"""[FARR_V3:EVIDENCE_GRAPH]
Build the smallest complete evidence graph needed to answer this multi-hop
question. Resolve the question independently; the two candidates may both be
wrong.

Each hop must contain:
- one atomic relation,
- the resolved value,
- a 1-based evidence document id,
- a short quote copied VERBATIM from that document.

The quote must directly support the relation, not merely mention an entity.
Use at least two hops and two distinct evidence documents. For comparison
questions, use separate hops for the two compared attributes and then derive
the requested comparison. For bridge questions, resolve the intermediate
entity before its requested property.

Mandatory final-answer format:
{answer_type_instruction(question)}

Return JSON only:
{{
  "question_type":"bridge|comparison|other",
  "hops":[
    {{"relation":"...", "value":"...", "doc_id":1, "quote":"exact quote"}}
  ],
  "complete":true,
  "answer":"exact concise answer"
}}

Question:
{question}

RAG candidate:
{rag_answer}

FLARE candidate:
{flare_answer}

Numbered evidence:
{format_documents(evidence, max_chars)}
"""


def graph_selector_prompt(
    question: str,
    rag_answer: str,
    flare_answer: str,
    graph_answer: str,
    graph: Dict[str, Any],
    graph_valid: bool,
    validation_reason: str,
    evidence: Sequence[Document],
    max_chars: int,
) -> str:
    graph_text = json.dumps(graph, ensure_ascii=False)
    return f"""[FARR_V3:SELECT]
Select the best exact answer to a multi-hop question from three candidates.
Judge the complete relation, not entity mentions. Candidate C was constructed
from an evidence graph whose quotes were checked by code.

Rules:
- Candidate C can be SUPPORTED only when graph_valid is true.
- A candidate is SUPPORTED only if every necessary bridge and final relation
  hold.
- For comparison, explicitly compare the two extracted attribute values.
- For numeric questions, match the exact requested quantity.
- Prefer concise answer-type compliance, but never prefer wording over factual
  support.

Mandatory answer format:
{answer_type_instruction(question)}

Return JSON only:
{{
  "candidate_a_label":"SUPPORTED|UNSUPPORTED|UNCERTAIN",
  "candidate_b_label":"SUPPORTED|UNSUPPORTED|UNCERTAIN",
  "candidate_c_label":"SUPPORTED|UNSUPPORTED|UNCERTAIN",
  "selected":"A|B|C",
  "reason":"brief relation-level reason"
}}

Question:
{question}

Candidate A (RAG):
{rag_answer}

Candidate B (FLARE):
{flare_answer}

Candidate C (evidence graph):
{graph_answer}

graph_valid:
{str(graph_valid).lower()}

Graph validation:
{validation_reason}

Evidence graph:
{graph_text}

Evidence:
{format_documents(evidence, max_chars)}
"""
