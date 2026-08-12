from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Union


Document = Union[str, Mapping[str, Any], Any]


@dataclass
class HopTrace:
    hop: int
    subquestion: str
    queries: List[str] = field(default_factory=list)
    intermediate_answer: str = ""
    missing_information: str = ""
    evidence_count: int = 0


@dataclass
class VerificationTrace:
    round: int
    answer: str
    label: str
    rationale: str = ""
    queries: List[str] = field(default_factory=list)


@dataclass
class FARRStats:
    llm_calls: int = 0
    retrieval_calls: int = 0
    planned_hops: int = 0
    completed_hops: int = 0
    revision_count: int = 0
    final_verification_label: str = "UNCERTAIN"
    hop_traces: List[HopTrace] = field(default_factory=list)
    verification_traces: List[VerificationTrace] = field(default_factory=list)
    candidate_answers: Dict[str, str] = field(default_factory=dict)
    selected_candidate: str = ""
    draft_answer: str = ""
    evidence_graph_valid: bool = False
    evidence_graph_hops: int = 0
    selection_reason: str = ""
    selector_used: bool = False
    selector_confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "llm_calls": self.llm_calls,
            "retrieval_calls": self.retrieval_calls,
            "planned_hops": self.planned_hops,
            "completed_hops": self.completed_hops,
            "revision_count": self.revision_count,
            "revision_applied": int(self.revision_count > 0),
            "final_verification_label": self.final_verification_label,
            "rag_candidate": self.candidate_answers.get("rag", ""),
            "flare_candidate": self.candidate_answers.get("flare", ""),
            "selected_candidate": self.selected_candidate,
            "draft_answer": self.draft_answer,
            "evidence_graph_valid": int(self.evidence_graph_valid),
            "evidence_graph_hops": self.evidence_graph_hops,
            "selection_reason": self.selection_reason,
            "selector_used": int(self.selector_used),
            "selector_confidence": self.selector_confidence,
            "hop_queries": " ||| ".join(
                query for hop in self.hop_traces for query in hop.queries
            ),
            "intermediate_answers": " ||| ".join(
                hop.intermediate_answer for hop in self.hop_traces
            ),
        }


@dataclass
class FARRResult:
    answer: str
    evidence: List[Document]
    stats: FARRStats

    @property
    def docs(self) -> List[Document]:
        """Compatibility alias used by the older experiment scripts."""
        return self.evidence
