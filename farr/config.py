from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class FARRConfig:
    """Runtime policy for the full FARR pipeline.

    Defaults target ordinary multi-hop QA rather than the conservative
    short-answer policy used by FARR-Lite.
    """

    initial_top_k: int = 8
    max_hops: int = 4
    max_queries_per_hop: int = 3
    per_query_top_k: int = 6
    hop_evidence_top_k: int = 6
    max_evidence_docs: int = 24
    verification_top_k: int = 6
    max_revision_rounds: int = 2
    max_chars_per_doc: int = 1400
    revise_on_labels: Tuple[str, ...] = ("UNSUPPORTED", "UNCERTAIN")
    enable_decomposition: bool = True
    enable_adaptive_queries: bool = True
    enable_verification: bool = True
    flare_confidence_threshold: float = 0.20
    flare_max_steps: int = 3
    fusion_evidence_top_k: int = 8
    candidate_selector_path: Optional[str] = None
    verbose: bool = False

    def __post_init__(self) -> None:
        positive = {
            "initial_top_k": self.initial_top_k,
            "max_hops": self.max_hops,
            "max_queries_per_hop": self.max_queries_per_hop,
            "per_query_top_k": self.per_query_top_k,
            "hop_evidence_top_k": self.hop_evidence_top_k,
            "max_evidence_docs": self.max_evidence_docs,
            "verification_top_k": self.verification_top_k,
            "max_chars_per_doc": self.max_chars_per_doc,
            "flare_max_steps": self.flare_max_steps,
            "fusion_evidence_top_k": self.fusion_evidence_top_k,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.max_revision_rounds < 0:
            raise ValueError("max_revision_rounds cannot be negative")
        if not 0.0 <= self.flare_confidence_threshold <= 1.0:
            raise ValueError("flare_confidence_threshold must be between 0 and 1")

        valid = {"SUPPORTED", "UNSUPPORTED", "UNCERTAIN"}
        labels = tuple(label.upper() for label in self.revise_on_labels)
        if not labels or any(label not in valid for label in labels):
            raise ValueError(f"revise_on_labels must contain labels from {sorted(valid)}")
        object.__setattr__(self, "revise_on_labels", labels)
