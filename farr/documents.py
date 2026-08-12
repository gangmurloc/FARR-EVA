from __future__ import annotations

import re
from typing import Any, Iterable, List, Sequence, Tuple

from .types import Document


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "how", "in", "is", "it", "of",
    "on", "or", "that", "the", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "with",
}


def doc_text(doc: Document) -> str:
    if isinstance(doc, str):
        return doc
    if isinstance(doc, dict):
        return str(
            doc.get("page_content")
            or doc.get("text")
            or doc.get("passage")
            or doc.get("content")
            or ""
        )
    return str(getattr(doc, "page_content", doc))


def doc_title(doc: Document) -> str:
    if isinstance(doc, dict):
        metadata = doc.get("metadata", {}) or {}
        return str(metadata.get("title") or doc.get("title") or "")
    metadata = getattr(doc, "metadata", {}) or {}
    return str(metadata.get("title", "")) if hasattr(metadata, "get") else ""


def normalize(text: str) -> str:
    return " ".join(TOKEN_RE.findall(str(text).lower()))


def content_tokens(text: str) -> set[str]:
    return {
        token for token in TOKEN_RE.findall(str(text).lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def doc_key(doc: Document) -> Tuple[str, str]:
    return normalize(doc_title(doc)), normalize(doc_text(doc))[:800]


def dedupe_documents(documents: Iterable[Document]) -> List[Document]:
    seen = set()
    result = []
    for doc in documents:
        key = doc_key(doc)
        if key in seen:
            continue
        seen.add(key)
        result.append(doc)
    return result


def rerank_documents(
    query: str,
    documents: Sequence[Document],
    top_k: int,
    bridge_context: str = "",
) -> List[Document]:
    """Dependency-aware lexical reranker with a small diversity reward."""

    query_terms = content_tokens(query)
    bridge_terms = content_tokens(bridge_context)
    candidates = []

    for position, doc in enumerate(dedupe_documents(documents)):
        text_terms = content_tokens(f"{doc_title(doc)} {doc_text(doc)}")
        query_overlap = len(query_terms & text_terms) / max(len(query_terms), 1)
        bridge_overlap = len(bridge_terms & text_terms) / max(len(bridge_terms), 1)
        title_overlap = len(query_terms & content_tokens(doc_title(doc)))
        retrieval_prior = 1.0 / (position + 2)
        score = (
            0.62 * query_overlap
            + 0.20 * bridge_overlap
            + 0.12 * min(title_overlap, 2)
            + 0.06 * retrieval_prior
        )
        candidates.append((score, position, doc, text_terms))

    selected: List[Document] = []
    selected_terms: set[str] = set()
    while candidates and len(selected) < top_k:
        best_index = max(
            range(len(candidates)),
            key=lambda i: (
                candidates[i][0]
                + 0.08 * len(candidates[i][3] - selected_terms)
                / max(len(candidates[i][3]), 1),
                -candidates[i][1],
            ),
        )
        _, _, doc, terms = candidates.pop(best_index)
        selected.append(doc)
        selected_terms |= terms

    return selected


def format_documents(documents: Sequence[Document], max_chars: int = 1400) -> str:
    blocks = []
    for index, doc in enumerate(documents, 1):
        title = doc_title(doc)
        text = re.sub(r"\s+", " ", doc_text(doc)).strip()
        if len(text) > max_chars:
            text = f"{text[:max_chars]}..."
        heading = f"[{index}] {title}" if title else f"[{index}]"
        blocks.append(f"{heading}\n{text}")
    return "\n\n".join(blocks)
