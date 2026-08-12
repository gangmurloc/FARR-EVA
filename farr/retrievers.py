from __future__ import annotations

from typing import List

from .documents import doc_text, doc_title
from .types import Document


class TfidfRetriever:
    """Small deterministic retriever suitable for HotpotQA distractor context."""

    def __init__(self, documents: List[Document]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not documents:
            raise ValueError("documents cannot be empty")
        self.documents = documents
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        texts = [f"{doc_title(doc)} {doc_text(doc)}" for doc in documents]
        self.matrix = self.vectorizer.fit_transform(texts)

    def __call__(self, query: str, top_k: int) -> List[Document]:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        indices = scores.argsort()[::-1][: min(top_k, len(self.documents))]
        return [self.documents[index] for index in indices]
