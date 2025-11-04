from __future__ import annotations

from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document


def build_bm25_retriever(
    docs: List[Document],
    k: int = 5,
) -> BM25Retriever:
    """
    Baut einen BM25-Retriever auf Basis der gegebenen Chunks.
    """
    if not docs:
        raise ValueError("Leere Dokumentliste für BM25-Retriever erhalten.")

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25
