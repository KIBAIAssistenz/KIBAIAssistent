from __future__ import annotations

from typing import List, Sequence, Tuple
from pathlib import Path

from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from .processor.pipeline_fix import process_pdfs_to_chunks
from .buildFaiss_fix import build_faiss_store, load_faiss_store, get_faiss_retriever
from .bm25_store_fix import build_bm25_retriever


def build_or_load_faiss_for_module(
    pdf_dir: str,
    faiss_dir: str,
) -> tuple[List[Document], FAISS]:
    """
    Lädt/erzeugt Chunks aus PDFs und baut/ladet dazu den FAISS-Index.
    Wenn Laden fehlschlägt, wird der Index neu gebaut.
    """
    # 1) Chunks erzeugen
    chunks = process_pdfs_to_chunks(pdf_dir)

    faiss_path = Path(faiss_dir)
    index_file = faiss_path / "index.faiss"

    # Sicherstellen, dass Verzeichnis existiert
    faiss_path.mkdir(parents=True, exist_ok=True)

    # 2) Versuchen zu laden, wenn index.faiss existiert
    if index_file.exists():
        print(f"🔁 Bestehenden FAISS-Index laden aus: {faiss_path}")
        try:
            vectorstore = load_faiss_store(faiss_dir)
            return chunks, vectorstore
        except Exception as e:
            print(f"⚠️ Konnte bestehenden FAISS-Index NICHT laden ({e}). Baue neu...")

    # 3) Wenn es keinen Index gibt oder Laden schiefging → neu bauen
    print(f"🆕 Neuer FAISS-Index wird gebaut in: {faiss_path}")
    vectorstore = build_faiss_store(chunks, faiss_dir)
    return chunks, vectorstore


def build_hybrid_retriever_for_module(
    pdf_dir: str,
    faiss_dir: str,
    k: int = 5,
    weights: Sequence[float] | None = None,
) -> EnsembleRetriever:
    """
    Baut einen hybriden Retriever (FAISS + BM25) für ein Modul.
    """
    # Chunks + FAISS
    chunks, faiss_vs = build_or_load_faiss_for_module(pdf_dir, faiss_dir)

    # FAISS-Retriever
    faiss_retriever = get_faiss_retriever(faiss_vs, k=k)

    # BM25-Retriever (arbeitet auf denselben Chunks)
    bm25_retriever = build_bm25_retriever(chunks, k=k)

    if weights is None:
        weights = [0.5, 0.5]

    hybrid = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=list(weights),
    )

    print("✅ Hybrid-Retriever (FAISS + BM25) gebaut.")
    return hybrid
