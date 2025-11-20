# experts/bis/retriever_bis.py
import sys
import pathlib

# Projektwurzel zum Pfad hinzufügen
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rag.retriever_fix import build_hybrid_retriever_for_module
import config


def make_bis_retriever():
    """
    Baut oder lädt den Hybrid-Retriever (FAISS + BM25)
    speziell für 'Betriebliche Informationssysteme (BIS)'.
    """

    retriever = build_hybrid_retriever_for_module(
        pdf_dir=config.PDF_DIR_BIS,
        faiss_dir=config.FAISS_DIR_BIS,
        k=5,
        weights=[0.5, 0.5],  # Gewichtung BM25 vs. Embeddings
    )

    return retriever
