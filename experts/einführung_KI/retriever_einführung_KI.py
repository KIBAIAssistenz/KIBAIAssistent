# experts/einführung_KI/retriever_einführung_KI.py
import sys
import pathlib

# Projektwurzel zum Pfad hinzufügen
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rag.retriever_fix import build_hybrid_retriever_for_module
import config


def make_einführung_ki_retriever():
    """
    Baut oder lädt den Hybrid-Retriever (FAISS + BM25)
    speziell für 'Einführung in die KI'.
    """
    retriever = build_hybrid_retriever_for_module(
        pdf_dir=config.PDF_DIR_INTRO,
        faiss_dir=config.FAISS_DIR_INTRO,
        k=5,
        weights=[0.5, 0.5],
    )
    return retriever
