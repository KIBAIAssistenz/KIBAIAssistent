# experts/machine_learning/retriever_ml.py
import sys
import pathlib

# Projektwurzel auf den Pfad setzen
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rag.retriever_fix import build_hybrid_retriever_for_module
import config


def make_machine_learning_retriever():
    """
    Baut oder lädt den Hybrid-Retriever (FAISS + BM25)
    speziell für 'Machine Learning'.
    """
    retriever = build_hybrid_retriever_for_module(
        pdf_dir=config.PDF_DIR_ML,
        faiss_dir=config.FAISS_DIR_ML,
        k=5,
        weights=[0.5, 0.5],
    )
    return retriever
