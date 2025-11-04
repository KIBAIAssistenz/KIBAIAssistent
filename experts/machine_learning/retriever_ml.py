# experts/einführung_KI/retriever_einführung_KI.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from rag.retriever import load_or_build_hybrid_retriever


def make_machine_learning_retriever():
    """
    Baut oder lädt den Hybrid-Retriever (FAISS + BM25)
    speziell für 'Machine Learning'.
    """
    return load_or_build_hybrid_retriever(subdir="machine_learning", k_faiss=5, k_bm25=3, weights=(0.5, 0.5))
