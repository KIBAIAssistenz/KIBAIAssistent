# from __future__ import annotations
# import sys, pathlib
# sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# from pathlib import Path
# from typing import Literal, Sequence

# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.retrievers import BM25Retriever
# from langchain.retrievers import EnsembleRetriever
# from langchain_huggingface import HuggingFaceEmbeddings


# from rag.buildFaiss import build_index_for

# EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"

# def _load_faiss(store: Path) -> FAISS:
#     emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
#     vs = FAISS.load_local(str(store), emb, allow_dangerous_deserialization=True)
#     return vs

# def _ensure_index(subdir: Literal["einführung_KI","machine_learning"]) -> Path:
#     store = Path(f"rag/stores/{subdir}")
#     return store if store.exists() else build_index_for(subdir)

# def load_or_build_hybrid_retriever(
#     subdir: Literal["einführung_KI","machine_learning"] = "einführung_KI",
#     k_faiss: int = 5,
#     k_bm25: int = 3,
#     weights: Sequence[float] = (0.5, 0.5),
#     search_type: str = "similarity",   # "similarity" | "mmr" etc.
# ) -> EnsembleRetriever:
#     """
#     Lädt (oder baut) einen Hybrid-Retriever (BM25 + FAISS) aus dem Store 'rag/stores/<subdir>'.
#     """
#     store = _ensure_index(subdir)
#     vs = _load_faiss(store)

#     # FAISS-Retriever
#     faiss_ret = vs.as_retriever(search_type=search_type, search_kwargs={"k": k_faiss})

#     # BM25 über die Dokumente in der FAISS-Docstore
#     docs = list(vs.docstore._dict.values())  # bewusst intern, aber stabil in LangChain
#     bm25_ret = BM25Retriever.from_documents(docs)
#     bm25_ret.k = k_bm25

#     return EnsembleRetriever(retrievers=[bm25_ret, faiss_ret], weights=list(weights))

from __future__ import annotations
 
from typing import List, Sequence, Tuple
from pathlib import Path
 
from langchain.retrievers import EnsembleRetriever
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
 
 
from pipelines.processor import process_pdfs_to_chunks
from .faiss_store import build_faiss_store, load_faiss_store, get_faiss_retriever
from .bm25_store import build_bm25_retriever
import config
 
 
def build_or_load_faiss_for_module(
    pdf_dir: str,
    faiss_dir: str,
) -> tuple[List[Document], FAISS]:
    """
    Lädt/erzeugt Chunks aus PDFs und baut/ladet dazu den FAISS-Index.
    """
    # 1) Chunks erzeugen (inkl. Laden + Cleaning + Chunking)
    chunks = process_pdfs_to_chunks(pdf_dir)
 
    # 2) FAISS-Index laden oder neu bauen
    if Path(faiss_dir).exists() and any(Path(faiss_dir).iterdir()):
        vectorstore = load_faiss_store(faiss_dir)
    else:
        vectorstore = build_faiss_store(chunks, faiss_dir)
 
    return chunks, vectorstore
 
 
def build_hybrid_retriever_for_module(
    pdf_dir: str,
    faiss_dir: str,
    k: int = 5,
    weights: Sequence[float] | None = None,
):
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
 
    return hybrid
 
 