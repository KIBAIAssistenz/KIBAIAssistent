# from __future__ import annotations
# import sys, pathlib
# sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# from pathlib import Path
# import faiss
# from typing import Literal

# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_community.docstore.in_memory import InMemoryDocstore
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings


# # -> halte deine Loader separat in rag/pdf_reader.py
# from rag.processor.pdf_text import load_pdfs


# EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
# CHUNK_SIZE = 700
# CHUNK_OVERLAP = 120

# def _load_docs(subdir: Literal["einführung_KI", "machine_learning"]):
#     if subdir == "einführung_KI":
#         return load_pdfs_einführung_KI()
#     elif subdir == "machine_learning":
#         return load_pdfs_machine_learning()
#     else:
#         raise ValueError(f"Unbekanntes Subdir: {subdir}")

# def build_index_for(subdir: Literal["einführung_KI", "machine_learning"]) -> Path:
#     """
#     Baut einen FAISS-Index für das angegebene Subdir und speichert ihn unter rag/stores/<subdir>.
#     Gibt den Pfad zum Store zurück.
#     """
#     docs = _load_docs(subdir)
#     if not docs:
#         raise RuntimeError(f"Keine Dokumente gefunden für '{subdir}'.")

#     splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
#     chunks = splitter.split_documents(docs)

#     emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
#     dim = len(emb.embed_query("hello"))

#     # L2-Index; Embeddings werden in der Vectorstore-Layer L2-normalisiert
#     index = faiss.IndexFlatL2(dim)

#     vs = FAISS(
#         embedding_function=emb,
#         index=index,
#         docstore=InMemoryDocstore(),
#         index_to_docstore_id={},
#         normalize_L2=True,  # Kosinus-Ähnlichkeit via L2-Normalisierung
#     )

#     vs.add_documents(chunks)

#     store = Path(f"rag/stores/{subdir}")
#     store.mkdir(parents=True, exist_ok=True)
#     vs.save_local(str(store))
#     print(f"✅ FAISS gespeichert: {store}")
#     return store

from __future__ import annotations
 
from typing import List
from pathlib import Path
 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain.schema import Document
 
import config
 
 
def _get_embeddings():
    """
    Liefert das Embedding-Objekt. Aktuell: HuggingFace (SentenceTransformers).
    """
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
 
 
def build_faiss_store(
    docs: List[Document],
    persist_dir: str,
) -> FAISS:
    """
    Erzeugt einen neuen FAISS-Vectorstore aus Dokumenten und speichert ihn lokal.
    """
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
 
    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    vectorstore.save_local(str(persist_path))
 
    return vectorstore
 
 
def load_faiss_store(
    persist_dir: str,
) -> FAISS:
    """
    Lädt einen bestehenden FAISS-Vectorstore von Platte.
    """
    persist_path = Path(persist_dir)
    if not persist_path.exists():
        raise FileNotFoundError(f"FAISS-Verzeichnis nicht gefunden: {persist_path}")
 
    embeddings = _get_embeddings()
    vectorstore = FAISS.load_local(
        str(persist_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore
 
 
def get_faiss_retriever(
    vectorstore: FAISS,
    k: int = 5,
):
    """
    Macht aus einem FAISS-Vectorstore einen Retriever.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return retriever
 
