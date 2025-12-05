from __future__ import annotations

from typing import List
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

import config


from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import config

def _get_embeddings():
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)

def build_faiss_store(docs: list[Document], persist_dir: str) -> FAISS:
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    vectorstore.save_local(str(persist_path))
    print(f"✅ FAISS-Store gespeichert unter: {persist_path}")
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
    print(f"✅ FAISS-Store geladen von: {persist_path}")
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
