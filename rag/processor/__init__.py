# pipelines/processor/__init__.py
from typing import List
from langchain.schema import Document

from .pdf_text import DocumentLoader
from .textcleaner import clean_documents
from .chunking import Chunker


def process_pdfs_to_chunks(
    pdf_dir: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """
    High-Level Pipeline:
    PDF -> load -> clean -> chunk
    Wird von Notebook / RAG / Agenten aufgerufen.
    """
    loader = DocumentLoader()
    raw_docs = loader.load_pdfs(pdf_dir)

    if not raw_docs:
        raise ValueError(f"Keine Dokumente geladen aus: {pdf_dir}")

    print("🧼 Texte bereinigen …")
    cleaned_docs = clean_documents(raw_docs)

    chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.split_documents(cleaned_docs)

    # Optional: Chunk-Index für Debugging / Logging
    for i, c in enumerate(chunks):
        c.metadata = {**(c.metadata or {}), "chunk_index": i}

    print(f"✅ Fertig: {len(chunks)} Chunks erzeugt.")
    return chunks
