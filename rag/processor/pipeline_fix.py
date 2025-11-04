from typing import List

from langchain.schema import Document

from .pdf_text_fix import DocumentLoader
from .textcleaner_fix import clean_documents
from .chunking_fix import Chunker


def process_pdfs_to_chunks(pdf_dir: str | None = None) -> List[Document]:
    """
    End-to-End-Verarbeitung:
    PDFs laden -> Text bereinigen -> in Chunks splitten.
    """
    loader = DocumentLoader()
    docs = loader.load_pdfs(pdf_dir)

    if not docs:
        print("⚠️  Keine Dokumente geladen – keine Chunks erzeugt.")
        return []

    cleaned_docs = clean_documents(docs)
    chunker = Chunker()
    chunks = chunker.split_documents(cleaned_docs)
    return chunks
