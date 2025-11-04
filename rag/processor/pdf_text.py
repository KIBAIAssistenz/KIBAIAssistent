# pipelines/processor/pdf_text.py
"""
Lädt PDF-Dokumente als LangChain-Documents.
"""
import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import config


class DocumentLoader:
    """Nur fürs Laden der PDFs zuständig."""

    def __init__(self):
        pass

    def load_pdfs(self, pdf_dir: str | None = None) -> List[Document]:
        pdf_dir = pdf_dir or config.PDF_DIR_INTRO  # Default z.B. Einführung KI
        all_pages: List[Document] = []

        if not os.path.exists(pdf_dir):
            print(f"⚠️  PDF-Verzeichnis nicht gefunden: {pdf_dir}")
            return all_pages

        pdf_files = sorted(Path(pdf_dir).glob("*.pdf"))
        if not pdf_files:
            print(f"⚠️  Keine PDF-Dateien in {pdf_dir} gefunden")
            return all_pages

        for pdf_path in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
                for i, p in enumerate(pages, start=1):
                    # Metadaten anreichern
                    p.metadata = {**p.metadata, "source": str(pdf_path), "page": i}
                all_pages.extend(pages)
                print(f"✓ PDF geladen: {pdf_path.name} ({len(pages)} Seiten)")
            except Exception as e:
                print(f"✗ Fehler beim Laden von {pdf_path.name}: {e}")

        print(f"\n📄 Insgesamt {len(all_pages)} Seiten aus {len(pdf_files)} PDFs geladen.")
        return all_pages
