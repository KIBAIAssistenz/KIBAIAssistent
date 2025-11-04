# python -m app.test_hybrid_intro
from __future__ import annotations
import sys, pathlib

# Projektwurzel auf sys.path setzen (wie bei dir)
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.retriever_fix import build_hybrid_retriever_for_module
import config


def main():
    # 1) Hybrid-Retriever für Einführung KI bauen / laden
    pdf_dir = config.PDF_DIR_INTRO      # z.B. "data/pdfs/einführung_ki"
    faiss_dir = config.FAISS_DIR_INTRO  # z.B. "stores/faiss/einführung_ki"

    intro_hybrid = build_hybrid_retriever_for_module(
        pdf_dir=pdf_dir,
        faiss_dir=faiss_dir,
        k=5,
        weights=[0.5, 0.5],
    )

    # 2) Testfrage
    query = "Was ist der Unterschied zwischen starker und schwacher KI?"
    print(f"\n🔎 Frage: {query}\n")

    # 3) Dokumente abrufen
    # EnsembleRetriever ist ein Retriever → .invoke gibt Chunks zurück
    docs = intro_hybrid.invoke(query)

    if not docs:
        print("Keine Dokumente gefunden.")
        return

    print(f"📄 Gefundene Chunks: {len(docs)}\n")

    # 4) Ergebnisse anzeigen
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        print(f"--- Treffer {i} ---")
        print("Quelle:", meta.get("source"))
        print("Seite:", meta.get("page"))
        # chunk_index gibt es nur, wenn du es im Chunker setzt – also optional:
        if "chunk_index" in meta:
            print("Chunk:", meta.get("chunk_index"))
        print(d.page_content[:500], "\n")


if __name__ == "__main__":
    main()
