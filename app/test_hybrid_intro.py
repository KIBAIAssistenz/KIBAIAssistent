# python -m app.test_hybrid_intro
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rag.retriever import load_or_build_hybrid_retriever
import config
from rag.processor import DocumentLoader
 
def main():
    # 1) Hybrid-Retriever für Einführung KI bauen / laden
    intro_hybrid = load_or_build_hybrid_retriever(
        pdf_dir=config.PDF_DIR_INTRO,
        faiss_dir=config.FAISS_DIR_INTRO,
        k=5,
        weights=[0.5, 0.5],  
    )
 
    # 2) Testfrage
    query = "Was ist der Unterschied zwischen starker und schwacher KI?"
    print(f"\n🔎 Frage: {query}\n")
 
    # 3) Dokumente abrufen
    #   (EnsembleRetriever ist ein Retriever, d.h. get_relevant_documents / invoke gibt Chunks zurück)
    docs = intro_hybrid.invoke(query)
    if docs is None:
        print("Keine Dokumente gefunden.")
    else:
        print(f"📄 Gefundene Chunks: {len(docs)}\n")
 
    # 4) Ergebnisse anzeigen
        for i, d in enumerate(docs, start=1):
            print(f"--- Treffer {i} ---")
            print("Quelle:", d.metadata.get("source"))
            print("Seite:", d.metadata.get("page"), "| Chunk:", d.metadata.get("chunk_index"))
            print(d.page_content[:500], "\n")
 
 
if __name__ == "__main__":
    main()