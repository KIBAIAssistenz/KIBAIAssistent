# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain.schema import Document
# from pathlib import Path

# def load_pdfs_einführung_KI():
#     """Lädt die PDFs für 'einführung_KI' und gibt alle Seiten als Liste von Dokumenten zurück."""

#     pdf_dir = Path("data/pdfs/einführung_KI")  # Speicherort der PDFs

#     pdf_files = [
#         "KI Ueberblick Teil 1.pdf",
#         "KI Ueberblick Teil 2.pdf",
#         "Problemloesen_als_Suche.pdf",
#         "Machine Learning_exam.pdf",
#         "Machine Learning.pdf",
#         "Wissensrepraesentation.pdf",
#         "Aussagenlogik.pdf",
#         "Praedikatenlogik.pdf",
#         "Deep Learning_exam.pdf",
#         "Deep Learning.pdf",
#     ]
# #
#     all_pages_pdf = []

#     for name in pdf_files:
#         pdf_path = pdf_dir / name
#         if not pdf_path.exists():
#             print(f"❌ Datei nicht gefunden: {pdf_path}")
#             continue

#         loader = PyMuPDFLoader(str(pdf_path))
#         pages = loader.load()

#         # Metadaten hinzufügen
#         for p in pages:
#             p.metadata.update({
#                 "module": "einführung_KI",       # <–– Einheitlich!
#                 "source_path": str(pdf_path),
#                 "source_name": name,
#             })

#         all_pages_pdf.extend(pages)
#         print(f"✅ {name}: {len(pages)} Seiten geladen")

#     print(f"\n📚 Insgesamt {len(all_pages_pdf)} Seiten aus {len(pdf_files)} PDF-Dateien geladen.")
#     return all_pages_pdf


# def load_pdfs_machine_learning():
#     """Lädt die PDFs für 'machine_learning' und gibt alle Seiten als Liste von Dokumenten zurück."""

#     pdf_dir = Path("data/pdfs/machine_learning")  # Speicherort der PDFs

#     pdf_files = [
#         "Folien_01_Intro_ML.pdf",
#         "Folien_02_Datenaufbereitung.pdf",
#         "Folien_03_Klassifikation.pdf",
#         "Folien_04_Regression.pdf",
#         "Folien_05_Evaluation.pdf",
#         "Folien_06_Typische_Probleme.pdf",
#         "Folien_07_Dimensionsreduktion.pdf",
#         "Folien_08_Unsupervised.pdf",
#         "Folien_09_Interpretierbarkeit.pdf",
#         "Folien_10_Ethische_Fragen",
#     ]

#     all_pages_pdf = []

#     for name in pdf_files:
#         pdf_path = pdf_dir / name
#         if not pdf_path.exists():
#             print(f"❌ Datei nicht gefunden: {pdf_path}")
#             continue

#         loader = PyMuPDFLoader(str(pdf_path))
#         pages = loader.load()

#         # Metadaten hinzufügen
#         for p in pages:
#             p.metadata.update({
#                 "module": "machine_learning",       # <–– Einheitlich!
#                 "source_path": str(pdf_path),
#                 "source_name": name,
#             })

#         all_pages_pdf.extend(pages)
#         print(f"✅ {name}: {len(pages)} Seiten geladen")

#     print(f"\n📚 Insgesamt {len(all_pages_pdf)} Seiten aus {len(pdf_files)} PDF-Dateien geladen.")
#     return all_pages_pdf