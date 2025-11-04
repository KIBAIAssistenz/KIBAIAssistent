import os
from dotenv import load_dotenv
load_dotenv()
 
# === Allgemeine Einstellungen ===
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150
 
# === Datenpfade ===
# PDF-Ordner nach Modul getrennt
DATA_DIR = "data/pdfs"
PDF_DIR_INTRO = f"{DATA_DIR}/einfuehrung_KI"
PDF_DIR_ML = f"{DATA_DIR}/machine_learning"
 
# Verzeichnis für verarbeitete Dateien
PROCESSED_DIR = "data/processed"
 
# === Modelleinstellungen ===
EMBEDDING_BACKEND = "huggingface"  # nur zur Info
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "openai/gpt-oss-20b"  # kann später auch was anderes sein
API_KEY_GROQ = os.environ.get("GROQ_API_KEY")
BASE_URL = "https://api.groq.com/openai/v1"
 
 
# === Debug / Logging ===
VERBOSE = True
 
# === Vectorstore-Pfade ===
FAISS_DIR_INTRO = "rag/stores/einfuehrung_KI"
FAISS_DIR_ML = "rag/stores/machine_learning"

