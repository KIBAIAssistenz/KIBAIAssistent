import os
from langchain_openai import ChatOpenAI
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

LLM_JUDGE_MODEL_CEREBRAS = "llama3.1-8b"

API_KEY_GROQ = os.environ.get("GROQ_API_KEY")
API_KEY_CEREBRAS = os.environ.get("CEREBRAS_API_KEY")

BASE_URL_GROQ = "https://api.groq.com/openai/v1"
BASE_URL_CEREBRAS = "https://api.cerebras.ai/v1"

 
# === Debug / Logging ===
VERBOSE = True
 
# === Vectorstore-Pfade ===
FAISS_DIR_INTRO = "rag/stores/einfuehrung_KI"
FAISS_DIR_ML = "rag/stores/machine_learning"
FAISS_DIR_BIS = "rag/stores/bis"
PDF_DIR_BIS = "data/pdfs/bis"

# === Experten LLM-Modelle bauen ===
EXPERT_TEMP = 0.3
EXPERT_LLM_MODEL = "gpt-oss-120b"

#EXPERT_LLM_MODEL = "llama-3.3-70b"

llm= ChatOpenAI(
    model=EXPERT_LLM_MODEL,
    temperature=EXPERT_TEMP,
    base_url=BASE_URL_CEREBRAS,
    api_key=API_KEY_CEREBRAS,
)

