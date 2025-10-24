from pathlib import Path
from typing import List, Dict, Literal
import json, faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

INDEX_PATH = Path("slides.index")
META_PATH  = Path("slides_meta.json")
MODEL_NAME = "clip-ViT-B-32"

_model = SentenceTransformer(MODEL_NAME)
_index = faiss.read_index(str(INDEX_PATH))
_meta  = json.loads(META_PATH.read_text())  # Liste von dicts

def _encode_text(q: str) -> np.ndarray:
    return _model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

def _encode_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return _model.encode([img], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

def retrieve(query: str, mode: Literal["text","image"]="text", k: int = 5,
             score_threshold: float | None = None) -> List[Dict]:
    q = _encode_text(query) if mode=="text" else _encode_image(query)
    scores, idxs = _index.search(q, k)
    hits: List[Dict] = []
    for score, i in zip(scores[0], idxs[0]):
        if i == -1: continue
        if score_threshold is not None and float(score) < score_threshold: continue
        m = _meta[i]
        hits.append({"filename": m["filename"], "path": m["path"], "score": float(score)})
    return hits

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print('Usage:\n  python retriever.py text  "deine Anfrage"\n  python retriever.py image path/zum/bild.png')
        raise SystemExit
    mode = sys.argv[1]
    query = " ".join(sys.argv[2:]) if mode=="text" else sys.argv[2]
    for h in retrieve(query, mode=mode, k=5):
        print(f"{h['score']:.4f}  {h['filename']}  {h['path']}")
