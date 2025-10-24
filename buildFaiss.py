from pathlib import Path
from PIL import Image
import numpy as np
import json, faiss
from sentence_transformers import SentenceTransformer

IMAGE_FOLDER = Path("data/pdfs/images")
INDEX_PATH   = Path("slides.index")
META_PATH    = Path("slides_meta.json")
MODEL_NAME   = "clip-ViT-B-32"

def list_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return [p for p in sorted(folder.glob("*")) if p.suffix.lower() in exts]

def main(batch_size: int = 32):
    model = SentenceTransformer(MODEL_NAME)

    imgs = list_images(IMAGE_FOLDER)
    if not imgs:
        print("❌ Keine Bilder in", IMAGE_FOLDER.resolve()); return

    names, paths, chunks = [], [], []
    embs_list = []

    for p in imgs:
        chunks.append(Image.open(p).convert("RGB"))
        names.append(p.name)
        paths.append(str(p))
        if len(chunks) == batch_size:
            embs = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
            embs_list.append(embs); chunks = []
    if chunks:
        embs = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        embs_list.append(embs)

    embeddings = np.vstack(embs_list).astype("float32")
    d = embeddings.shape[1]

    index = faiss.IndexFlatIP(d)      # IP + normalisierte Vektoren -> Cosine
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps([{"filename": n, "path": p} for n,p in zip(names, paths)],
                                    ensure_ascii=False, indent=2))

    print(f"✅ Index gespeichert: {INDEX_PATH} | Meta: {META_PATH}")
    print(f"📦 Vektoren: {len(names)} | Dim: {d}")

if __name__ == "__main__":
    main()