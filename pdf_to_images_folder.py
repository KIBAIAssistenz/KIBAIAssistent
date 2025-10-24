from pathlib import Path
import fitz  # PyMuPDF

def pdf_to_images_folder(pdf_dir="data/pdfs", img_subfolder="images", dpi=180):
    pdf_dir = Path(pdf_dir)
    img_dir = pdf_dir / img_subfolder
    img_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print("❌ Keine PDF-Dateien gefunden:", pdf_dir.resolve())
        return

    total = 0
    for pdf_path in pdf_files:
        print(f"📄 {pdf_path.name}")
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=dpi)  # falls TypeError: PyMuPDF updaten
            out = img_dir / f"{pdf_path.stem}_slide_{i:03d}.png"
            pix.save(out.as_posix())
            total += 1
        doc.close()
    print(f"✅ Export fertig: {total} Seiten -> {img_dir}")

if __name__ == "__main__":
    pdf_to_images_folder("data/pdfs")



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

