from typing import List

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

import config


class Chunker:
    """Chunkt bereits bereinigte Documents in kleinere Textstücke."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        print("\n" + "=" * 60)
        print("✂️  DOKUMENTE SPLITTEN")
        print("=" * 60 + "\n")

        splits = self.splitter.split_documents(documents)

        lengths = [len(s.page_content) for s in splits] or [0]
        print(f"Original Dokumente: {len(documents)}")
        print(f"Chunks nach Split: {len(splits)}")
        print(f"Durchschnittliche Länge: {np.mean(lengths):.1f} Zeichen")
        print(f"Min: {int(np.min(lengths))} | Max: {int(np.max(lengths))} Zeichen")
        return splits
