# pipelines/processor/textcleaner.py
import re
from typing import Iterable, List

from langchain.schema import Document

# typische Muster aus deinen Folien / Zusammenfassungen
HEADER_FOOTER_HINTS = [
    r"(Einführung in KI|Einfuehrung in KI|Maschinelles Lernen)",
    r"(Bachelor Angewandte Informatik|BAI)",
    r"(Dozent|Prof\.|Professorin?)",
    r"(Sommersemester|Wintersemester|SS\d{2}|WS\d{2})",
    r"(Seite|Page)\s*\d+",
    r"\d{2}\.\d{2}\.\d{4}",  # Datum wie 12.04.2025
]


def remove_headers_footers(
    text: str,
    extra_hints: Iterable[str] | None = None,
    drop_short_lines: bool = False,
    min_len: int = 20,
) -> str:
    hints = list(HEADER_FOOTER_HINTS) + list(extra_hints or [])
    hint_re = re.compile("|".join(hints), re.IGNORECASE)

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if hint_re.search(line or ""):
            continue
        if drop_short_lines and len(line.strip()) < min_len:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def fix_pdf_hyphenation(text: str) -> str:
    # Silbentrennungen: Ler-\nnen → Lernen
    text = re.sub(r"-\s*\n(?=[a-zäöüß])", "", text)
    # Mehrfache Umbrüche normalisieren
    text = re.sub(r"\n+", "\n", text)
    return text


def clean_text(t: str) -> str:
    t = remove_headers_footers(t)
    t = fix_pdf_hyphenation(t)
    t = re.sub(r"\s+", " ", t)           # Whitespace normalisieren
    t = re.sub(r"[–—]", "-", t)          # Gedankenstriche vereinheitlichen
    return t.strip()


def clean_documents(docs: List[Document]) -> List[Document]:
    """Hilfsfunktion: ganze Document-Liste bereinigen."""
    cleaned_docs: List[Document] = []
    for d in docs:
        txt = clean_text(d.page_content or "")
        cleaned_docs.append(
            Document(page_content=txt, metadata=d.metadata)
        )
    return cleaned_docs
