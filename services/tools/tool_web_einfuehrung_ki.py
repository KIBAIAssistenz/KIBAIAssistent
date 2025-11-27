# services/tools/tool_web_einfuehrung_ki.py

import os
import re
from pydantic import BaseModel, Field
from langchain.tools import tool
import wikipedia

# --- User Agent für Wikipedia setzen (Pflicht laut Policy) ---
USER_AGENT = "FHNW-KI-Lernassistent/1.0 (https://fhnw.ch)"
os.environ["WIKIPEDIA_USER_AGENT"] = USER_AGENT
wikipedia.set_user_agent(USER_AGENT)
wikipedia.set_lang("de")  # zuerst deutsch versuchen


class KIQuestionInput(BaseModel):
    # WICHTIG: Feld heisst 'question', damit es zum Router passt
    question: str = Field(
        ...,
        description="Frage der Studierenden zum Modul 'Einführung in die KI' für die Websuche.",
    )


def _extract_search_term(question: str) -> str:
    """
    Versucht, aus einer natürlichen Frage den eigentlichen Suchbegriff zu extrahieren.

    Beispiele:
    - 'Woher kommt Joseph Weizenbaum?'      -> 'Joseph Weizenbaum'
    - 'Wer ist Alan Turing?'                -> 'Alan Turing'
    - 'Was ist Machine Learning?'           -> 'Machine Learning'
    - 'Gib mir die Definition von KI von Wikipedia' -> 'KI'
    - 'Definition von künstlicher Intelligenz'      -> 'künstliche Intelligenz'

    Falls nichts passt: Frage grob säubern und direkt verwenden.
    """
    q = question.strip()
    # Satzzeichen am Ende entfernen
    q = re.sub(r"[?!\.]+$", "", q).strip()

    # 1) Spezialfall: "... Definition von X (von Wikipedia)"
    m = re.search(r"(?i)definition von (.+)", q)
    if m:
        term = m.group(1).strip()
        # "von Wikipedia" am Ende entfernen, falls vorhanden
        term = re.sub(r"(?i)\s+von\s+wikipedia$", "", term).strip()
        return term

    lower = q.lower()

    prefixes = [
        "wer ist ",
        "wer war ",
        "was ist ",
        "was war ",
        "woher kommt ",
        "von wo kommt ",
        "wo kommt ",
        "erkläre ",
        "erkläre mir ",
    ]

    for pref in prefixes:
        if lower.startswith(pref):
            return q[len(pref):].strip()

    # Fallback: ganze Frage zurückgeben
    return q

@tool(
    args_schema=KIQuestionInput,
    description="Wikipedia-Websuche für Fragen zu 'Einführung in die KI', wenn keine Modulfolien helfen.",
)
def ki_web_search(question: str) -> dict:
    """
    Fallback-Websuche für das Modul 'Einführung in die KI'.

    - Nutzt Wikipedia (de, ggf. englisch als Fallback)
    - Wird nur vom Router aufgerufen, wenn die Chain sagt:
      'Ich weiss es nicht basierend auf den vorhandenen Dokumenten.'
    - Gibt immer ein Dict mit 'answer' und 'source_type' zurück.
    """
    try:
        print(f"🔍 [KI-Web-Tool] Originalfrage: {question}")
        search_term = _extract_search_term(question)
        print(f"🔍 [KI-Web-Tool] Suchbegriff:  {search_term}")

        # 1) In deutschsprachiger Wikipedia suchen
        wikipedia.set_lang("de")
        hits = wikipedia.search(search_term)

        if not hits:
            # Fallback: englische Wikipedia
            wikipedia.set_lang("en")
            hits = wikipedia.search(search_term)

        if not hits:
            # wieder zurück auf de stellen
            wikipedia.set_lang("de")
            return {
                "answer": (
                    "Ich habe in der Webrecherche (Wikipedia) keine wirklich "
                    "passenden Informationen zu deiner Frage gefunden."
                ),
                "source_type": "web",
            }

        # 2) Versuche zuerst, direkt die Seite zum Suchbegriff zu holen
        try:
            page = wikipedia.page(search_term, auto_suggest=True)
        except Exception:
            # Fallback: erste Trefferseite nehmen
            page = wikipedia.page(hits[0], auto_suggest=False)

        summary = page.summary

        # Sprache wieder auf deutsch zurückstellen
        wikipedia.set_lang("de")

        answer_text = (
            "Web-Zusammenfassung (Wikipedia, eventuell leicht vereinfacht):\n\n"
            f"{summary}"
        )

        return {
            "answer": answer_text,
            "source_type": "web",
        }

    except Exception as e:
        # Sprache sicherheitshalber zurücksetzen
        try:
            wikipedia.set_lang("de")
        except Exception:
            pass

        return {
            "answer": f"⚠️ Fehler bei der Webrecherche (Wikipedia): {e}",
            "source_type": "web",
        }
