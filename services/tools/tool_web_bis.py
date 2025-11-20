# services/tools/tool_web_bis.py

import os
from pydantic import BaseModel, Field
from langchain.tools import tool
import wikipedia

# --- User Agent laut Wikipedia-Policy ---
USER_AGENT = "FHNW-KI-Lernassistent/1.0 (https://fhnw.ch)"
os.environ["WIKIPEDIA_USER_AGENT"] = USER_AGENT
wikipedia.set_user_agent(USER_AGENT)
wikipedia.set_lang("de")  # zuerst deutsch


class BISQuestionInput(BaseModel):
    # Feld heißt 'question' → wichtig für Router-Kompatibilität
    question: str = Field(
        ...,
        description="Frage der Studierenden zum Modul 'Business Information Systems' für die Websuche.",
    )


@tool(
    args_schema=BISQuestionInput,
    description="Wikipedia-Websuche für Fragen zu 'Business Information Systems' (BIS), wenn Modulfolien nichts liefern.",
)
def bis_web_search(question: str) -> dict:
    """
    Fallback-Websuche für das Modul 'BIS'.
    Wird nur genutzt, wenn:
        - Die BIS-Chain nichts weiß
        - Kein anderes Modul eine passende Antwort liefert
    """
    try:
        print(f"🔍 [BIS-Web-Tool] Frage: {question}")

        # 1) deutschsprachige Wikipedia-Suche
        hits = wikipedia.search(question)
        if not hits:
            # Fallback englisch
            wikipedia.set_lang("en")
            hits = wikipedia.search(question)

        if not hits:
            wikipedia.set_lang("de")
            return {
                "answer": (
                    "Ich konnte in der Wikipedia-Webrecherche keine passenden Informationen finden."
                ),
                "source_type": "web",
            }

        # 2) erste Seite holen
        page = wikipedia.page(hits[0], auto_suggest=False)
        summary = page.summary

        # Sprache zurücksetzen
        wikipedia.set_lang("de")

        return {
            "answer": (
                "Web-Zusammenfassung (Wikipedia, ggf. leicht vereinfacht):\n\n"
                f"{summary}"
            ),
            "source_type": "web",
        }

    except Exception as e:
        try:
            wikipedia.set_lang("de")
        except:
            pass

        return {
            "answer": f"⚠️ Fehler bei der BIS-Wikipedia-Suche: {e}",
            "source_type": "web",
        }
