import os
from pydantic import BaseModel, Field
from langchain.tools import tool
import wikipedia

# Gleicher User-Agent wie beim KI-Tool
USER_AGENT = "FHNW-KI-Lernassistent/1.0 (https://fhnw.ch)"
os.environ["WIKIPEDIA_USER_AGENT"] = USER_AGENT
wikipedia.set_user_agent(USER_AGENT)
wikipedia.set_lang("de")


class MLQuestionInput(BaseModel):
    # WICHTIG: Feldname 'question' – passt zum Router-Aufruf
    question: str = Field(
        ...,
        description="Frage der Studierenden zum Modul 'Machine Learning' für die Websuche.",
    )


@tool(
    args_schema=MLQuestionInput,
    description="Wikipedia-Websuche für Machine-Learning-Themen, wenn keine Modulfolien helfen.",
)
def ml_web_search(question: str) -> dict:
    """
    Fallback-Websuche für das Modul 'Machine Learning'.

    - Nutzt Wikipedia (de, ggf. englisch als Fallback)
    - Wird nur vom Router verwendet, wenn die Chain nichts aus den Folien weiss.
    """
    try:
        print(f"🔍 [ML-Web-Tool] Frage: {question}")

        # 1) Deutschsprachige Wikipedia versuchen
        hits = wikipedia.search(question)
        if not hits:
            # Fallback: englische Wikipedia
            wikipedia.set_lang("en")
            hits = wikipedia.search(question)

        if not hits:
            wikipedia.set_lang("de")
            return {
                "answer": (
                    "Ich habe in der Webrecherche (Wikipedia) keine wirklich "
                    "passenden Informationen zu deiner Frage gefunden."
                ),
                "source_type": "web",
            }

        page = wikipedia.page(hits[0], auto_suggest=False)
        summary = page.summary

        # Sprache zurück auf deutsch
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
        try:
            wikipedia.set_lang("de")
        except Exception:
            pass

        return {
            "answer": f"⚠️ Fehler bei der Webrecherche (Wikipedia): {e}",
            "source_type": "web",
        }
