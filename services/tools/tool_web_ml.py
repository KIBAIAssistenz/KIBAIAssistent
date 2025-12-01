# # services/tools/tool_web_ml.py

# import os
# import re
# from pydantic import BaseModel, Field
# from langchain.tools import tool
# import wikipedia

# # Gleicher User-Agent wie beim KI-Tool
# USER_AGENT = "FHNW-KI-Lernassistent/1.0 (https://fhnw.ch)"
# os.environ["WIKIPEDIA_USER_AGENT"] = USER_AGENT
# wikipedia.set_user_agent(USER_AGENT)
# wikipedia.set_lang("de")


# class MLQuestionInput(BaseModel):
#     # WICHTIG: Feldname 'question' – passt zum Router-Aufruf
#     question: str = Field(
#         ...,
#         description="Frage der Studierenden zum Modul 'Machine Learning' für die Websuche.",
#     )
# def _extract_search_term(question: str) -> str:
#     """
#     Versucht, aus einer natürlichen Frage den eigentlichen Suchbegriff zu extrahieren.

#     Beispiele:
#     - 'Woher kommt Geoffrey Hinton?'                   -> 'Geoffrey Hinton'
#     - 'Wer ist Yann LeCun?'                            -> 'Yann LeCun'
#     - 'Was ist Supervised Learning?'                   -> 'Supervised Learning'
#     - 'Gib mir die Definition von Deep Learning'       -> 'Deep Learning'
#     - 'Gib mir die Definition von k-means von Wikipedia' -> 'k-means'
#     - 'Definition von Random Forest'                   -> 'Random Forest'

#     Falls nichts passt: Frage grob säubern und direkt verwenden.
#     """
#     q = question.strip()
#     # Satzzeichen am Ende entfernen
#     q = re.sub(r"[?!\.]+$", "", q).strip()

#     # 1) Spezialfall: "... Definition von X (von Wikipedia)"
#     m = re.search(r"(?i)definition von (.+)", q)
#     if m:
#         term = m.group(1).strip()
#         # "von Wikipedia" am Ende entfernen, falls vorhanden
#         term = re.sub(r"(?i)\s+von\s+wikipedia$", "", term).strip()
#         return term

#     lower = q.lower()

#     prefixes = [
#         "wer ist ",
#         "wer war ",
#         "was ist ",
#         "was war ",
#         "woher kommt ",
#         "von wo kommt ",
#         "wo kommt ",
#         "erkläre ",
#         "erkläre mir ",
#         "wer hat ",  # z.B. 'wer hat Deep Learning erfunden'
#     ]

#     for pref in prefixes:
#         if lower.startswith(pref):
#             return q[len(pref):].strip()

#     # Fallback: ganze Frage zurückgeben
#     return q


# @tool(
#     args_schema=MLQuestionInput,
#     description="Wikipedia-Websuche für Fragen zum Modul 'Maschinelles Lernen', wenn keine Modulfolien helfen.",
# )
# def ml_web_search(question: str) -> dict:
#     """
#     Fallback-Websuche für das Modul 'Maschinelles Lernen'.

#     - Nutzt Wikipedia (de, ggf. englisch als Fallback)
#     - Wird nur vom Router aufgerufen, wenn die Chain sagt:
#       'Ich weiss es nicht basierend auf den vorhandenen Dokumenten.'
#     - Gibt immer ein Dict mit 'answer' und 'source_type' zurück.
#     """
#     try:
#         print(f"🔍 [ML-Web-Tool] Originalfrage: {question}")
#         search_term = _extract_search_term(question)
#         print(f"🔍 [ML-Web-Tool] Suchbegriff:  {search_term}")

#         # 1) In deutschsprachiger Wikipedia suchen
#         wikipedia.set_lang("de")
#         hits = wikipedia.search(search_term)

#         if not hits:
#             # Fallback: englische Wikipedia
#             wikipedia.set_lang("en")
#             hits = wikipedia.search(search_term)

#         if not hits:
#             # wieder zurück auf de stellen
#             wikipedia.set_lang("de")
#             return {
#                 "answer": (
#                     "Ich habe in der Webrecherche (Wikipedia) keine wirklich "
#                     "passenden Informationen zu deiner Frage gefunden."
#                 ),
#                 "source_type": "web",
#             }

#         # 2) Versuche zuerst, direkt die Seite zum Suchbegriff zu holen
#         try:
#             page = wikipedia.page(search_term, auto_suggest=True)
#         except Exception:
#             # Fallback: erste Trefferseite nehmen
#             page = wikipedia.page(hits[0], auto_suggest=False)

#         summary = page.summary

#         # Sprache wieder auf deutsch zurückstellen
#         wikipedia.set_lang("de")

#         answer_text = (
#             "Web-Zusammenfassung (Wikipedia, eventuell leicht vereinfacht):\n\n"
#             f"{summary}"
#         )

#         return {
#             "answer": answer_text,
#             "source_type": "web",
#         }

#     except Exception as e:
#         # Sprache sicherheitshalber zurücksetzen
#         try:
#             wikipedia.set_lang("de")
#         except Exception:
#             pass

#         return {
#             "answer": f"⚠️ Fehler bei der Webrecherche (Wikipedia): {e}",
#             "source_type": "web",
#         }

# services/tools/tool_web_ml.py

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

# import os
# import re
# from pydantic import BaseModel, Field
# from langchain.tools import tool
# import wikipedia

# # Gleicher User-Agent wie beim KI-Tool
# USER_AGENT = "FHNW-KI-Lernassistent/1.0 (https://fhnw.ch)"
# os.environ["WIKIPEDIA_USER_AGENT"] = USER_AGENT
# wikipedia.set_user_agent(USER_AGENT)
# wikipedia.set_lang("de")


# # ------------------------------
# # 🔍 Query-Normalisierung
# # ------------------------------

# STOPWORDS = {
#     "was", "ist", "sit", "wer", "wie", "wann", "wo", "wieso", "warum",
#     "erkläre", "erklaere", "bitte", "mir", "die", "das", "der", "eine",
#     "ein", "frage", "zu", "von", "im", "in", "bezieht", "sich", "auf"
# }

# def normalize_query(question: str) -> str:
#     """
#     Extrahiert den relevanten Suchbegriff aus einer ganzen Frage.
#     Entfernt Stopwörter, findet Akronyme wie MCP, SVM, CNN, KNN usw.
#     Macht Wikipedia-Suche extrem viel robuster.
#     """

#     # Wörter extrahieren (nur alphanumerische Tokens)
#     tokens = re.findall(r"[A-Za-z0-9]+", question)

#     # Stopwörter filtern
#     filtered = [t for t in tokens if t.lower() not in STOPWORDS]

#     # Wenn nichts übrig bleibt → Fallback: komplettes Fragewort
#     if not filtered:
#         return question.strip()

#     # Große Akronyme bevorzugen (MCP, SVM, SGD, CNN, LSTM ...)
#     caps = [t for t in filtered if t.isupper()]
#     if caps:
#         return caps[0]

#     # Falls kein Akronym, das letzte relevante Token nehmen
#     # (z. B. "Wie funktioniert Gradient Boosting?" → "Boosting")
#     return filtered[-1]


# # ------------------------------
# # Pydantic Input Schema
# # ------------------------------

# class MLQuestionInput(BaseModel):
#     question: str = Field(
#         ...,
#         description="Frage der Studierenden zum Modul 'Machine Learning' für die Websuche.",
#     )


# # ------------------------------
# # 🌐 Web-Tool
# # ------------------------------

# @tool(
#     args_schema=MLQuestionInput,
#     description="Wikipedia-Websuche für Machine-Learning-Themen, wenn keine Modulfolien helfen.",
# )
# def ml_web_search(question: str) -> dict:
#     """
#     Fallback-Websuche für das Modul 'Machine Learning'.

#     Verbesserte Version:
#     - Query Normalisierung (Entfernen von Stopwörtern etc.)
#     - Priorisierung von Abkürzungen / ML-Begriffen
#     - auto_suggest für Tippfehlerkorrektur
#     - Sprachfallback: DE → EN
#     """

#     try:
#         print(f"\n🔍 [ML-Web-Tool] Originalfrage: {question}")

#         # --- 1) Query normalisieren ---
#         query = normalize_query(question)
#         print(f"🔧 [ML-Web-Tool] Normalisierte Query: {query}")

#         # --- 2) Wikipedia-Suche (DE) ---
#         hits = wikipedia.search(query)
#         print(f"📄 [ML-Web-Tool] Treffer (DE): {hits}")

#         # Fallback englisch, falls nötig
#         if not hits:
#             wikipedia.set_lang("en")
#             hits = wikipedia.search(query)
#             print(f"📄 [ML-Web-Tool] Treffer (EN): {hits}")

#         if not hits:
#             # Sprache zurücksetzen
#             wikipedia.set_lang("de")
#             return {
#                 "answer": (
#                     f"Ich habe keine passenden Wikipedia-Informationen zur Anfrage '{query}' gefunden."
#                 ),
#                 "source_type": "web",
#             }

#         # --- 3) Seite laden (mit auto_suggest) ---
#         page = wikipedia.page(hits[0], auto_suggest=True)
#         summary = page.summary

#         # Sprache zurück auf DE setzen
#         wikipedia.set_lang("de")

#         answer_text = (
#             "Web-Zusammenfassung (Wikipedia, automatisch korrigiert & vereinfacht):\n\n"
#             f"{summary}"
#         )

#         return {
#             "answer": answer_text,
#             "source_type": "web",
#         }

#     except Exception as e:
#         try:
#             wikipedia.set_lang("de")
#         except Exception:
#             pass

#         return {
#             "answer": f"⚠️ Fehler bei der Webrecherche (Wikipedia): {e}",
#             "source_type": "web",
#         }
