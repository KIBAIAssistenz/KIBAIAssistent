# # services/tools/tool_web_bis.py

# import os
# import re
# from pydantic import BaseModel, Field
# from langchain.tools import tool
# import wikipedia

# # --- User Agent laut Wikipedia-Policy ---
# USER_AGENT = "FHNW-KI-Lernassistent/1.0 (https://fhnw.ch)"
# os.environ["WIKIPEDIA_USER_AGENT"] = USER_AGENT
# wikipedia.set_user_agent(USER_AGENT)
# wikipedia.set_lang("de")  # zuerst deutsch


# class BISQuestionInput(BaseModel):
#     # Feld heißt 'question' → wichtig für Router-Kompatibilität
#     question: str = Field(
#         ...,
#         description="Frage der Studierenden zum Modul 'Business Information Systems' für die Websuche.",
#     )

# def _extract_search_term(question: str) -> str:
#     """
#     Versucht, aus einer natürlichen Frage den eigentlichen Suchbegriff zu extrahieren.

#     Beispiele:
#     - 'Was ist ein ERP-System?'                         -> 'ERP-System'
#     - 'Definition von Enterprise Service Bus'          -> 'Enterprise Service Bus'
#     - 'Gib mir die Definition von Geschäftsprozess'    -> 'Geschäftsprozess'
#     - 'Wer ist Peter Mertens?'                         -> 'Peter Mertens'

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
#         "wer hat ",
#     ]

#     for pref in prefixes:
#         if lower.startswith(pref):
#             return q[len(pref):].strip()

#     # Fallback: ganze Frage zurückgeben
#     return q


# @tool(
#     args_schema=BISQuestionInput,
#     description="Wikipedia-Websuche für Fragen zu 'Betriebliche Informationssysteme (BIS)', wenn keine Modulfolien helfen.",
# )
# def bis_web_search(question: str) -> dict:
#     """
#     Fallback-Websuche für das Modul 'Betriebliche Informationssysteme (BIS)'.

#     - Nutzt Wikipedia (de, ggf. englisch als Fallback)
#     - Wird nur vom Router aufgerufen, wenn die Chain sagt:
#       'Ich weiss es nicht basierend auf den vorhandenen Dokumenten.'
#     - Gibt immer ein Dict mit 'answer' und 'source_type' zurück.
#     """
#     try:
#         print(f"🔍 [BIS-Web-Tool] Originalfrage: {question}")
#         search_term = _extract_search_term(question)
#         print(f"🔍 [BIS-Web-Tool] Suchbegriff:  {search_term}")

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