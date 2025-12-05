from typing import Dict, Optional

# Marker aus deinem Prompt, wenn das Modell auf Basis der Folien nichts weiß
UNKNOWN_MARKERS = [
    "Ich weiss es nicht basierend auf den vorhandenen Dokumenten.",
    "Ich weiß es nicht basierend auf den vorhandenen Dokumenten.",
]

# ---------------------------------------------------------
# 🔧 Tippfehlerkorrektur (Option B)
# ---------------------------------------------------------
def fix_common_typos(q: str) -> str:
    """
    Korrigiert kleine, häufige Tippfehler, die dafür sorgen,
    dass die Experten-Chains sofort 'Unknown' zurückgeben.
    """
    words = q.split()
    fixed = []

    for w in words:
        lw = w.lower()

        # Häufigster Fehler: "sit" → "ist"
        if lw == "sit":
            fixed.append("ist")
            continue

        # Varianten mit Satzzeichen
        if lw in ["sit?", "sit.", "ist?", "ist."]:
            fixed.append("ist")
            continue

        # generische kleine Korrektur für Wörter, die fast wie "ist" aussehen
        if len(w) <= 3 and sum(a != b for a, b in zip(lw.ljust(3), "ist")) <= 1:
            fixed.append("ist")
            continue

        fixed.append(w)

    return " ".join(fixed)


# ---------------------------------------------------------
#   Router Core Logic
# ---------------------------------------------------------

def _is_unknown_answer(answer: str) -> bool:
    """Prüft, ob die Antwort einer der 'ich weiss es nicht'-Sätze ist."""
    if not isinstance(answer, str):
        return False
    return any(marker in answer for marker in UNKNOWN_MARKERS)


def answer_with_module_and_web_fallback(
    *,
    active_expert_name: str,        # z.B. "Einführung in die KI"
    experts: Dict[str, dict],       # {"Einführung in die KI": {...}, "Machine Learning": {...}}
    web_tools: Dict[str, object],   # {"Einführung in die KI": ki_web_search, ...}
    question: str,
    history: Optional[list] = None,
) -> dict:
    """
    Routing-Logik:

      1. Immer zuerst die Chain des aktiven Moduls aufrufen (mit Folien).
      2. Wenn die Antwort NICHT Unknown ist:
         → Antwort aus Folien zurückgeben.
      3. Wenn Unknown:
         - prüfen, ob es zu einem anderen Modul gehört
         - ansonsten Web-Fallback.
    """

    history = history or []

    # ---------------------------------------------------------
    # 🔥 0) Tippfehlerkorrektur BEFORE alles andere
    # ---------------------------------------------------------
    question = fix_common_typos(question)

    # ---------------------------------------------------------
    # 1) Chain des aktiven Moduls aufrufen
    # ---------------------------------------------------------
    active_expert = experts[active_expert_name]
    chain = active_expert["chain"]

    try:
        chain_input = {
            "question": question,
            "history": history,
        }
        answer = chain.invoke(chain_input)
    except Exception as e:
        return {
            "answer": f"⚠️ Fehler beim Aufruf der Experten-Chain: {e}",
            "source_type": "none",
        }

    # Wenn das aktive Modul eine normale Antwort liefern konnte → fertig
    if not _is_unknown_answer(answer):
        return {
            "answer": answer,
            "source_type": "folien",
        }

    # ---------------------------------------------------------
    # 2) Andere Module testen
    # ---------------------------------------------------------
    candidate_module = None

    for name, expert in experts.items():
        if name == active_expert_name:
            continue

        other_chain = expert.get("chain")
        if other_chain is None:
            continue

        try:
            other_answer = other_chain.invoke(
                {
                    "question": question,
                    "history": history,
                }
            )
        except Exception:
            continue

        if not _is_unknown_answer(other_answer):
            candidate_module = name
            break

    if candidate_module is not None:
        hint = (
            f"Deine Frage scheint inhaltlich besser zum Modul **'{candidate_module}'** zu passen.\n\n"
            f"Du befindest dich aktuell im Modul **'{active_expert_name}'**.\n"
            "Bitte wechsle in das passende Modul, um die erklärenden Folien und Antworten zu erhalten."
        )
        return {
            "answer": hint,
            "source_type": "wrong_module",
        }

    # ---------------------------------------------------------
    # 3) Web-Fallback, wenn kein Modul sinnvoll antworten kann
    # ---------------------------------------------------------
    active_web_tool = web_tools.get(active_expert_name)
    if active_web_tool is None:
        return {
            "answer": answer,
            "source_type": "folien",
        }

    print(f"[Router] Fallback auf Web-Tool für Modul: {active_expert_name}")
    tool_output = active_web_tool.invoke({"question": question})

    if isinstance(tool_output, dict) and "answer" in tool_output:
        answer_text = (
            tool_output["answer"]
            + "\n\n---\n**Quelle:** Externe Webrecherche (Internet), "
              "nicht aus den Vorlesungsfolien."
        )
        return {
            "answer": answer_text,
            "source_type": "web",
        }

    # Fallback, falls Tool nur einen String liefert
    return {
        "answer": (
            str(tool_output)
            + "\n\n---\n**Quelle:** Externe Webrecherche (Internet), "
              "nicht aus den Vorlesungsfolien."
        ),
        "source_type": "web",
    }

