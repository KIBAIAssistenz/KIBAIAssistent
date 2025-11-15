# experts/router.py

from typing import Dict, Optional

# Marker aus deinem Prompt, wenn das Modell auf Basis der Folien nichts weiß
UNKNOWN_MARKERS = [
    "Ich weiss es nicht basierend auf den vorhandenen Dokumenten.",
    "Ich weiß es nicht basierend auf den vorhandenen Dokumenten.",
]


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
      2. Wenn die Antwort NICHT den Standard-„Ich weiss es nicht“-Text enthält:
         → Antwort aus Folien zurückgeben (source_type='folien').

      3. Wenn die Antwort den Marker enthält:
         3.1 Frage alle anderen Module ebenfalls (deren Chain).
             - Wenn ein anderes Modul eine "normale" Antwort geben kann
               (also NICHT den Unknown-Marker), dann:
                 → Hinweise: "Bitte in Modul X wechseln"
                 → KEIN Web-Fallback.
                 → source_type='wrong_module'
         3.2 Nur wenn KEIN anderes Modul sinnvoll antworten kann:
             → Web-Tool des aktiven Moduls aufrufen (source_type='web').
    """

    history = history or []

    # ---------- 1) Aktives Modul ----------
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

    # ---------- 2) Andere Module testen ----------
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

        # Wenn das andere Modul NICHT den Unknown-Satz liefert,
        # dann scheint es dort tatsächlich behandelt zu werden.
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

    # ---------- 3) Kein Modul kann sinnvoll antworten → Web-Fallback ----------
    active_web_tool = web_tools.get(active_expert_name)
    if active_web_tool is None:
        # Kein Web-Tool konfiguriert → ursprüngliche Unknown-Antwort zurückgeben
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

    # Fallback, falls das Tool aus irgendeinem Grund nur einen String liefert
    return {
        "answer": (
            str(tool_output)
            + "\n\n---\n**Quelle:** Externe Webrecherche (Internet), "
              "nicht aus den Vorlesungsfolien."
        ),
        "source_type": "web",
    }
