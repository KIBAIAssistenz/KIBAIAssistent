# # UI.py
# import os
# import gradio as gr

# from services.llm_connector import llm

# from experts.einführung_KI.expert_einführung_KI import build_einführung_KI_expert
# from experts.machine_learning.expert_ml import build_machine_learning_expert

# os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# EXPERT_FACTORIES = {
#     "Einführung in die KI": lambda: build_einführung_KI_expert(llm),
#     "Machine Learning":     lambda: build_machine_learning_expert(llm),
# }
# EXPERT_CACHE = {}
# DEFAULT_MODULE = "Einführung in die KI"

# def get_expert(label: str):
#     if label in EXPERT_CACHE:
#         return EXPERT_CACHE[label]
#     print(f"[UI] Baue Experte '{label}' ...")
#     exp = EXPERT_FACTORIES[label]()   # kann beim ersten Mal länger dauern
#     EXPERT_CACHE[label] = exp
#     print(f"[UI] Experte '{label}' bereit.")
#     return exp

# def answer(message, history, module_label: str):
#     try:
#         exp = get_expert(module_label)
#     except Exception as e:
#         return f"⚠️ Konnte den Experten '{module_label}' nicht laden: {e}"
#     try:
#         # Wenn deine Chain History braucht: exp["chain"].invoke({"question": message, "history": history})
#         return exp["chain"].invoke(message)
#     except Exception as e:
#         return f"⚠️ Fehler bei der Anfrage: {e}"

# def build_app():
#     with gr.Blocks(title="BAI Lernassistent") as demo:
#         gr.Markdown("# 📘 Wähle dein Modul")

#         module_radio = gr.Radio(
#             choices=list(EXPERT_FACTORIES.keys()),
#             value=DEFAULT_MODULE,
#             label="Modul",
#             info="Du kannst jederzeit wechseln."
#         )
#         info = gr.Markdown(f"✅ **{DEFAULT_MODULE}** gewählt.")
#         preload_btn = gr.Button("🔄 Modul vorladen (optional)")

#         chat = gr.ChatInterface(
#             fn=answer,
#             additional_inputs=[module_radio],
#             type="messages",                      # falls das bei dir Probleme macht, entferne die Zeile
#             title="Chat zum gewählten Modul",
#             textbox=gr.Textbox(placeholder="Stelle deine Frage zum gewählten Modul …"),
#         )

#         def on_module_change(sel):
#             return gr.update(value=f"✅ **{sel}** gewählt.")
#         module_radio.change(on_module_change, inputs=module_radio, outputs=info)

#         def preload(sel):
#             try:
#                 get_expert(sel)
#                 return gr.update(value=f"✅ **{sel}** geladen und bereit.")
#             except Exception as e:
#                 return gr.update(value=f"❌ Fehler beim Laden von **{sel}**: {e}")
#         preload_btn.click(preload, inputs=module_radio, outputs=info)

#     return demo

# if __name__ == "__main__":
#     print("[UI] baue App …")
#     app = build_app()
#     print("[UI] starte Gradio …")
#     app.launch(
#         inbrowser=True,
#         server_name="127.0.0.1",
#         server_port=7860,   # bei Konflikt eine andere Zahl nehmen, z.B. 7861
#         show_error=True,
#     )

# UI.py

# app/UI.py

# app/UI.py

import os
import sys
import pathlib

# 🔧 Pfad fixen: gehe eine Ebene über /app → Projektwurzel (KIBAIAssistent-1)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import gradio as gr

# LLM / Services
from services.llm_connector import llm

# Experten
from experts.einführung_KI.expert_einführung_KI import build_einführung_KI_expert
from experts.machine_learning.expert_ml import build_machine_learning_expert

# Router + Web-Tools
from experts.router import answer_with_module_and_web_fallback
from services.tools.tool_web_einfuehrung_ki import ki_web_search
from services.tools.tool_web_ml import ml_web_search

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# =========================================
# Experten-Definition & Caching
# =========================================

EXPERT_FACTORIES = {
    "Einführung in die KI": lambda: build_einführung_KI_expert(llm),
    "Machine Learning":     lambda: build_machine_learning_expert(llm),
}
EXPERT_CACHE = {}
DEFAULT_MODULE = "Einführung in die KI"

# Web-Tools passend zu den Modulen
WEB_TOOLS = {
    "Einführung in die KI": ki_web_search,
    "Machine Learning":     ml_web_search,
}


def get_expert(label: str):
    """Lädt (und cached) den Experten für das gewählte Modul."""
    if label in EXPERT_CACHE:
        return EXPERT_CACHE[label]

    print(f"[UI] Baue Experte '{label}' ...")
    exp = EXPERT_FACTORIES[label]()   # kann beim ersten Mal länger dauern
    EXPERT_CACHE[label] = exp
    print(f"[UI] Experte '{label}' bereit.")
    return exp


# =========================================
# Antwort-Funktion für den Chat
# =========================================

def answer(message, history, module_label: str):
    """
    message: letzte User-Nachricht (String)
    history: bisheriger Verlauf (Liste von (user, assistant)-Tupeln)
    module_label: aktuell gewähltes Modul (z.B. 'Einführung in die KI')
    """
    print(f"[UI] Frage: {message} | Modul: {module_label}")

    # Alle Experten-Objekte (mit Cache) bereitstellen
    try:
        experts = {label: get_expert(label) for label in EXPERT_FACTORIES.keys()}
    except Exception as e:
        return f"⚠️ Konnte die Experten nicht laden: {e}"

    try:
        result = answer_with_module_and_web_fallback(
            active_expert_name=module_label,
            experts=experts,
            web_tools=WEB_TOOLS,
            question=message,
            history=history,
        )

        answer_text = result["answer"]
        source_type = result.get("source_type", "unknown")
        print(f"[UI] source_type = {source_type}")

        return answer_text

    except Exception as e:
        print(f"[UI] Fehler in answer(): {e}")
        return f"⚠️ Fehler bei der Anfrage: {e}"


# =========================================
# Gradio-App
# =========================================

def build_app():
    with gr.Blocks(title="BAI Lernassistent") as demo:
        gr.Markdown("# 📘 Wähle dein Modul")

        module_radio = gr.Radio(
            choices=list(EXPERT_FACTORIES.keys()),
            value=DEFAULT_MODULE,
            label="Modul",
            info="Du kannst jederzeit wechseln."
        )
        info = gr.Markdown(f"✅ **{DEFAULT_MODULE}** gewählt.")
        preload_btn = gr.Button("🔄 Modul vorladen (optional)")

        chat = gr.ChatInterface(
            fn=answer,
            additional_inputs=[module_radio],
            # type='tuples' → message: str, history: Liste[(user, assistant)]
            type="tuples",
            title="Chat zum gewählten Modul",
            textbox=gr.Textbox(placeholder="Stelle deine Frage zum gewählten Modul …"),
        )

        def on_module_change(sel):
            return gr.update(value=f"✅ **{sel}** gewählt.")
        module_radio.change(on_module_change, inputs=module_radio, outputs=info)

        def preload(sel):
            try:
                get_expert(sel)
                return gr.update(value=f"✅ **{sel}** geladen und bereit.")
            except Exception as e:
                return gr.update(value=f"❌ Fehler beim Laden von **{sel}**: {e}")
        preload_btn.click(preload, inputs=module_radio, outputs=info)

    return demo


if __name__ == "__main__":
    print("[UI] baue App …")
    app = build_app()
    print("[UI] starte Gradio …")
    app.launch(
        inbrowser=True,
        server_name="127.0.0.1",
        server_port=7860,   # bei Konflikt eine andere Zahl nehmen, z.B. 7861
        show_error=True,
    )
