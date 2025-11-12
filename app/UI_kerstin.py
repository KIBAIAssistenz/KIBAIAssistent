
# import sys, pathlib
# sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# from flask import Flask, render_template, request, jsonify
# import markdown
# from feedback.feedback_collector import collect_feedback

# from services.llm_connector import llm
# from experts.einführung_KI.expert_einführung_KI import build_einführung_KI_expert
# from experts.machine_learning.expert_ml import build_machine_learning_expert
# from feedback.feedback_collector import collect_feedback
# from reward_model.reward_model import compute_reward_from_feedback
# from rlhf_pipeline.trainer import train_agent

# app = Flask(__name__)

# # ==========================
# # Experten-Definition
# # ==========================
# EXPERT_FACTORIES = {
#     "Einführung in die KI": lambda: build_einführung_KI_expert(llm),
#     "Machine Learning": lambda: build_machine_learning_expert(llm),
# }
# EXPERT_CACHE = {}

# def get_expert(label):
#     if label not in EXPERT_CACHE:
#         EXPERT_CACHE[label] = EXPERT_FACTORIES[label]()
#     return EXPERT_CACHE[label]


# # ==========================
# # ROUTE: Startseite
# # ==========================
# @app.route("/")
# def index():
#     return render_template("index.html", modules=list(EXPERT_FACTORIES.keys()))


# # ==========================
# # ROUTE: Frage an Chatbot
# # ==========================
# @app.route("/ask", methods=["POST"])
# def ask():
#     data = request.json
#     message = data["message"]
#     module = data["module"]
#     print(f"[Flask] Anfrage erhalten: Modul = {module}, Frage = {message}")

#     try:
#         expert = get_expert(module)
#         response = expert["chain"].invoke(message)

#         # Markdown → HTML
#         html_response = markdown.markdown(
#             response,
#             extensions=["tables", "fenced_code", "nl2br", "sane_lists"]
#         )

#         return jsonify({"response": html_response})
#     except Exception as e:
#         print(f"[Flask] Fehler: {e}")
#         return jsonify({"response": f"⚠️ Fehler: {str(e)}"})


# # ==========================
# # ROUTE: Feedback speichern
# # ==========================
# @app.route("/feedback", methods=["POST"])
# def feedback():
#     """Nimmt Sternbewertungen aus der UI entgegen und speichert sie"""
#     data = request.json
#     user_id = data.get("user_id", "anonymous")
#     message = data.get("message", "")
#     response = data.get("response", "")
#     rating = int(data.get("rating", 0))
#     comment = data.get("comment", "")

#     result = collect_feedback(user_id, message, response, rating, comment)
#     print(f"[Flask] Feedback erhalten: {result}")
#     return jsonify(result)


# if __name__ == "__main__":
#     app.run(debug=True)

# @app.route("/feedback", methods=["POST"])
# def feedback():
#     data = request.json
#     collect_feedback(**data)
#     avg_reward = compute_reward_from_feedback()
#     update_model_parameters(avg_reward)
#     return jsonify({"status": "updated"})

# if __name__ == "__main__":
#     app.run(debug=True)


import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flask import Flask, render_template, request, jsonify
import markdown

# Eigene Module
from feedback.feedback_collector import collect_feedback
from reward_model.reward_model import compute_reward_from_feedback
from rlhf_pipeline.rlhf_pipeline import train_agent

# Experten & LLM
from services.llm_connector import llm
from experts.einführung_KI.expert_einführung_KI import build_einführung_KI_expert
from experts.machine_learning.expert_ml import build_machine_learning_expert


app = Flask(__name__, template_folder="templates", static_folder="static")

# ==========================
# Experten-Definition
# ==========================
EXPERT_FACTORIES = {
    "Einführung in die KI": lambda: build_einführung_KI_expert(llm),
    "Machine Learning": lambda: build_machine_learning_expert(llm),
}
EXPERT_CACHE = {}

def get_expert(label):
    if label not in EXPERT_CACHE:
        EXPERT_CACHE[label] = EXPERT_FACTORIES[label]()
    return EXPERT_CACHE[label]


# ==========================
# ROUTE: Startseite
# ==========================
@app.route("/")
def index():
    """Lädt die Haupt-UI."""
    return render_template("index.html", modules=list(EXPERT_FACTORIES.keys()))


# ==========================
# ROUTE: Frage an Chatbot
# ==========================
@app.route("/ask", methods=["POST"])
def ask():
    """Empfängt Nutzerfragen und gibt KI-Antworten zurück."""
    data = request.json
    message = data["message"]
    module = data["module"]

    print(f"[Flask] Anfrage erhalten: Modul = {module}, Frage = {message}")

    try:
        expert = get_expert(module)
        response = expert["chain"].invoke(message)

        # Markdown → HTML
        html_response = markdown.markdown(
            response,
            extensions=["tables", "fenced_code", "nl2br", "sane_lists"]
        )

        return jsonify({"response": html_response})
    except Exception as e:
        print(f"[Flask] Fehler: {e}")
        return jsonify({"response": f"⚠️ Fehler: {str(e)}"})


# ==========================
# ROUTE: Feedback + Echtzeit-Lernen
# ==========================
@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Speichert Nutzerfeedback (1–5 Sterne) UND führt sofort RLHF-Update aus.
    """
    data = request.json or {}
    user_id = data.get("user_id", "anonymous")
    message = data.get("message", "")
    response = data.get("response", "")
    rating = int(data.get("rating", 0))
    comment = data.get("comment", "")

    # Schritt 1: Feedback speichern
    result = collect_feedback(user_id, message, response, rating, comment)
    print(f"[Flask] Feedback gespeichert: {result}")

    # Schritt 2: Reward neu berechnen
    avg_reward = compute_reward_from_feedback()
    print(f"[Flask] Durchschnittlicher Reward: {avg_reward:.2f}")

    # Schritt 3: Policy sofort aktualisieren (Echtzeit-Lernen)
    train_agent()
    print("[Flask] Policy-Model in Echtzeit aktualisiert ✅")

    return jsonify({
        "status": "updated",
        "avg_reward": avg_reward,
        "message": "Feedback verarbeitet und Modell in Echtzeit verbessert."
    })


# ==========================
# APP STARTEN
# ==========================
if __name__ == "__main__":
    print("🚀 FHNW Lernassistent läuft auf http://127.0.0.1:5000")
    app.run(debug=True)
