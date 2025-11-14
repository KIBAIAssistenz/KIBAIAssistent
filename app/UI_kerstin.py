import os
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    redirect,
    url_for,
)
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

# 🔥 WICHTIG: macOS / Safari / Chrome blockieren Sessions bei schlechtem key
app.secret_key = os.urandom(32)   # <-- Fix Session NICHT löschen!


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
# LOGIN / LOGOUT
# ==========================
@app.route("/login", methods=["GET", "POST"])
def login():
    print("LOGIN-PAGE AUFGERUFEN")  # Debug

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        print("LOGIN VERSUCH:", email, password)  # Debug

        # --- DEMO-LOGIN (später mit DB ersetzen) ---
        if email == "student@fhnw.ch" and password == "1234":
            session["user"] = email
            print("LOGIN ERFOLGREICH — SESSION GEFÜLLT!")  # Debug
            return redirect(url_for("index"))
        else:
            print("LOGIN FEHLGESCHLAGEN")  # Debug
            return render_template("login.html", error="Falsche Login-Daten!")

    return render_template("login.html")


@app.route("/logout")
def logout():
    print("LOGOUT — SESSION GELÖSCHT")  # Debug
    session.pop("user", None)
    return redirect(url_for("login"))


# ==========================
# ROUTE: Startseite (geschützt)
# ==========================
@app.route("/")
def index():
    print("INDEX AUFGERUFEN — SESSION:", dict(session))  # Debug

    # Wenn nicht eingeloggt → zuerst Login-Seite
    if "user" not in session:
        print("NICHT EINGELOGGT — REDIRECT /login")  # Debug
        return redirect(url_for("login"))

    session["history"] = []
    return render_template("index.html", modules=list(EXPERT_FACTORIES.keys()))


# ==========================
# ROUTE: Frage an Chatbot
# ==========================
@app.route("/ask", methods=["POST"])
def ask():
    if "user" not in session:
        print("ASK BLOCKIERT — NICHT EINGELOGGT")  # Debug
        return jsonify({"response": "⛔ Bitte zuerst einloggen."})

    data = request.json
    message = data["message"]
    module = data["module"]

    print(f"[Flask] Anfrage erhalten: Modul = {module}, Frage = {message}")
    history = session.get("history", [])

    try:
        expert = get_expert(module)
        response = expert["chain"].invoke(
            {
                "question": message,
                "history": history,
            }
        )

        # Verlauf updaten
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        session["history"] = history

        # Markdown → HTML
        html_response = markdown.markdown(
            response,
            extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
        )
        return jsonify({"response": html_response})

    except Exception as e:
        print(f"[Flask] Fehler: {e}")
        return jsonify({"response": f"⚠️ Fehler: {str(e)}"})


# ==========================
# ROUTE: Feedback + RLHF
# ==========================
@app.route("/feedback", methods=["POST"])
def feedback():
    if "user" not in session:
        return jsonify({"status": "error", "message": "Nicht eingeloggt."})

    data = request.json or {}
    user_id = session.get("user", "anonymous")
    message = data.get("message", "")
    response = data.get("response", "")
    rating = int(data.get("rating", 0))
    comment = data.get("comment", "")

    result = collect_feedback(user_id, message, response, rating, comment)
    print(f"[Flask] Feedback gespeichert: {result}")

    avg_reward = compute_reward_from_feedback()
    print(f"[Flask] Durchschnittlicher Reward: {avg_reward:.2f}")

    train_agent()
    print("[Flask] Policy-Model in Echtzeit aktualisiert ✅")

    return jsonify(
        {
            "status": "updated",
            "avg_reward": avg_reward,
            "message": "Feedback verarbeitet und Modell in Echtzeit verbessert.",
        }
    )


# ==========================
# APP STARTEN
# ==========================
if __name__ == "__main__":
    print("🚀 FHNW Lernassistent läuft auf http://127.0.0.1:5000")
    app.run(debug=True)
