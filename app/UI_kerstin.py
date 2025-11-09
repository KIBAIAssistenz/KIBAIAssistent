import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from flask import Flask, render_template, request, jsonify
import markdown

from services.llm_connector import llm
from experts.einführung_KI.expert_einführung_KI import build_einführung_KI_expert
from experts.machine_learning.expert_ml import build_machine_learning_expert

app = Flask(__name__)

EXPERT_FACTORIES = {
    "Einführung in die KI": lambda: build_einführung_KI_expert(llm),
    "Machine Learning": lambda: build_machine_learning_expert(llm),
}
EXPERT_CACHE = {}

def get_expert(label):
    if label not in EXPERT_CACHE:
        EXPERT_CACHE[label] = EXPERT_FACTORIES[label]()
    return EXPERT_CACHE[label]

@app.route("/")
def index():
    return render_template("index.html", modules=list(EXPERT_FACTORIES.keys()))

@app.route("/ask", methods=["POST"])
def ask():
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

if __name__ == "__main__":
    app.run(debug=True)
