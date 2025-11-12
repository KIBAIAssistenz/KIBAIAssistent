import os
import json
import sys
from datetime import datetime

# --- Dynamisch den Projektpfad hinzufügen ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# --- Jetzt funktionieren die Importe ---
from reward_model.reward_model import compute_reward_from_feedback
from rlhf_pipeline.rlhf_pipeline import update_model_parameters  # oder update_model_parameters je nach Benennung

# --- Feedback speichern ---
def collect_feedback(user_id, message, response, rating, comment=None):
    feedback = {
        "user_id": user_id,
        "message": message,
        "response": response,
        "rating": rating,
        "comment": comment,
        "timestamp": datetime.now().isoformat()
    }

    # relativer Pfad innerhalb des Projekts
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_dir, "feedback_log.json")

    with open(log_path, "a") as f:
        f.write(json.dumps(feedback) + "\n")

    return {"status": "success", "message": f"Feedback saved at {log_path}"}


