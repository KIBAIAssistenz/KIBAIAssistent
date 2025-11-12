import os
import sys
import json
import random
from datetime import datetime
 
# =========================================
# 1️⃣ Pfade setzen, damit Python Module findet
# =========================================
# Projekt-Root (eine Ebene über "feedback/")
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
 
# Jetzt funktionieren diese Importe:
from reward_model.reward_model import compute_reward_from_feedback
from rlhf_pipeline.rlhf_pipeline import train_agent
 
# =========================================
# 2️⃣ Feedback-Datei definieren
# =========================================
feedback_path = os.path.join(project_root, "feedback", "feedback_log.json")
 
# Beispiel-Daten
fake_users = [f"student_{i}" for i in range(1, 11)]
fake_questions = [
    "Was ist Machine Learning?",
    "Wie funktioniert ein neuronales Netz?",
    "Erkläre Reinforcement Learning.",
    "Was ist der Unterschied zwischen KI und ML?"
]
fake_responses = [
    "Machine Learning ist ein Teilgebiet der KI.",
    "Neuronale Netze bestehen aus Schichten, die Daten verarbeiten.",
    "RL lernt durch Belohnung und Bestrafung.",
    "KI ist der Oberbegriff, ML ist ein Teil davon."
]
 
# =========================================
# 3️⃣ Funktion: künstliches Feedback erzeugen
# =========================================
def simulate_feedback(num_entries=50, bias="mixed"):
    """
    bias = 'positive' (4–5 Sterne), 'negative' (1–2 Sterne) oder 'mixed' (1–5)
    """
    with open(feedback_path, "a") as f:
        for _ in range(num_entries):
            user = random.choice(fake_users)
            message = random.choice(fake_questions)
            response = random.choice(fake_responses)
 
            if bias == "positive":
                rating = random.randint(4, 5)
            elif bias == "negative":
                rating = random.randint(1, 2)
            else:
                rating = random.randint(1, 5)
 
            feedback = {
                "user_id": user,
                "message": message,
                "response": response,
                "rating": rating,
                "comment": f"Auto-feedback: {rating} Sterne",
                "timestamp": datetime.now().isoformat()
            }
            f.write(json.dumps(feedback) + "\n")
 
    print(f"✅ {num_entries} künstliche Feedbacks erzeugt (Bias='{bias}')")
 
# =========================================
# 4️⃣ Testlauf: RLHF-Loop simulieren
# =========================================
if __name__ == "__main__":
    print("🚀 RLHF-Testlauf gestartet...\n")
 
    # --- Negatives Feedback simulieren ---
    simulate_feedback(num_entries=20, bias="negative")
    avg_reward = compute_reward_from_feedback()
    print(f"📉 Nach negativem Feedback: Reward={avg_reward:.2f}")
    train_agent()
 
    # --- Positives Feedback simulieren ---
    simulate_feedback(num_entries=20, bias="positive")
    avg_reward = compute_reward_from_feedback()
    print(f"📈 Nach positivem Feedback: Reward={avg_reward:.2f}")
    train_agent()
 
    print("\n✅ RLHF-Testlauf abgeschlossen!")

