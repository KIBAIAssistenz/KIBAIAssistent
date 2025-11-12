import json
import os
import numpy as np

def compute_reward_from_feedback(feedback_path=None):
    """
    Liest alle Feedback-Bewertungen (1–5 Sterne) aus und berechnet
    den durchschnittlichen Reward zwischen 0 und 1.
    """

    # Standardpfad zur Feedback-Datei
    if feedback_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        feedback_path = os.path.join(base_dir, "feedback", "feedback_log.json")

    rewards = []
    if not os.path.exists(feedback_path):
        print("⚠️ Keine Feedback-Datei gefunden.")
        return 0

    with open(feedback_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line)
                rating = int(entry["rating"])
                # Sicherheitscheck: Nur Werte 1–5 zulassen
                if 1 <= rating <= 5:
                    rewards.append(rating)
            except Exception:
                continue

    if not rewards:
        print("⚠️ Keine gültigen Feedback-Einträge gefunden.")
        return 0

    avg_rating = np.mean(rewards)
    # Skaliere auf 0–1 (optional, z. B. für RLHF-Berechnungen)
    normalized_reward = (avg_rating - 1) / 4

    print(f"📊 Durchschnittliche Bewertung: {avg_rating:.2f} Sterne")
    print(f"🎯 Normalisierter Reward: {normalized_reward:.2f}")

    return normalized_reward
