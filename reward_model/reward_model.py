import json

import os

import numpy as np

import textstat  # 🔹 für Lesbarkeits-Analyse (Flesch-Reading-Ease)
 
def compute_reward_from_feedback(feedback_path=None):

    """

    Liest Feedback-Einträge (1–5 Sterne) aus, berechnet

    einen kombinierten Reward:

      ➤ 70 % basierend auf Nutzerbewertung

      ➤ 30 % basierend auf Verständlichkeit (Flesch Reading Ease Score)

    Gibt den normalisierten Durchschnitts-Reward (0–1) zurück.

    """
 
    # 🔹 Standardpfad zur Feedback-Datei

    if feedback_path is None:

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        feedback_path = os.path.join(base_dir, "feedback", "feedback_log.json")
 
    if not os.path.exists(feedback_path):

        print("⚠️ Keine Feedback-Datei gefunden.")

        return 0
 
    rewards = []

    with open(feedback_path, "r") as f:

        for line in f:

            try:

                entry = json.loads(line)

                rating = int(entry.get("rating", 0))

                response = entry.get("response", "")
 
                # 🔹 Nur Werte 1–5 zulassen

                if 1 <= rating <= 5:

                    # --- Verständlichkeits-Score ---

                    try:

                        simplicity_score = textstat.flesch_reading_ease(response)

                        # Begrenze und normalisiere auf 0–1

                        simplicity_norm = min(max(simplicity_score / 100, 0), 1)

                    except Exception:

                        simplicity_norm = 0.5  # Fallback, falls Textanalyse fehlschlägt
 
                    # --- Kombinierter Reward ---

                    user_reward = (rating - 1) / 4  # 1–5 → 0–1

                    combined_reward = (0.7 * user_reward) + (0.3 * simplicity_norm)

                    rewards.append(combined_reward)

            except Exception:

                continue
 
    if not rewards:

        print("⚠️ Keine gültigen Feedback-Einträge gefunden.")

        return 0
 
    avg_reward = np.mean(rewards)

    avg_rating = np.mean([(r * 4) + 1 for r in [(r - 0.3 * (r / 0.7)) for r in rewards]]) if rewards else 0
 
    print(f"📊 Durchschnittlicher kombinierter Reward (inkl. Verständlichkeit): {avg_reward:.2f}")

    print(f"🎯 Normalisierter Reward: {avg_reward:.2f}")

    return avg_reward
 
 
# ============================

# 🧪 Testlauf (optional)

# ============================

if __name__ == "__main__":

    print("🚀 Teste Reward-Berechnung...\n")

    reward = compute_reward_from_feedback()

    print(f"\n✅ Berechneter Reward: {reward:.3f}")

 