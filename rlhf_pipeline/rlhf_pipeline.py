import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import os
from reward_model.reward_model import compute_reward_from_feedback

def load_policy_model():
    """
    Lädt das aktuelle Modell (Simulation: Parameter in JSON-Datei)
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "policy_model.json")

    # Falls kein Modell vorhanden ist, Standardwerte setzen
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        default_model = {"learning_rate": 0.001, "reward_history": []}
        with open(model_path, "w") as f:
            json.dump(default_model, f, indent=2)
        return default_model, model_path

    with open(model_path, "r") as f:
        model = json.load(f)
    return model, model_path


def update_model_parameters(model, reward):
    """
    Simuliert eine Policy-Anpassung basierend auf dem Reward.
    - Hoher Reward → Lernrate leicht verringern (Modell stabilisiert sich)
    - Niedriger Reward → Lernrate leicht erhöhen (mehr Anpassung nötig)
    """
    if reward > 0.7:
        model["learning_rate"] *= 0.95  # positives Feedback
    elif reward < 0.4:
        model["learning_rate"] *= 1.05  # negatives Feedback

    model["reward_history"].append(reward)
    return model


def train_agent():
    """
    Führt den vollständigen RLHF-Schritt aus:
    1. Reward berechnen
    2. Modell laden
    3. Modell anpassen
    4. Speichern
    """
    print("🚀 Starte RLHF-Training...")

    # Schritt 1: Reward berechnen
    reward = compute_reward_from_feedback()

    # Schritt 2: Modell laden
    model, model_path = load_policy_model()

    # Schritt 3: Modell anpassen
    model = update_model_parameters(model, reward)

    # Schritt 4: Neues Modell speichern
    with open(model_path, "w") as f:
        json.dump(model, f, indent=2)

    print(f"✅ Training abgeschlossen. Neues Modell gespeichert unter:\n{model_path}")
    print(f"📈 Aktuelle Lernrate: {model['learning_rate']:.6f}")
    print(f"📊 Letzter Reward: {reward:.2f}")


if __name__ == "__main__":
    train_agent()
