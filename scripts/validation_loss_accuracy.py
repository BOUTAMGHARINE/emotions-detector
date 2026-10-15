
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, log_loss, classification_report


MODEL_PATH = "../results/model/final_emotion_model.keras"
TEST_PATH = "../data/test_with_emotions.csv"

EMOTION_NAMES = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


def evaluate_model():
    # 1. Charger les données de test
    test = pd.read_csv(TEST_PATH)

    # Vérifier les colonnes nécessaires
    required_columns = {"pixels", "emotion"}
    if not required_columns.issubset(test.columns):
        raise ValueError(
            "Le fichier de test doit contenir les colonnes "
            "'pixels' et 'emotion'."
        )

    # 2. Convertir les pixels en images 48x48
    X_test = np.array([
        np.array(row.split(), dtype=np.float32).reshape(48, 48)
        for row in test["pixels"]
    ])

    # 3. Normaliser les pixels entre 0 et 1
    X_test /= 255.0

    # 4. Ajouter le canal des images en niveaux de gris
    X_test = np.expand_dims(X_test, axis=-1)

    # 5. Récupérer les vraies émotions
    y_test = test["emotion"].to_numpy()

    # 6. Charger le modèle
    model = load_model(MODEL_PATH)

    # 7. Prédire les probabilités
    y_proba = model.predict(X_test, verbose=0)

    # 8. Obtenir les classes prédites
    y_pred = np.argmax(y_proba, axis=1)

    # 9. Calculer les métriques
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba, labels=np.arange(7))

    print(f"Nombre d'images testées : {len(X_test)}")
    print(f"Test accuracy : {accuracy:.2%}")
    print(f"Test loss : {loss:.4f}")

    print("\nRapport de classification :")
    print(classification_report(
        y_test,
        y_pred,
        labels=np.arange(7),
        target_names=EMOTION_NAMES,
        zero_division=0
    ))

    if accuracy > 0.60:
        print("Objectif atteint : accuracy supérieure à 60 %.")
    else:
        print("Objectif non atteint : accuracy inférieure ou égale à 60 %.")


if __name__ == "__main__":
    evaluate_model()
