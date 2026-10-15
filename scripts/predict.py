
import numpy as np
from tensorflow.keras.models import load_model

from preprocess import preprocess_video


# Correspondance entre les classes et les émotions
EMOTION_NAMES = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}


# Chemins des fichiers
VIDEO_PATH = (
    "../results/preprocessing_test/"
    "4769799-hd_1080_1920_25fps.mp4"
)

MODEL_PATH = "../results/model/final_emotion_model.keras"


def predict_video():
    """Détecter les visages dans une vidéo et prédire leurs émotions."""

    # 1. Prétraiter la vidéo et extraire les visages
    print("Prétraitement de la vidéo...")

    faces = preprocess_video(VIDEO_PATH)

    if len(faces) == 0:
        print("Aucun visage détecté dans la vidéo.")
        return

    print(f"Nombre de visages extraits : {len(faces)}")

    # 2. Charger le modèle entraîné une seule fois
    print("Chargement du modèle...")

    model = load_model(MODEL_PATH)

    # 3. Prédire les probabilités des sept émotions
    print("Prédiction des émotions...")

    predictions = model.predict(faces, verbose=0)

    # 4. Afficher les résultats
    print("\nRésultats des prédictions :")

    for i, probabilities in enumerate(predictions):
        emotion_id = int(np.argmax(probabilities))
        confidence = float(probabilities[emotion_id])

        emotion = EMOTION_NAMES[emotion_id]

        print(
            f"Visage {i + 1}: "
            f"{emotion} "
            f"(confiance : {confidence:.2%})"
        )


if __name__ == "__main__":
    predict_video()
