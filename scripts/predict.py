
import numpy as np
from tensorflow.keras.models import load_model

from preprocess import preprocess_video


# Mapping between classes and emotions
EMOTION_NAMES = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}


# File paths
VIDEO_PATH = (
    "../results/preprocessing_test/"
    "4769799-hd_1080_1920_25fps.mp4"
)

MODEL_PATH = "../results/model/final_emotion_model.keras"


def predict_video():
    """Detect faces in a video and predict their emotions."""

    # 1. Preprocess the video and extract faces
    print("Preprocessing video...")

    faces = preprocess_video(VIDEO_PATH)

    if len(faces) == 0:
        print("No faces detected in the video.")
        return

    print(f"Number of faces extracted: {len(faces)}")

    # 2. Load the trained model once
    print("Loading model...")

    model = load_model(MODEL_PATH)

    # 3. Predict the probabilities for the seven emotions
    print("Predicting emotions...")

    predictions = model.predict(faces, verbose=0)

    # 4. Display the results
    print("\nPrediction results:")

    for i, probabilities in enumerate(predictions):
        emotion_id = int(np.argmax(probabilities))
        confidence = float(probabilities[emotion_id])

        emotion = EMOTION_NAMES[emotion_id]

        print(
            f"Visage {i + 1}: "
            f"{emotion} "
            f"(confidence: {confidence:.2%})"
        )


if __name__ == "__main__":
    predict_video()
