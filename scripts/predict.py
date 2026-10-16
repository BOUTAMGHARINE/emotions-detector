
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


def predict():
    # 1. Load the test data
    test = pd.read_csv(TEST_PATH)

  
    # 2. Convert the pixels into 48x48 images
    X_test = np.array([
        np.array(row.split(), dtype=np.float32).reshape(48, 48)
        for row in test["pixels"]
    ])

    X_test /= 255.0

    X_test = np.expand_dims(X_test, axis=-1)

    y_test = test["emotion"].to_numpy()

    model = load_model(MODEL_PATH)

    y_proba = model.predict(X_test, verbose=0)

    y_pred = np.argmax(y_proba, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba, labels=np.arange(7))

    print(f"Number of test images: {len(X_test)}")
    print(f"Test accuracy : {accuracy:.2%}")
    print(f"Test loss : {loss:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=np.arange(7),
        target_names=EMOTION_NAMES,
        zero_division=0
    ))

    


if __name__ == "__main__":
    predict()
