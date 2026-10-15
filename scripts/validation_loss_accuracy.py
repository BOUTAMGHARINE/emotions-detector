
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
    # 1. Load the test data
    test = pd.read_csv(TEST_PATH)

    # Check for the required columns
    required_columns = {"pixels", "emotion"}
    if not required_columns.issubset(test.columns):
        raise ValueError(
            "The test file must contain the columns "
            "'pixels' and 'emotion'."
        )

    # 2. Convert the pixels into 48x48 images
    X_test = np.array([
        np.array(row.split(), dtype=np.float32).reshape(48, 48)
        for row in test["pixels"]
    ])

    # 3. Normalize pixel values to the range 0-1
    X_test /= 255.0

    # 4. Add the channel dimension for grayscale images
    X_test = np.expand_dims(X_test, axis=-1)

    # 5. Get the ground-truth emotions
    y_test = test["emotion"].to_numpy()

    # 6. Load the model
    model = load_model(MODEL_PATH)

    # 7. Predict probabilities
    y_proba = model.predict(X_test, verbose=0)

    # 8. Get the predicted classes
    y_pred = np.argmax(y_proba, axis=1)

    # 9. Calculate the metrics
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

    if accuracy > 0.60:
        print("Goal met: accuracy is above 60%.")
    else:
        print("Goal not met: accuracy is 60% or lower.")


if __name__ == "__main__":
    evaluate_model()
