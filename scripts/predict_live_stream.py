
from datetime import datetime
import time

import cv2
import numpy as np
from tensorflow.keras.models import load_model


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


# Path to the trained model
MODEL_PATH = "../results/model/final_emotion_model.keras"


def main():
    # 1. Load the model once
    print("Loading model...")
    model = load_model(MODEL_PATH)
    print("Model loaded.")

    # 2. Load the Haar Cascade face detector
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        raise RuntimeError(
            "Unable to load the face detector."
        )

    # 3. Open the webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Unable to open the webcam. "
            "Check that it is connected and access is permitted."
        )

    print("Webcam started.")
    print("Press 'q' to quit.")

    last_terminal_output = 0.0

    try:
        while True:
            # 4. Capture a frame from the webcam
            ret, frame = camera.read()

            if not ret:
                print("Unable to read a frame from the webcam.")
                break

            # 5. Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 6. Detect faces
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )

            # 7. Process each detected face
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]

                # Resize to match the model input
                face = cv2.resize(face, (48, 48))

                # Normalize pixel values to the range 0-1
                face = face.astype(np.float32) / 255.0

                # Expected shape: (1, 48, 48, 1)
                face = np.expand_dims(face, axis=-1)
                face = np.expand_dims(face, axis=0)

                # 8. Predict probabilities
                probabilities = model.predict(
                    face,
                    verbose=0,
                )[0]

                emotion_id = int(np.argmax(probabilities))
                confidence = float(probabilities[emotion_id])
                emotion = EMOTION_NAMES[emotion_id]

                current_time = time.monotonic()
                if current_time - last_terminal_output >= 1:
                    print("Preprocessing ...")
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(
                        f"{timestamp}s : {emotion} , {confidence:.0%}",
                        flush=True,
                    )
                    last_terminal_output = current_time

                # 9. Draw a rectangle around the face
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2,
                )

                # Display the emotion and confidence
                label = f"{emotion}: {confidence:.0%}"

                text_y = max(y - 10, 25)

                cv2.putText(
                    frame,
                    label,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # 10. Display the video stream
            cv2.imshow("Emotion Detector - Press q to quit", frame)

            # Quit when the q key is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # 11. Release resources, even if an error occurs
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")


if __name__ == "__main__":
    main()
