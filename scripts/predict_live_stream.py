
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np
from tensorflow.keras.models import load_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "results/model/final_emotion_model.keras"
CAPTURE_DURATION_SECONDS = 20
MIN_PREPROCESSED_IMAGES = 20
MAX_CONSECUTIVE_READ_FAILURES = 3
OUTPUT_ROOT = PROJECT_ROOT / "results/preprocessing_test/webcam"

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


def main():
    model = load_model(MODEL_PATH)

    #  Load the Haar Cascade face detector
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        raise RuntimeError(
            "Unable to load the face detector."
        )

    capture_dir = OUTPUT_ROOT / datetime.now().strftime(
        "capture_%Y%m%d_%H%M%S_%f"
    )
    capture_dir.mkdir(parents=True, exist_ok=False)

    # 3. Open the webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Unable to open the webcam. "
            "Check that it is connected and access is permitted."
        )

    print("Webcam started.")
    print(
        f"Keep the webcam open for at least {CAPTURE_DURATION_SECONDS} "
        f"seconds and until {MIN_PREPROCESSED_IMAGES} face images are saved."
    )
    print("Press 'q' to stop the stream.")
    print(f"Preprocessed images will be saved in: {capture_dir}")

    last_terminal_output = 0.0
    saved_images = 0
    consecutive_read_failures = 0
    capture_started = time.monotonic()
    requirement_met = False
##
    try:
        while True:
            # 4. Capture a frame from the webcam
            ret, frame = camera.read()

            if not ret:
                consecutive_read_failures += 1
                print(
                    "Warning: unable to read a webcam frame "
                    f"({consecutive_read_failures}/"
                    f"{MAX_CONSECUTIVE_READ_FAILURES} consecutive failures).",
                    flush=True,
                )
                if consecutive_read_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    raise RuntimeError(
                        "Unable to read webcam frames after "
                        f"{MAX_CONSECUTIVE_READ_FAILURES} consecutive attempts."
                    )
                continue

            consecutive_read_failures = 0

            elapsed = time.monotonic() - capture_started

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
                face_image = cv2.resize(gray[y:y + h, x:x + w], (48, 48))

                if saved_images < MIN_PREPROCESSED_IMAGES:
                    image_name = f"face_{saved_images + 1:02d}.png"
                    image_path = capture_dir / image_name
                    if not cv2.imwrite(str(image_path), face_image):
                        raise OSError(
                            f"Unable to save preprocessed face image: "
                            f"{image_path}"
                        )
                    saved_images += 1

                # Normalize pixel values to the range 0-1
                face = face_image.astype(np.float32) / 255.0

                # Expected shape: (1, 48, 48, 1)
                face = np.expand_dims(face, axis=-1)
                face = np.expand_dims(face, axis=0)

                # 8. Predict probabilities
                probabilities = model(face, training=False).numpy()[0]

                emotion_id = int(np.argmax(probabilities))
                confidence = float(probabilities[emotion_id])
                emotion = EMOTION_NAMES[emotion_id]

                current_time = time.monotonic()
                if current_time - last_terminal_output >= 1:
                    print("Preprocessing .......")
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

            # 10. Display the video stream and capture progress
            remaining_seconds = max(
                0, CAPTURE_DURATION_SECONDS - int(elapsed)
            )
            status = (
                f"Saved faces: {saved_images}/{MIN_PREPROCESSED_IMAGES} | "
                f"Minimum time remaining: {remaining_seconds}s"
            )
            cv2.putText(
                frame,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Emotion Detector - Webcam capture", frame)

            elapsed = time.monotonic() - capture_started
            if cv2.waitKey(1) & 0xFF == ord("q"):
                requirement_met = (
                    elapsed >= CAPTURE_DURATION_SECONDS
                    and saved_images >= MIN_PREPROCESSED_IMAGES
                )
                if not requirement_met:
                    print(
                        "Capture stopped before the audit requirement was met: "
                        f"{elapsed:.0f}/{CAPTURE_DURATION_SECONDS} seconds, "
                        f"{saved_images}/{MIN_PREPROCESSED_IMAGES} face images."
                    )
                break

    finally:
        # 11. Release resources, even if an error occurs
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")

    if not requirement_met:
        raise RuntimeError(
            "Webcam capture did not meet the audit requirement: keep it open "
            f"for at least {CAPTURE_DURATION_SECONDS} seconds and save at "
            f"least {MIN_PREPROCESSED_IMAGES} face images."
        )

    print(
        f"Audit requirement met: captured for at least "
        f"{CAPTURE_DURATION_SECONDS} seconds and saved {saved_images} "
        f"preprocessed images in {capture_dir}."
    )


if __name__ == "__main__":
    main()
