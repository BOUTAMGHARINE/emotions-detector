import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test_with_emotions.csv")






# Dataset analysis

# Missing values
print(train.isnull().sum())

# Number of images
print(train.shape[0], "images")

# Available classes
print("The classes are:", train["emotion"].unique())

# Number of pixels per image
pixels = np.array(
    train["pixels"].iloc[0].split(),
    dtype=np.float32
)

print("Number of pixels per image:", len(pixels))

# Emotion distribution
print("Distribution of the 7 emotions:")
print(train["emotion"].value_counts())

# Image dimensions
image = pixels.reshape(48, 48)

print("Image dimensions:", image.shape)


# Emotion names
emotion_names = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

train["emotion_name"] = train["emotion"].map(emotion_names)


def convert_images_to_numeric(df) :
    images_numerics = []
    for i in range(len(df)) :
        pixels = np.array(
            df["pixels"].iloc[i].split(),dtype=np.float32
        )
        images_numerics.append(pixels.reshape(48,48))
    
    
    return images_numerics


# Visualisation
def show_images():

    plt.figure(figsize=(12, 5))

    # Disgust image
    disgust_image = train[
        train["emotion"] == 1
    ]["pixels"].iloc[0]

    disgust_image = np.array(
        disgust_image.split(),
        dtype=np.float32
    ).reshape(48, 48)

    plt.imshow(disgust_image, cmap="gray")
    plt.title("Emotion: Disgust")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("./sample_image_disgust.png")
    plt.close()


    # 20 images
    plt.figure(figsize=(12, 8))

    for i in range(20):

        pixels = np.array(
            train["pixels"].iloc[i].split(),
            dtype=np.float32
        )

        image = pixels.reshape(48, 48)

        plt.subplot(4, 5, i + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"Emotion: {train['emotion_name'].iloc[i]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("./sample_images.png")
    plt.close()



def preprocessing(train, test):
    images_train = convert_images_to_numeric(train)
    images_test = convert_images_to_numeric(test)

    images_train = np.array(images_train, dtype=np.float32)
    images_test = np.array(images_test, dtype=np.float32)

    images_train = np.expand_dims(images_train, axis=-1)
    images_test = np.expand_dims(images_test, axis=-1)

    images_train /= 255.0
    images_test /= 255.0

    return images_train, images_test




show_images()



def preprocess_video(video_path):
    output_dir = "../results/preprocessing_test"
    os.makedirs(output_dir, exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    # OpenCV's Haar Cascade face detector
    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )
    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        cap.release()
        raise RuntimeError("Unable to load the face detector.")

    processed_faces = []
    frame_index = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Process every 10th frame initially
        if frame_index % 10 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                # Crop the face
                face = gray[y:y + h, x:x + w]

                # Resize to 48 x 48
                face = cv2.resize(face, (48, 48))

                # Normalize pixel values to the range 0-1
                face = face.astype(np.float32) / 255.0

                # Add the channel dimension for the CNN
                face = np.expand_dims(face, axis=-1)

                processed_faces.append(face)

                # Save a sample image for verification
                image_to_save = (face[:, :, 0] * 255).astype(np.uint8)
                image_path = os.path.join(
                    output_dir,
                    f"image{saved_count}.png"
                )
                cv2.imwrite(image_path, image_to_save)
                saved_count += 1

        frame_index += 1

    cap.release()

    print(f"Frames processed: {frame_index}")
    print(f"Faces preprocessed: {saved_count}")

    if not processed_faces:
        print("No faces detected in the video.")
        return np.empty((0, 48, 48, 1), dtype=np.float32)

    return np.array(processed_faces, dtype=np.float32)

    
    
    
    
    
                    
                
                

        
    
preprocess_video(
    "../results/preprocessing_test/4769799-hd_1080_1920_25fps.mp4"
)
    
    
