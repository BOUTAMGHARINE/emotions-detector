import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")


# Analyse du dataset

# Valeurs manquantes
print(train.isnull().sum())

# Nombre d'images
print(train.shape[0], "images")

# Classes présentes
print("Les classes sont :", train["emotion"].unique())

# Nombre de pixels par image
pixels = np.array(
    train["pixels"].iloc[0].split(),
    dtype=np.float32
)

print("Nombre de pixels par image :", len(pixels))

# Distribution des émotions
print("Distribution des 7 émotions :")
print(train["emotion"].value_counts())

# Dimension de l'image
image = pixels.reshape(48, 48)

print("Dimension de l'image :", image.shape)


# Noms des émotions
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


# Visualisation
def show_images():

    plt.figure(figsize=(12, 5))

    # Image Disgust
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
    plt.savefig("../results/sample_image_disgust.png")
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
    plt.savefig("../results/sample_images.png")
    plt.close()


show_images()