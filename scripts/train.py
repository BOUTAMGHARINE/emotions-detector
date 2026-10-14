import pandas as pd
import numpy as np
from preprocess import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers,models
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras










train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")



images_train, images_test = preprocessing(train, test)
X = images_train
y = train["emotion"].to_numpy()


X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=train["emotion"]
)





def build_model():
    model = models.Sequential([
        # Bloc 1
        layers.Conv2D(
            32, (3, 3),
            activation="relu",
            padding="same",
            input_shape=(48, 48, 1)
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloc 2
        layers.Conv2D(
            64, (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Bloc 3
        layers.Conv2D(
            128, (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.30),

        # Bloc 4
        layers.Conv2D(
            256, (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.30),

        # Bloc 5
        layers.Conv2D(
            512, (3, 3),
            activation="relu",
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.Dropout(0.30),

        # Classification
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(7, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


model = build_model()
model.summary()


callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor="val_loss",
        save_best_only=True
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
]



history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=64,
    callbacks=callbacks
)



val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=1)

print(f"Validation loss : {val_loss:.4f}")
print(f"Validation accuracy : {val_accuracy:.2%}")





# Prédictions sur les images de validation
y_pred_prob = model.predict(X_val)
y_pred = np.argmax(y_pred_prob, axis=1)


emotion_names = [
    "Angry", "Disgust", "Fear", "Happy",
    "Sad", "Surprise", "Neutral"
]

print(classification_report(
    y_val,
    y_pred,
    labels=list(range(7)),
    target_names=emotion_names,
    zero_division=0
))

print("Matrice de confusion :")
print(confusion_matrix(y_val, y_pred))




