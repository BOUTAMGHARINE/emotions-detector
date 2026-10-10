import pandas as pd
import numpy as np
from preprocess import preprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers,models
from sklearn.metrics import classification_report, confusion_matrix









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
        
        layers.Input(shape=(48,48,1)),
        
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(150, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(200, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),     
        
        layers.Flatten(),
        layers.Dense(128, activation="relu"),    
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




history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=64
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

