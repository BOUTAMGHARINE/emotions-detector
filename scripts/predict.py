import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, log_loss

# Charger le modèle entraîné
model = load_model("best_model.keras")
# Charger les données de test avec les labels réels
test = pd.read_csv("../data/test_with_emotions.csv")

# Préparer les images
X_test = np.array([
    np.array(row.split(), dtype=np.float32).reshape(48, 48)
    for row in test["pixels"]
])

# Normaliser les pixels : [0, 255] -> [0, 1]
X_test = X_test / 255.0

# Ajouter la dimension du canal : (N, 48, 48, 1)
X_test = np.expand_dims(X_test, axis=-1)

# Récupérer les émotions réelles
y_test = test["emotion"].to_numpy()

# Prédire les probabilités pour les 7 émotions
y_proba = model.predict(X_test)

# Obtenir la classe prédite
y_pred = np.argmax(y_proba, axis=1)

# Calculer les métriques
accuracy = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, y_proba, labels=np.arange(7))

print(f"Test accuracy: {accuracy * 100:.2f}%")
print(f"Test loss: {loss:.4f}")

if accuracy > 0.60:
    print("Objectif atteint : accuracy supérieure à 60 %.")
else:
    print("Objectif non atteint : accuracy inférieure ou égale à 60 %.")
