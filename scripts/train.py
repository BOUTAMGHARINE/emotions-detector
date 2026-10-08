import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from preprocess import train




X = np.array([
    np.array(pixels.split(), dtype=np.float32)
    for pixels in train["pixels"]
])

y = train["emotion"].values