import pandas as pd 
import numpy as np 





train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")




print(train.isnull().sum())
print(train.shape[0]," images,")
print("les classe sont :",train["emotion"].unique())
print("nombres des pixels par image, ",len(np.array(train["pixels"].iloc[0].split(), dtype=np.float32)))
print("destrubution de 7 emotion",train["emotion"].value_counts())
print("demention de l image:",len(train["pixels"].iloc[0]))

pixels = np.array(train["pixels"].iloc[0].split(), dtype=np.float32)

image = pixels.reshape(48, 48)

print("Dimension de l'image :", image.shape)





