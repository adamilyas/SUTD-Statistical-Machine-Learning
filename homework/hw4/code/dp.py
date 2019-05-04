import numpy as np
from sklearn import decomposition, datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

data = datasets.load_diabetes().data
U, S, Vt = np.linalg.svd(scale(data))
print("Singular val.")
print(S)
print("V matrix")
print(Vt.T)

print("Most impt 3 components of first 10 print")
pca = PCA(n_components=3)
transformed_data = pca.fit_transform(scale(data))
print(transformed_data[:10])
