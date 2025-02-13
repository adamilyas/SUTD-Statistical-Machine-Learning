import numpy as np
from sklearn import decomposition, datasets
import matplotlib.pyplot as plt
import pandas as pd

def StandardScaler(X):
    """
    X: 2d array
    Returns
        X with each columns scaled so that column mean=0 and column std=1
    """
    N, d = X.shape
    scaled_data = np.zeros((N, d))
    for col_index in range(10):
        current_column = X[:, col_index]
        col_mean = np.mean(current_column)
        col_std = np.std(current_column)
        scaled_data[:, col_index] = [(el - col_mean)/col_std for el in current_column]
    return scaled_data

def pca_transform(X, n_components=2, show=False):
    """
    X: 2d array
    Perform SVD on X to obtain u, s, vh
    Returns
        Most important n_components
    """
    X_scaled = StandardScaler(X)
    u, s, vh = np.linalg.svd(X_scaled)
    if show:
        print("Matrix V: ")
        print(vh.T)
        print('Singular Values:')
        print(s)
    XV = X_scaled@vh.T
    return XV[:, :n_components]

if __name__ == "__main__":
    X = datasets.load_diabetes().data
    X_pca = pca_transform(X, n_components=3, show=True)
    print("3 most important components of first 10 data points:")
    print(X_pca[:10])
