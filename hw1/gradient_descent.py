import os
import matplotlib.pyplot as plt

import torch
from torch import tensor
import pandas as pd
import numpy as np

def load_data():
    link = "https://www.dropbox.com/s/0rjqoaygjbk3sp8/bostonhouseprices3features.txt?dl=1"
    data = np.genfromtxt(link, delimiter=',', skip_header=1)
    # Convert input and target to tensors
    inputs = data[:, [0,1,2]]
    inputs = inputs.astype(np.float32)
    inputs = torch.from_numpy(inputs)

    target = data[:, 3]
    target = target.astype(np.float32)
    target = torch.from_numpy(target)


    return inputs, target.reshape(-1,1)

def add_ones(X):
    """
    Add a column of ones at the left hand side of matrix X
    X: (N, d) tensor
    Returns
        (N, d+1) tensor
    """
    ones = torch.ones((X.shape[0],1), dtype=torch.float32)
    X = torch.cat((ones, X), dim=-1)
    return X

def make_tensor(*args):
    """
    Check if arguments are tensor, converts arguments to tensor
    accepts and returns Iterables
    """
    for el in args:
        if not torch.is_tensor(el):
            el = tensor(el)
    return args

def optimal_weight(X,y, bias=True):
    """
    Using invertible matrix method
    X: (N, d) matrix
    y: (d, 1) column vector
    Returns:
        tensor of bias and weigths
    """
    X, y = make_tensor(X,y)
    if bias:    
        X = add_ones(X)
    
    inv = torch.inverse(X.t()@X)
    w_opt = inv @ X.t() @ y    
    return w_opt.reshape(-1,1)

def calc_cost(X, y, theta):
    """
    X: (N, d) tensor
    y: (N, 1) tensor
    theta: (d, 1) or (d+1, 1) tensor
    
    Returns Mean Squared Error
    """
    if theta.shape[0]-1 == X.shape[1]:
        X = add_ones(X)
    y_pred = X@theta
    return ((y_pred - y)**2).sum()/N   

def update_theta(X,y,theta, alpha):
    gradient = X.t() @ (X@theta - y)
    theta_new = (theta - alpha*(gradient)/N)
    return theta_new

def batch_gradient_descent(X,y,theta,alpha=0.1,max_iter=200):
    """
    X: (N, d) matrix (iterable)
    y: (N, 1) column vector (iterable)
    theta: (d,1) or (d+1,1) column vector (iterable)
    alpha: learning rate (float)
    max_iter: no. of epoch (int)
    
    Returns:
        theta: calculated bias and weights 
        cost_history: list containing losses over each epoch
        theta_history: list containing theta over each epoch
    """
    X, y = tensor(X), tensor(y)    
    assert X.shape[0] == y.shape[0], "Dimensions must fit"
    if theta.shape[0]-1 == X.shape[1]:
        X = add_ones(X)
    
    N, d = X.shape
    theta_history = []
    cost_history = []
    for i in range(max_iter):
        print(f"Epoch: {i}")
        theta = update_theta(X,y,theta, alpha)
        cost = calc_cost(X,y,theta)
        print (f"Loss= {cost}")
        theta_history.append(theta)
        cost_history.append(cost)
        
    return theta, cost_history, theta_history

if __name__ == "__main__":
    feature_names = ["RM", "RAD", "CRIM"]
    X, y = load_data()
    N, d = X.shape
    # initialize theta
    theta_random = torch.randn((d+1,1))

    def print_result(result, feature_names=None):
        if feature_names==None:
            feature_names = [i for i in range(len(result))]
        print(f"Bias: {round(result[0][0], 3)}")
        for i in range(1, len(feature_names)):
            print(f" Theta {(feature_names[i])}: {round(result[i][0], 3)}")

    # visualisastion
    plt.figure(figsize=(15, 10))
    plt.title("Effect of Learning Rate (alpha) on error")
    # random init of theta
    theta_random = torch.randn((d+1,1))

    learning_rates = [0.008, 0.005, 0.001, 0.0005, 0.0001]
    store = {"cost": [],"theta": []}
    for alpha in learning_rates:
        final_theta, cost_history, _ = batch_gradient_descent(X, y, theta_random, alpha)
        final_cost = cost_history[-1].item() # tensor

        plt.plot(cost_history, label=f"Alpha: {alpha}, Final Cost: {round(final_cost, 3)}")
        plt.ylabel("Mean Squared Error")
        plt.xlabel("Epochs")
        plt.legend()

        final_cost = round(cost_history[-1].item(),3)
        store["cost"].append(final_cost)
        store["theta"].append(final_theta)
        
    plt.savefig("./alpha_v.png")
    plt.show()

    # output results
    for i in range(len(learning_rates)):
        print(f"\nWhen alpha={learning_rates[i]}")
        print(f"Mean Squared Error= {store['cost'][i]}")
        print_result(store["theta"][i].tolist(), feature_names=None)
