import torch

t_type = torch.float64

def add_ones(X):
    """
    Add a column of ones at the left hand side of matrix X
    X: (N, d) tensor
    Returns
        (N, d+1) tensor
    """
    ones = torch.ones((X.shape[0],1), dtype=t_type)
    X = torch.cat((ones, X), dim=-1)
    return X

def make_tensor(*args):
    """
    Check if arguments are tensor, converts arguments to tensor
    accepts and returns Iterables
    """
    tensors = [el if torch.is_tensor(el) else torch.tensor(el, dtype=t_type) for el in args ]
    return tensors[0] if len(tensors)==1 else tensors

def minmax_scale(X):
    """
    X: 2 dim. numpy array or torch tensor
    """
    N, d = X.shape
    for i in range(d):
        col = X[:, i]
        col_max, col_min = col.max(), col.min()
        if col_max == col_min:
            continue
        else:
            X[:, i] = (col - col_min) / (col_max - col_min)
    return X
