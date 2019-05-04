import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from matplotlib.patches import Ellipse
from sklearn.datasets.samples_generator import make_blobs

def load_data():
    'load data as given in the question'
    K = 3
    NUM_DATAPTS = 150

    X,y = make_blobs(n_samples=NUM_DATAPTS, 
                     centers=K,shuffle=False, 
                     random_state=0,
                     cluster_std=0.6)

    g1 = np.asarray([[2.0,0],[-0.9,1]])
    g2 = np.asarray([[1.4,0],[0.5,0.7]])

    mean1 = np.mean(X[:int(NUM_DATAPTS/K)])
    mean2 = np.mean(X[int(NUM_DATAPTS/K):2*int(NUM_DATAPTS/K)])

    X[:int(NUM_DATAPTS/K)] = np.einsum('nj, ij -> ni',
                                       X[:int(NUM_DATAPTS/K)] - mean1, g1) + mean1
    X[int(NUM_DATAPTS/K):2*int(NUM_DATAPTS/K)] = np.einsum('nj, ij -> ni',
                                                          X[int(NUM_DATAPTS/K):2*int(NUM_DATAPTS/K)] - mean2,
                                                           g2) + mean2
    X[:,1] -= 4
    return X

def init_centroids(X, n_cluster):
    """
    Random draw first cluster centroid from current data
    Second centroid will be point with largest euclidean distance from first centroid
    Third centroid will be furthest point from both first and second 
    j centroid will be furthest point from 1st, ... , (j-1)th centroid
    
    X: 2d Array of data
    n_cluster: number of clusters

    Returns
        2d array of init cluster centroids
    """
    N, d = X.shape
    mean_indices = [np.random.randint(N)]
    for j in range(n_cluster-1):
        furthest_distance = 0
        furthest_point_index = None
        for i in range(N):
            if i in mean_indices:
                continue

            current_point = X[i]
            current_distance = sum([sum((current_point - X[index])**2) for index in mean_indices])

            if current_distance > furthest_distance:
                furthest_distance = current_distance
                furthest_point_index = i

        mean_indices.append(furthest_point_index)
    return X[mean_indices]

def initialize(X, K=3):
    N, d = X.shape
    #MEANS = np.random.rand(K,2)
    MEANS = init_centroids(X, n_cluster=K)
    COVARIANCES = np.array([np.eye(N=d) for _ in range(K)])
    CLUSTER_COEFFICIENTS = np.random.uniform(size=(K))
    return MEANS, COVARIANCES, CLUSTER_COEFFICIENTS

def multivariate_gaussian_pdf(x, mu, sigma):
    D = x.shape[0]
    exp_term = np.exp(-0.5*(x - mu).T@np.linalg.inv(sigma)@(x - mu))
    a = 1/ (np.sqrt(2*np.pi)**D)
    b = 1/np.sqrt(LA.det(sigma))
    return a*b*exp_term

def calc_gamma(x, cluster_idx):
    mu = MEANS[cluster_idx]
    sigma = COVARIANCES[cluster_idx]
    pi_k = CLUSTER_COEFFICIENTS[cluster_idx]
    gamma =  pi_k*multivariate_gaussian_pdf(x, mu, sigma) / sum([CLUSTER_COEFFICIENTS[cluster_j]*multivariate_gaussian_pdf(x, MEANS[cluster_j], COVARIANCES[cluster_j]) 
     for cluster_j in range(K)])
    return gamma

def E_step():
    gamma = np.zeros((NUM_DATAPTS, K))
    for i in range(NUM_DATAPTS):
        x = X[i]
        for cluster_idx in range(K):
            current_gamma = calc_gamma(x, cluster_idx)
            gamma[i, cluster_idx] = current_gamma
    return gamma

def M_step(gamma):
    N_effect = [sum([gamma[i, cluster_j] for i in range(NUM_DATAPTS)]) for cluster_j in range(K)]
    
    MEANS = np.array([
        (1/N_effect[cluster_idx]) * sum([gamma[i, cluster_idx]*X[i] for i in range(NUM_DATAPTS)])
              for cluster_idx in range(K)])
    
    COVARIANCES = np.zeros((3,2,2))
    for cluster_idx in range(K):
        mean = MEANS[cluster_idx]       
        sigma_k = 0
        for i in range(NUM_DATAPTS):
            x = X[i]
            x_minus_mu = (x - mean).reshape(-1,1)
            sigma_k += gamma[i, cluster_idx]*x_minus_mu@(x_minus_mu.T)            
        COVARIANCES[cluster_idx] = (1/N_effect[cluster_idx] * sigma_k)

    CLUSTER_COEFFICIENTS = [N_k/NUM_DATAPTS for N_k in N_effect]
    
    return MEANS, COVARIANCES, CLUSTER_COEFFICIENTS

def log_likelihood(X, mu, sigma, cluster_coef):
    ll = 0
    for i in range(NUM_DATAPTS):
        x = X[i]
        ll_i = 0
        for cluster_idx in range(K):
            ll_i += cluster_coef[cluster_idx]*multivariate_gaussian_pdf(x, mu[cluster_idx], sigma[cluster_idx])
        ll += np.log(ll_i)
    return ll

def plot_result(gamma=None):
    ax = plt.subplot(111, aspect='equal')
    ax.set_xlim([-5,5])
    ax.set_ylim([-5,5])
    ax.scatter(X[:, 0], X[:, 1], c=gamma, s=50, cmap=None)
    
    for k in range(K):
        l, v = LA.eig(COVARIANCES[k])
        theta = np.arctan(v[1,0]/v[0,0])
        e = Ellipse((MEANS[k,0],MEANS[k,1]),6*l[0],6*l[1],
                    theta*180/np.pi)
        e.set_alpha(0.5)
        ax.add_artist(e)
    plt.show()

if __name__ == "__main__":
    K = 3
    NUM_DATAPTS = 150
    X = load_data()
    MEANS, COVARIANCES, CLUSTER_COEFFICIENTS = initialize(X)
    MEANS = init_centroids(X, K)

    max_iter = 50;

    prev_ll = -np.inf
    for _ in range(max_iter):
        gamma = E_step()
        MEANS, COVARIANCES, CLUSTER_COEFFICIENTS = M_step(gamma)
        ll = log_likelihood(X, MEANS, COVARIANCES, CLUSTER_COEFFICIENTS)

        improvement = ll - prev_ll
        prev_ll = ll

        # stopping condition: if converge
        if improvement < 0.01:
            print(f"Algorithm converges within {_} iterations")
            break
        print(f"Iteration: {_}")
        print(f"Log Likelihood: {ll}")
        print(f"Improvement: {improvement}")    
    plot_result(gamma)
