from sklearn.neighbors import KDTree
import numpy as np

def make_graph_matrix(X):
    tree = KDTree(X)
    n = X.shape[0]
    graph = tree.query(X, k = 4)[1][:,1:]
    W = np.zeros((n,n))
    for i in range(n):
        W[i,graph[i,:]] = 1
        W[graph[i,:],i] = 1
    return W