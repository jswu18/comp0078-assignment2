from sklearn.neighbors import KDTree
import numpy as np

def make_graph_matrix(X):
    tree = KDTree(X)
    graph = tree.query(X, k = 4)[1][:,1:]
    W = np.zeros((800,800))
    for i in range(800):
        W[i,graph[i,:]] = 1
        W[graph[i,:],i] = 1
    return W