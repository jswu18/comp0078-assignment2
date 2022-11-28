import numpy as np
from sklearn import KDTree

def _make_graph_matrix(X):
    tree = KDTree(X)
    n = X.shape[0]
    graph = tree.query(X, k = 4)[1][:,1:]
    W = np.zeros((n,n))
    for i in range(n):
        W[i,graph[i,:]] = 1
        W[graph[i,:],i] = 1
    return W


def laplacian_interpolation(X: np.ndarray, y: int) -> np.ndarray:
    """ 
    Inputs:
     
    D: graph degree matrix
    w: graph adjacency matrix (both nxn)

    Note: we have ordered our graph nodes so the first l datapoints are the 
    labelled datapoints.

    The following expression was given in the paper provided, and hence used in
    this example for a closed-form solution.

    """
    W = _make_graph_matrix(X)
    s = y.shape[0]
    D = np.diag(W.sum(0))
    return np.sign(np.linalg.solve(D[s:,s:] - W[s:,s:], W[s:,:s] @ y))
