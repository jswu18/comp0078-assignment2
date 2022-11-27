import numpy as np


def Laplace_interpolation(W: np.ndarray, l: int) -> np.ndarray:
    """ 
    Inputs:
     
    D: graph degree matrix
    w: graph adjacency matrix (both nxn)

    Note: we have ordered our graph nodes so the first l datapoints are the 
    labelled datapoints.

    The following expression was given in the paper provided, and hence used in
    this example for a closed-form solution.

    """
    D = np.diag(W.sum(0))
    return np.sign(np.linalg.solve(D[l:,l:] - W[l:,l:], W[l:,:l] * y[:l]))
