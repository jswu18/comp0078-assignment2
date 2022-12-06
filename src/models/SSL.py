import numpy as np


class LaplacianInterpolation:
    def __init__(self):
        self.W = None

    def predict(self, W, y):
        """
        Inputs:

        W : 3-NN adjacency matrix for the datapoints in X
        y : set of labels for x_1, ..., x_l

        Output:

        y_hat: predicted labels for y_{l+1}, ... , y_{n} 

        Note: we have ordered our graph nodes so the first l datapoints are the
        labelled datapoints.

        The following expression was given in the paper provided, and hence used in
        this example for a closed-form solution.

        """
        s = y.shape[0]
        D = np.diag(W.sum(0))
        return np.sign(
            np.linalg.lstsq(D[s:, s:] - W[s:, s:], W[s:, :s] @ y, rcond=None)[0]
        )


class LaplacianKernelInterpolation:
    def __init__(self):
        pass

    def predict(self, W, y):
        
        """
        Inputs:

        W : 3-NN adjacency matrix for the datapoints in X
        y : set of labels for x_1, ..., x_l

        Output:
        
        y_hat: predicted labels for y_{l+1}, ... , y_{n} 
        
        """

        L = np.diag(W.sum(0)) - W
        m = y.shape[0]
        L_pinv = np.linalg.pinv(L)
        alpha = np.linalg.pinv(L_pinv[:m, :m]) @ y
        return np.sign(L_pinv[:m, :].T @ alpha)[m:]
