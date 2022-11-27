from sklearn.neighbors import KDTree

def make_graph_matrix(X):
    tree = KDTree(X)
    graph = {idx : None for idx in range(X.shape[0])}
    graph = tree.query(X, k = 4)[1][:,1:]
    W = np.zeros((800,800))
    for i in range(800):
        W[i,graph[i,:]] = 1
        W[graph[i,:],i] = 1
    return W