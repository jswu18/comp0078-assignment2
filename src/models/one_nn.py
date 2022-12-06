from sklearn.neighbors import KDTree


def one_nn_predict(X_train, y_train, X_test):
    tree = KDTree(X_train)
    return y_train[tree.query(X_test, k=1)[1]].reshape(-1, 1)
