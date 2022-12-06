from sklearn.neighbors import KDTree

def KNN_predict(X_train,y_train, X_test, K):
    tree = KDTree(X_train)
    return y_train[tree.query(X_test, k=1)[1]]