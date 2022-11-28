from sklearn.neighbors import KDTree
import numpy as np


def ssl_data_sample(data, sample_size):
    idxs = np.random.choice(list(np.where(data[:,0] == -1.)[0]), size = sample_size, replace = False).tolist()
    idxs += np.random.choice(list(np.where(data[:,0] == 1.)[0]), size = sample_size, replace = False).tolist()
    temp = np.concatenate((data[idxs], np.delete(data, idxs, axis = 0)))
    X,y_train, y_test = temp[:,1:], temp[:2*sample_size,0], temp[2*sample_size:,0]
    return X, y_train, y_test