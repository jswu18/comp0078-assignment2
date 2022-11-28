from src.models.helpers import ssl_data_sample
from src.models.SSL import laplacian_interpolation
import os
import numpy as np
import pandas as pd

DATAPATH50 = os.path.join('data','dtrain13_50.dat')
DATAPATH100 = os.path.join('data','dtrain13_100.dat')
DATAPATH200 = os.path.join('data','dtrain13_200.dat')
DATAPATH400 = os.path.join('data','dtrain13_400.dat')
REPORT_LAPLACIAN_OUTPATH = os.path.join('outputs', 'laplacian_interpolation_report')

def laplacian_experimental_report(
    datasets,
    n_iters,
    ):
    sample_range = [1,2,4,8,16]
    errors = np.zeros((4,5,n_iters))
    for iter in range(n_iters):
        for i in range(4):
            for j in range(5):
                X,y_train,y_test = ssl_data_sample(datasets[i],sample_range[j])
                errors[i,j, iter] = (laplacian_interpolation(X,y_train) == y_test).mean()
    return errors.mean(-1), errors.std(-1)


def write_report_to_csv(means, stds, outpath):
    df = pd.DataFrame(means).round(2).astype('str')
    df2 = pd.DataFrame(stds).round(2).astype('str')

    report = df + '±' + df2
    report.columns = [1,2,4,8,16]
    report.index = [50,100,200,400]
    report.to_csv(outpath)

def q2():
    datasets = []
    for path in [DATAPATH50,DATAPATH100,DATAPATH200,DATAPATH400]:
        data = np.genfromtxt(path)
        data[:,0] -= 2
        datasets.append(data)
    means, stds = laplacian_experimental_report(datasets, n_iters = 20)
    write_report_to_csv(means, stds, outpath = REPORT_LAPLACIAN_OUTPATH)

if __name__ == '__main__':
    q2()



