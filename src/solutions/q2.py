from src.models.helpers import ssl_data_sample, KNN_adjacency_matrix
from src.models.SSL import LaplacianInterpolation, LaplacianKernelInterpolation
import os
import numpy as np
import pandas as pd
import dataframe_image as dfi
import matplotlib.pyplot as plt

outpath = os.path.join("outputs", "part2")
DATAPATH50 = os.path.join("data", "dtrain13_50.dat")
DATAPATH100 = os.path.join("data", "dtrain13_100.dat")
DATAPATH200 = os.path.join("data", "dtrain13_200.dat")
DATAPATH400 = os.path.join("data", "dtrain13_400.dat")
REPORT_LAPLACIAN_OUTPATH = os.path.join(outpath, "laplacian_interpolation_report.png")
REPORT_LAPLACIAN_KERNEL_OUTPATH = os.path.join(
    outpath, "laplacian_kernel_interpolation_report.png"
)
GRAPH_LABEL_DIAGRAM_PATH = os.path.join(outpath, 'graph_label_diagram.png')

def experimental_report(
    model,
    datasets,
    n_iters,
):
    sample_range = [1, 2, 4, 8, 16]
    accuracy = np.zeros((4, 5, n_iters))

    for iter in range(n_iters):
        for i, dataset in enumerate(datasets):
            y = dataset[:, 0]
            W = KNN_adjacency_matrix(dataset[:, 1:], k=3)
            for j, sample_size in enumerate(sample_range):
                sample = ssl_data_sample(y, sample_size)
                y_train, y_test = (
                    y[sample[: 2 * sample_size]],
                    y[sample[2 * sample_size :]],
                )
                accuracy[i, j, iter] = (
                    model.predict(W[sample, :][:, sample], y_train) == y_test
                ).mean()
                if i ==0:
                    graph_edge_label_diagram(W,y,GRAPH_LABEL_DIAGRAM_PATH)
    return accuracy.mean(-1), accuracy.std(-1)


def write_report_to_csv(means, stds, outpath):
    df = pd.DataFrame(means).round(2).astype("str")
    df2 = pd.DataFrame(stds).round(4).astype("str")
    report = df + "±" + df2
    report.columns = [1, 2, 4, 8, 16]
    report.index = [50, 100, 200, 400]
    report = pd.concat(
        [pd.concat([report], keys=["# of known labels (per class)"], axis=1)],
        keys=["accuracy"],
    )
    dfi.export(report, outpath)

    
def edge_colour(y1,y2):
    if y1 != y2:
        return 2.0
    if y1 == -1.0:
        return 1.0
    else:
        return 3.0


def graph_edge_label_diagram(W, y, outpath):
    edge_colours = np.zeros((y.shape[0],y.shape[0]))
    for i, y1 in enumerate(y):
        for j, y2 in enumerate(y):
            edge_colours[i,j] = edge_colour(y1,y2)

    W2 = W * edge_colours
    fig = plt.imshow(W2)
    plt.savefig(outpath)


def q2():
    datasets = []
    for path in [DATAPATH50, DATAPATH100, DATAPATH200, DATAPATH400]:
        data = np.genfromtxt(path)
        data[:, 0] -= 2
        datasets.append(data)
    means, stds = experimental_report(LaplacianInterpolation(), datasets, n_iters=20)
    write_report_to_csv(means, stds, outpath=REPORT_LAPLACIAN_OUTPATH)

    means, stds = experimental_report(
        LaplacianKernelInterpolation(), datasets, n_iters=20
    )
    write_report_to_csv(means, stds, outpath=REPORT_LAPLACIAN_KERNEL_OUTPATH)

if __name__ == "__main__":
    q2()
