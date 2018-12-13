import numpy
import numpy as np
from matplotlib import pyplot as plt

from matplotlib.pyplot import figure, subplot, scatter, plot, savefig
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize_scatter(data, label_ids, id_to_label_dict, figsize=(20, 20), save_as="plot.png"):
    data_2d = TSNE(n_components=2).fit_transform(data)
    plt.figure(figsize=figsize)
    # plt.grid()

    nb_classes = len(np.unique(label_ids))

    for label_id in np.unique(label_ids):
        plt.scatter(data_2d[np.where(label_ids == label_id), 0],
                    data_2d[np.where(label_ids == label_id), 1],
                    marker='o',
                    color=plt.cm.Set1(label_id / float(nb_classes)),
                    linewidth='1',
                    alpha=0.8,
                    label=id_to_label_dict[label_id])
    plt.legend(loc='best')
    savefig(save_as)
    plt.show()


if __name__ == "__main__":
    # here is an example
    X = np.array([[0, 0, 0],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    visualize_scatter(X, [0, 1, 0, 0], {0: "Class_0", 1: "Class_1"})
