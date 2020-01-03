# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import numpy as np

import sys

from svm import SVC

sys.path.append("..")


def plot(clf, X, y, grid_size=20):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size),
                         np.linspace(y_min, y_max, grid_size),
                         indexing='ij')
    flatten = lambda m: np.array(m).reshape(-1, )

    result = []
    for (i, j) in itertools.product(range(grid_size), range(grid_size)):
        point = np.array([xx[i, j], yy[i, j]]).reshape(1, 2)
        # print(point)
        # exit()
        result.append(clf.predict(point))

    Z = np.array(result).reshape(xx.shape)

    plt.contourf(xx, yy, Z,
                 cmap=cm.Paired,
                 levels=[-0.001, 0.001],
                 extend='both',
                 alpha=0.8)
    plt.scatter(flatten(X[:, 0]), flatten(X[:, 1]),
                c=flatten(y), cmap=cm.Paired)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.show()


if __name__ == '__main__':
    samples = 10
    features = 2

    X = np.matrix(np.random.normal(size=samples * features).reshape(samples, features))
    y = 2 * (X.sum(axis=1) > 0) - 1.0

    clf = SVC(kernel="linear", C=1.0)
    clf.fit(X, y)

    plot(clf, X, y)

    pred = clf.predict(np.array([-2.76242623, -3.05595614]).reshape(1, 2))
    print(pred)
