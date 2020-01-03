# -*- coding: utf-8 -*-

import numpy as np
import cvxopt
import cvxopt.solvers


class SVC(object):
    def __init__(self, kernel='linear', C=1.0, sigma=1.0, **kwargs):
        """
        kernel: 选择核函数，默认为线性核linear
        C: 求对偶问题所需要的常数参数，默认为1.0
        sigma: 高斯核函数的参数sigma，默认为1.0
        """
        # 选择核函数
        if kernel not in ['linear', 'gaussian']:
            raise ValueError("Not supported kernel.")
        elif kernel == 'linear':
            kernel_fn = Kernel.linear()
        else:
            kernel_fn = Kernel.gaussian(sigma)

        self.kernel = kernel_fn
        self.C = C
        self._predictor = None

    def fit(self, X, y):
        """
        训练数据
        """

        # 获取拉格朗日乘子法中的乘子
        lagr = self._lagr_multiplier(X, y)
        self._predictor = self._fit(X, y, lagr)

    def predict(self, X):
        """
        预测
        """
        return self._predictor.predict(X)

    def _fit(self, X, y, lagr, support_vector_threhold=1e-5):

        # 获得支持向量
        support_vectors_id = lagr > support_vector_threhold
        support_lagr = lagr[support_vectors_id]
        support_vectors = X[support_vectors_id]
        support_vector_tags = y[support_vectors_id]

        # 计算偏差
        bias = np.mean([y_k - Predictor(kernel=self.kernel,
                                        bias=0.0,
                                        W=support_lagr,
                                        support_vectors=support_vectors,
                                        support_vector_tags=support_vector_tags)
                       .predict(x_k) for (y_k, x_k) in zip(support_vector_tags, support_vectors)])
        # 返回训练结果
        return Predictor(kernel=self.kernel,
                         bias=bias,
                         W=support_lagr,
                         support_vectors=support_vectors,
                         support_vector_tags=support_vector_tags)

    def _lagr_multiplier(self, X, y):
        """
        求解拉格朗日乘子
        """
        samples, features = X.shape

        k = self._mapping(X)

        # 对偶问题求解
        # 利用cvxopt二次规划优化包

        # 第一步
        P = cvxopt.matrix(np.outer(y, y) * k)
        q = cvxopt.matrix(-1 * np.ones(samples))

        G_std = cvxopt.matrix(np.diag(np.ones(samples) * -1))
        h_std = cvxopt.matrix(np.zeros(samples))

        # 第二步: \alpha_i <= C
        G_slack = cvxopt.matrix(np.diag(np.ones(samples)))
        h_slack = cvxopt.matrix(np.ones(samples) * self.C)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        # 第三步：\sum_i=1^n \alpha_i * y_i = 0
        A = cvxopt.matrix(y, (1, samples))
        b = cvxopt.matrix(0.0)

        res = cvxopt.solvers.qp(P, q, G, h, A, b)
        # 返回乘子
        return np.ravel(res['x'])

    def _mapping(self, X):
        """
        利用核函数映射到高维空间
        """
        samples, features = X.shape
        k = np.zeros((samples, samples))
        # 循环将所有样本映射到高维
        for i, xi in enumerate(X):
            for j, xj in enumerate(X):
                k[i, j] = self.kernel(xi, xj)
        return k


class Kernel(object):
    """
    核函数相关类
    """

    @staticmethod
    def linear():
        """
        线性核
        """
        # 线性核直接求内积即可
        return lambda X, y: np.inner(X, y)

    @staticmethod
    def gaussian(sigma):
        """
        高斯核
        高斯核需要指定sigma参数
        """
        return lambda X, y: np.exp(-np.sqrt(np.linalg.norm(X - y) ** 2 / (2 * sigma ** 2)))


class Predictor(object):
    def __init__(self, kernel, bias, W, support_vectors, support_vector_tags):
        """
        kernel: 核函数
        bias：偏置
        support_vectors: 支持向量
        support_vector_tags：支持向量对应的标签
        """
        self._kernel = kernel
        self._bias = bias
        self._W = W
        self._support_vectors = support_vectors
        self._support_vector_tags = support_vector_tags
        assert len(support_vectors) == len(support_vector_tags)
        assert len(W) == len(support_vector_tags)

    def predict(self, x):
        res = self._bias
        for z_i, x_i, y_i in zip(self._W, self._support_vectors, self._support_vector_tags):
            res += z_i * y_i * self._kernel(x_i, x)
        # 返回正负预测结果
        return np.sign(res).item()
