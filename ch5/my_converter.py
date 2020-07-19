#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils import as_float_array
from numpy.testing import assert_array_equal


class MeanDiscrete(TransformerMixin):
    def fit(self, X):
        # 尝试对X进行转换，数据转换成float类型
        X = as_float_array(X)
        # 计算数据集的均值
        self.mean = X.mean(axis=0)
        # 返回它本身，进行链式调用transformer.fit(X).transform(X)
        return self

    def transform(self, X):
        X = as_float_array(X)
        # 检查输入是否合法
        assert X.shape[1] == self.mean.shape[0]
        # 返回X中大于均值的数据
        return X > self.mean


def test_meandiscrete():
    X_test = np.array(
        [
            [0, 2],
            [3, 5],
            [6, 8],
            [9, 11],
            [12, 14],
            [15, 17],
            [18, 20],
            [21, 23],
            [24, 26],
            [27, 29],
        ]
    )
    mean_discrete = MeanDiscrete()
    mean_discrete.fit(X_test)
    # 与正确的计算结果进行比较，检查内部参数是否正确设置
    assert_array_equal(mean_discrete.mean, np.array([13.5, 15.5]))
    # 转换后的X
    X_transfromed = mean_discrete.transform(X_test)
    # 验证数据
    X_expected = np.array(
        [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
    )
    assert_array_equal(X_transfromed, X_expected)


if __name__ == "__main__":
    test_meandiscrete()
