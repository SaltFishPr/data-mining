#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt


# 创建转换函数
def convert_number(x):
    try:
        res = float(x)
        return res
    except ValueError:
        return np.nan


if __name__ == "__main__":
    # 创建数据加载的转换器
    converters = defaultdict(convert_number, {i: convert_number for i in range(1588)})
    converters[1558] = lambda x: 1 if x.strip() == "ad." else 0
    # 使用转换器读取数据集
    temp = pd.read_csv("ad.data", header=None, converters=converters)
    # 删除所有含有nan的行,axis=0是数据索引(index)，axis=1是列标签(column)
    ads = temp.dropna(axis=0, how="any")
    print(ads[10:15])
    print("------------------")

    X = ads.drop(1558, axis=1).values
    y = ads[1558]

    # 参数为主成分数量
    pca = PCA(n_components=5)
    Xd = pca.fit_transform(X)
    # 设置输出选项
    # 第一个参数为输出精度位数，第二个参数是使用定点表示法打印浮点数
    np.set_printoptions(precision=3, suppress=True)
    print(pca.explained_variance_ratio_)
    print("------------------")

    clf = DecisionTreeClassifier(random_state=14)
    scores_reduced = cross_val_score(clf, Xd, y, scoring="accuracy")
    print(np.mean(scores_reduced))
    print("------------------")

    # 获取数据集类别的所有取值
    classes = set(y)
    # 指定在图形中用什么颜色表示这两个类别
    colors = ["red", "green"]
    # 同时遍历这两个容器
    for cur_class, color in zip(classes, colors):
        # 为属于当前类别的所有个体创建遮罩层
        mask = (y == cur_class).values
        plt.scatter(
            Xd[mask, 0], Xd[mask, 1], marker="o", color=color, label=int(cur_class)
        )
    plt.legend()
    plt.show()
    from ch5.my_converter import MeanDiscrete

    pipline = Pipeline(
        [
            ("mean_discrete", MeanDiscrete()),
            ("classifier", DecisionTreeClassifier(random_state=14)),
        ]
    )
    scores_mean_discrete = cross_val_score(pipline, X, y, scoring="accuracy")
    print("Mean Discrete performance: {0:.3f}".format(scores_mean_discrete.mean()))
