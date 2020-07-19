#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.stats import pearsonr
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

if __name__ == "__main__":
    adult = pd.read_csv(
        "adult.data",
        header=None,
        names=[
            "Age",
            "Work-Class",
            "fnlwgt",
            "Education",
            "Education-Num",
            "Marital-Status",
            "Occupation",
            "Relationship",
            "Race",
            "Sex",
            "Capital-gain",
            "Capital-loss",
            "Hours-per-week",
            "Native-Country",
            "Earnings-Raw",
        ],
    )
    # 去除空值
    adult.dropna(how="all", inplace=True)
    # 输出详细描述
    print(adult["Hours-per-week"].describe())
    # 输出中位数
    print(adult["Education-Num"].median())
    # 输出工作的种类
    print(adult["Work-Class"].unique())
    # 将工作时长二值化为是否超过40h
    adult["LongHours"] = adult["Hours-per-week"] > 40
    print("----------------")

    # 构造测试数据集
    X = np.arange(30).reshape((10, 3))
    X[:, 1] = 1
    print(X)
    print("----------------")
    vt = VarianceThreshold()
    Xt = vt.fit_transform(X)
    # 第二列消失了，因为第二列都是1，方差为0，不包括具有区别意义的信息
    print(Xt)
    print("----------------")
    print(vt.variances_)
    print("----------------")

    # 构造数据集
    X = adult[
        ["Age", "Education-Num", "Capital-gain", "Capital-loss", "Hours-per-week"]
    ]
    y = (adult["Earnings-Raw"] == " >50K").values
    # 使用SelectKBest转换器，用卡方打分
    transformer = SelectKBest(score_func=chi2, k=3)
    # 调用fit_transform方法对相同的数据集进行预处理和转换
    Xt_chi2 = transformer.fit_transform(X, y)
    # 输出每个特征的得分
    print(transformer.scores_)
    print("----------------")

    # 用皮尔逊相关系数计算相关性

    def mutivariate_pearsonr(X, y):
        scores, pvalues = [], []
        for column in range(X.shape[1]):
            cur_score, cur_p = pearsonr(X[:, column], y)
            scores.append(abs(cur_score))
            pvalues.append(cur_p)
        return np.array(scores), np.array(pvalues)

    transformer = SelectKBest(score_func=mutivariate_pearsonr, k=3)
    Xt_pearson = transformer.fit_transform(X, y)
    print(transformer.scores_)
    print("----------------")

    clf = DecisionTreeClassifier(random_state=14)
    scores_chi2 = cross_val_score(clf, Xt_chi2, y, scoring="accuracy")
    scores_pearson = cross_val_score(clf, Xt_pearson, y, scoring="accuracy")
    print("卡方: {}".format(np.mean(scores_chi2)))
    print("----------------")
    print("pearson:  {}".format(np.mean(scores_pearson)))
    print("----------------")
