#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.datasets import load_iris
from collections import defaultdict  # 初始化数据字典
from operator import itemgetter  # 得到一个列表的制定元素
from sklearn.model_selection import train_test_split  # 将一个数据集且分为训练集和测试集
from sklearn.metrics import classification_report


def train(X, y_true, feature):
    """
    Computes the predictors and error for a given feature using the OneR algorithm

    Parameters
    ----------
    X: array [n_samples, n_features]
        The two dimensional array that holds the dataset. Each row is a sample, each column
        is a feature.

    y_true: array [n_samples,]
        The one dimensional array that holds the class values. Corresponds to X, such that
        y_true[i] is the class value for sample X[i].

    feature: int
        An integer corresponding to the index of the variable we wish to test.
        0 <= variable < n_features

    Returns
    -------
    predictors: dictionary of tuples: (value, prediction)
        For each item in the array, if the variable has a given value, make the given prediction.

    error: float
        The ratio of training data that this rule incorrectly predicts.
    """
    # Check that variable is a valid number
    n_samples, n_features = X.shape
    assert 0 <= feature < n_features
    # Get all of the unique values that this variable has
    # X[:, feature]为numpy矩阵的索引用法，第一维：所有数组，第二维：feature，set去重得到value有几个取值
    # 这个feature特征值在每个数据中有多少个取值
    values = set(X[:, feature])
    # Stores the predictors array that is returned
    predictors = dict()
    errors = []
    for current_value in values:  # 对每个特征值的每个取值调用train_feature_value函数获得该取值出现最多的类和错误率
        most_frequent_class, error = train_feature_value(
            X, y_true, feature, current_value
        )
        predictors[current_value] = most_frequent_class  # 该取值出现最多的类
        errors.append(error)  # 存储错误率
    # Compute the total error of using this feature to classify on
    total_error = sum(errors)
    return predictors, total_error  # 返回预测方案（即feature的取值分别对应哪个类别）和总错误率


# Compute what our predictors say each sample is based on its value
# y_predicted = np.array([predictors[sample[feature]] for sample in X])


def train_feature_value(X, y_true, feature, value):
    # Create a simple dictionary to count how frequency they give certain
    # predictions
    class_counts = defaultdict(int)
    # Iterate through each sample and count the frequency of each class/value pair
    # 第feature个特征的值为value的时候，在每个种类中出现的次数，这里的植物有三个种类，因此最终class_counts有三个键值对
    for sample, y in zip(X, y_true):
        if sample[feature] == value:
            class_counts[y] += 1
    # Now get the best one by sorting (highest first) and choosing the first item
    # 对class_count以value由大到小排列
    sorted_class_counts = sorted(class_counts.items(), key=itemgetter(1), reverse=True)
    most_frequent_class = sorted_class_counts[0][0]  # 出现最多次的类
    # The error is the number of samples that do not classify as the most frequent class
    # *and* have the feature value.
    n_samples = X.shape[1]
    error = sum(
        [
            class_count
            for class_value, class_count in class_counts.items()
            if class_value != most_frequent_class
        ]
    )  # error就是除去上面那个类的其它value的和
    return most_frequent_class, error  # 返回出现次数最多的类和错误率


def predict(X_test, model):
    variable = model["variable"]  # 使用哪个feature作为OneRule进行预测
    predictor = model["predictor"]  # 一个字典，保存着feature取值对应哪一类
    y_predicted = np.array([predictor[int(sample[variable])] for sample in X_test])
    return y_predicted  # 返回预测结果


if __name__ == "__main__":
    dataset = load_iris()
    X = dataset.data
    y = dataset.target
    print(dataset.DESCR)
    n_samples, n_features = X.shape

    # Compute the mean for each attribute 计算每个属性的均值
    attribute_means = X.mean(axis=0)
    assert attribute_means.shape == (n_features,)
    X_d = np.array(X >= attribute_means, dtype="int")

    random_state = 14
    X_train, X_test, y_train, y_test = train_test_split(
        X_d, y, random_state=random_state
    )  # 分割训练集和测试集
    print("There are {} training samples".format(y_train.shape))
    print("There are {} testing samples".format(y_test.shape))

    # Compute all of the predictors
    # 对每个特征返回预测器和错误率[0：{0: x, 1: x}, sum_error， ...]
    all_predictors = {
        variable: train(X_train, y_train, variable)
        for variable in range(X_train.shape[1])
    }

    errors = {
        variable: error for variable, (mapping, error) in all_predictors.items()
    }  # 把每个预测器的值提取出来
    # Now choose the best and save that as "model"
    # Sort by error
    # 找出最好的那个feature构成的预测器
    best_variable, best_error = sorted(errors.items(), key=itemgetter(1))[0]
    print(
        "The best model is based on variable {0} and has error {1:.2f}".format(
            best_variable, best_error
        )
    )

    # Choose the bset model
    model = {"variable": best_variable, "predictor": all_predictors[best_variable][0]}
    print(model)

    y_predicted = predict(X_test, model)
    print(y_predicted)
    print(classification_report(y_test, y_predicted))  # 生成测试结果
    print(np.mean(y_predicted == y_test) * 100)
