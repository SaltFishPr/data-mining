#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import csv
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier  # 导入K近邻分类器
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score  # 导入交叉检验的

# 把每个特征值的值域规范化到0，1之间，最小值用0代替，最大值用1代替
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline  # 流水线


if __name__ == "__main__":
    # 数据集大小已知有351行，每行35个值前34个为天线采集的数据，最后一个 g/b 表示数据的好坏
    X = np.zeros((351, 34), dtype="float")
    y = np.zeros((351,), dtype="bool")

    # 打开根目录的数据集文件
    with open("ionosphere.data", "r", encoding="utf-8") as input_file:
        # 创建csv阅读器对象
        reader = csv.reader(input_file)
        # 使用枚举函数为每行数据创建索引
        for i, row in enumerate(reader):
            # 获取行数据的前34个值，并将其转化为浮点型，保存在X中
            data = [float(datum) for datum in row[:-1]]
            # Set the appropriate row in our dataset
            X[i] = data  # 数据集
            # 1 if the class is 'g', 0 otherwise
            y[i] = row[-1] == "g"  # 类别

    # 创建训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=14)
    print("There are {} samples in the training dataset".format(X_train.shape[0]))
    print("There are {} samples in the testing dataset".format(X_test.shape[0]))
    print("Each sample has {} features".format(X_train.shape[1]))

    # 初始化一个K近邻分类器实例，该算法默认选择5个近邻作为分类依据
    estimator = KNeighborsClassifier()
    # 用训练数据进行训练
    estimator.fit(X_train, y_train)
    # 使用测试集测试算法，评价其表现
    y_predicted = estimator.predict(X_test)
    # 准确性
    accuracy = np.mean(y_test == y_predicted) * 100
    print("The accuracy is {0:.1f}%".format(accuracy))

    # 使用交叉检验的方式获得平均准确性
    scores = cross_val_score(estimator, X, y, scoring="accuracy")
    average_accuracy = np.mean(scores) * 100
    print("The average accuracy is {0:.1f}%".format(average_accuracy))

    # 设置参数
    # 参数的选取跟数据集的特征息息相关
    avg_scores = []
    all_scores = []
    parameter_values = list(range(1, 21))
    for n_neighbors in parameter_values:
        estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores = cross_val_score(estimator, X, y, scoring="accuracy")
        avg_scores.append(np.mean(scores))
        all_scores.append(scores)

    # 作出n_neighbors不同取值和分类正确率之间的关系的折线图
    plt.figure(figsize=(32, 20))
    plt.plot(parameter_values, avg_scores, "-o", linewidth=5, markersize=24)
    plt.show()

    # 模拟脏数据
    X_broken = np.array(X)
    X_broken[:, ::2] /= 10
    # 对比两种情况下预测准确率
    estimator = KNeighborsClassifier()
    original_scores = cross_val_score(estimator, X, y, scoring="accuracy")
    print(
        "The original average accuracy for is {0:.1f}%".format(
            np.mean(original_scores) * 100
        )
    )
    broken_scores = cross_val_score(estimator, X_broken, y, scoring="accuracy")
    print(
        "The broken average accuracy for is {0:.1f}%".format(
            np.mean(broken_scores) * 100
        )
    )

    # 组合成为一个工作流
    X_transformed = MinMaxScaler().fit_transform(X_broken)  # 完成训练和转换
    estimator = KNeighborsClassifier()
    transformed_scores = cross_val_score(
        estimator, X_transformed, y, scoring="accuracy"
    )
    print(
        "The average accuracy for is {0:.1f}%".format(np.mean(transformed_scores) * 100)
    )

    # 创建流水线
    # 流水线的每一步都用('名称',步骤)的元组表示
    scaling_pipeline = Pipeline(
        [("scale", MinMaxScaler()), ("predict", KNeighborsClassifier())]  # 规范特征取值
    )  # 预测

    # 调用流水线
    scores = cross_val_score(scaling_pipeline, X_broken, y, scoring="accuracy")
    print(
        "The pipelin scored an average accuracy for is {0:.1f}%".format(
            np.mean(scores) * 100
        )
    )
