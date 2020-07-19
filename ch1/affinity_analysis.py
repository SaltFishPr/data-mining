#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
from operator import itemgetter

if __name__ == "__main__":
    dataset_filename = "affinity_dataset.txt"
    X = np.loadtxt(dataset_filename)
    n_samples, n_features = X.shape  # 样本数量，特征数量
    features = ["bread", "milk", "cheese", "apples", "bananas"]  # 商品名列表

    # 如果xxx，那么xxx 就是一条规则。规则由前提条件和结论两部分组成
    # 这里注意'如果买A则他们会买B'和'如果买B则他们会买A'不是一个规则，在下面的循环中体现出来
    valid_rules = defaultdict(int)  # 规则应验
    invalid_rules = defaultdict(int)  # 规则无效
    num_occurences = defaultdict(int)  # 商品购买数量字典

    for sample in X:  # 对数据集里的每个消费者
        for premise in range(n_features):
            if sample[premise] == 0:  # 如果这个商品没有买，继续看下一个商品
                continue
            num_occurences[premise] += 1  # 这个商品购买数量+1
            for conclusion in range(n_features):
                if premise == conclusion:  # 跳过此商品
                    continue
                if sample[conclusion] == 1:
                    valid_rules[(premise, conclusion)] += 1  # 规则应验
                else:
                    invalid_rules[(premise, conclusion)] += 1  # 规则无效
    support = valid_rules  # 支持度字典，即规则应验次数
    confidence = defaultdict(float)  # 置信度字典
    for premise, conclusion in valid_rules.keys():  # 条件/结论
        rule = (premise, conclusion)
        confidence[rule] = (
            valid_rules[rule] / num_occurences[premise]
        )  # 置信度 = 规则发生的次数/条件发生的次数

    def print_rule(premise, conclusion, support, confidence, features):
        premise_name = features[premise]
        conclusion_name = features[conclusion]
        print(
            "Rule: If a person buys {0} they will also buy {1}".format(
                premise_name, conclusion_name
            )
        )
        print(" - Confidence: {0:.3f}".format(confidence[(premise, conclusion)]))
        print(" - Support: {0}".format(support[(premise, conclusion)]))
        print("")

    # 得到支持度最高的规则，items()返回字典所有元素的列表，itemgetter(1)表示用支持度的值作为键，进行降序排列
    sorted_support = sorted(support.items(), key=itemgetter(1), reverse=True)
    for i in range(5):
        print("Rule #{0}".format(i + 1))
        premise, conclusion = sorted_support[i][0]
        print_rule(premise, conclusion, support, confidence, features)

    sorted_confidence = sorted(confidence.items(), key=itemgetter(1), reverse=True)
    for i in range(5):
        print("Rule #{0}".format(i + 1))
        premise, conclusion = sorted_confidence[i][0]
        print_rule(premise, conclusion, support, confidence, features)
