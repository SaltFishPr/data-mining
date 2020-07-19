#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import numpy as np
from sklearn.base import TransformerMixin
from nltk import word_tokenize
from sklearn.feature_extraction import DictVectorizer  # 接受元素为字典的列表，将其转换为矩阵
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB  # 用于二值特征分类的 BernoulliNB 分类器，
from sklearn.pipeline import Pipeline


# 创建转换器类
class NLTKBOW(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [{word: True for word in word_tokenize(document)} for document in X]


if __name__ == "__main__":
    tweets = []
    input_filename = ""
    classes_filename = ""
    with open(input_filename) as inf:
        for line in inf:
            if len(line.strip()) == 0:
                continue
            tweets.append(json.loads(line)["text"])

    with open(classes_filename, "r") as inf:
        labels = json.load(inf)

    # 组装流水线
    pipline = Pipeline(
        [
            ("bag-of-words", NLTKBOW()),
            ("vectorizer", DictVectorizer()),
            ("naive-bayes", BernoulliNB()),
        ]
    )
    # 用F1值来评估
    scores = cross_val_score(pipline, tweets, labels, scoring="f1")
    print("Score: {:.3f}".format(np.mean(scores)))

    model = pipline.fit(tweets, labels)
    nb = model.named_steps["naive-bayes"]
    feature_probabilities = nb.feature_log_prob_
    top_features = np.argsort(-feature_probabilities[1])[:50]
    dv = model.named_steps["vectorizer"]
    for i, feature_index in enumerate(top_features):
        print(
            i,
            dv.feature_names_[feature_index],
            np.exp(feature_probabilities[1][feature_index]),
        )
