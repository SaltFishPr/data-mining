# !/usr/bin/env python
# author: Salt Fish
# -*- coding: utf-8 -*-
import os
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from collections import Counter
from scipy.sparse import csr_matrix  # 稀疏矩阵
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree  # 计算最小生成树MST
from scipy.sparse.csgraph import connected_components  # 连通分支
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import HashingVectorizer


base_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(base_folder, "raw")
text_output_folder = os.path.join(base_folder, "textonly")

if __name__ == "__main__":
    # 分簇的数量
    n_clusters = 10
    pipeline = Pipeline(
        [
            ("feature_extraction", TfidfVectorizer(max_df=0.4)),
            ("clusterer", KMeans(n_clusters=n_clusters)),
        ]
    )
    documents = [
        open(os.path.join(text_output_folder, filename)).read()
        for filename in os.listdir(text_output_folder)
    ]

    # 不为fit函数指定目标类别，进行训练
    pipeline.fit(documents)
    # 使用训练过的算法预测
    # labels包含每个数据点的簇标签，标签相同的数据点属于同一个簇，标签本身没有含义
    labels = pipeline.predict(documents)

    # 使用Counter类查看每个簇的数据点数量
    c = Counter(labels)
    for cluster_number in range(n_clusters):
        print(
            "Cluster {} contains {} samples".format(cluster_number, c[cluster_number])
        )
    # 惯性权重，这个值没有意义，但是可以用来确定n_clusters
    print(pipeline.named_steps["clusterer"].inertia_)
    print("--------------------")

    # n_clusters依次取2到20之间的值，每取一个值，k-means算法运行10次，得到拐点值
    # inertia_scores = []
    # n_clusters_values = list(range(2, 20))
    # for n_clusters in n_clusters_values:
    #     # 当前的惯性权重组
    #     cur_inertia_scores = []
    #     X = TfidfVectorizer(max_df=0.4).fit_transform(documents)
    #     for i in range(10):
    #         km = KMeans(n_clusters=n_clusters).fit(X)
    #         cur_inertia_scores.append(km.inertia_)
    #     inertia_scores.append(cur_inertia_scores)
    #     print("{} : {}".format(n_clusters, np.mean(cur_inertia_scores)))
    # print("--------------------")

    # 设置n_clusters值为6， 重新运行算法
    n_clusters = 6
    pipeline = Pipeline(
        [
            ("feature_extraction", TfidfVectorizer(max_df=0.4)),
            ("clusterer", KMeans(n_clusters=n_clusters)),
        ]
    )
    pipeline.fit(documents)
    labels = pipeline.predict(documents)
    terms = pipeline.named_steps["feature_extraction"].get_feature_names()
    c = Counter(labels)
    for cluster_number in range(n_clusters):
        print(
            "Cluster {} contains {} samples".format(cluster_number, c[cluster_number])
        )
        print(" Most important terms")
        centroid = pipeline.named_steps["clusterer"].cluster_centers_[cluster_number]
        most_important = centroid.argsort()
        for i in range(5):
            term_index = most_important[-(i + 1)]
            print(
                " {0} {1} (score: {2:.4f})".format(
                    i + 1, terms[term_index], centroid[term_index]
                )
            )
    print("--------------------")
    # 用K-means算法转化特征
    X = pipeline.transform(documents)

    def create_coassociation_matrix(labels):
        rows = []
        cols = []
        unique_labels = set(labels)
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            for index1 in indices:
                for index2 in indices:
                    rows.append(index1)
                    cols.append(index2)
        data = np.ones((len(rows),))
        return csr_matrix((data, (rows, cols)), dtype="float")

    C = create_coassociation_matrix(labels)
    print(C)
    print((365 ** 2 - create_coassociation_matrix(labels).nnz) / 365 ** 2)

    mst = minimum_spanning_tree(C)
    mst = minimum_spanning_tree(-C)

    pipeline.fit(documents)
    labels2 = pipeline.predict(documents)
    C2 = create_coassociation_matrix(labels2)
    C_sum = (C + C2) / 2
    mst = minimum_spanning_tree(-C_sum)
    # 删除低于阈值的边
    mst.data[mst.data > -1] = 0
    number_of_clusters, labels = connected_components(mst)
    print()

    # 创建证据累积算法类
    class EAC(BaseEstimator, ClusterMixin):
        def __init__(
            self, n_clusterings=10, cut_threshold=0.5, n_clusters_range=(3, 10)
        ):
            self.n_clusterings = n_clusterings
            self.cut_threshold = cut_threshold
            self.n_clusters_range = n_clusters_range

        def fit(self, X, y=None):
            C = sum(
                (
                    create_coassociation_matrix(self._single_clustering(X))
                    for i in range(self.n_clusterings)
                )
            )
            mst = minimum_spanning_tree(-C)
            mst.data[mst.data > -self.cut_threshold] = 0
            mst.eliminate_zeros()
            self.n_components, self.labels_ = connected_components(mst)
            return self

        def _single_clustering(self, X):
            n_clusters = np.random.randint(*self.n_clusters_range)
            km = KMeans(n_clusters=n_clusters)
            return km.fit_predict(X)

        def fit_predict(self, X):
            self.fit(X)
            return self.labels_

    pipeline = Pipeline(
        [("feature_extraction", TfidfVectorizer(max_df=0.4)), ("clusterer", EAC())]
    )
    pipeline.fit(documents)
    number_of_clusters, labels = (
        pipeline["clusterer"].n_components,
        pipeline["clusterer"].labels_,
    )
    print(number_of_clusters)
    print(labels)

    n_clusters = 6
    vec = TfidfVectorizer(max_df=0.4)
    X = vec.fit_transform(documents)
    mbkm = MiniBatchKMeans(random_state=14, n_clusters=3)
    batch_size = 10
    for iteration in range(int(X.shape[0] / batch_size)):
        start = batch_size * iteration
        end = batch_size * (iteration + 1)
        mbkm.partial_fit(X[start:end])
    labels = mbkm.predict(X)
    c = Counter(labels)
    for cluster_number in range(n_clusters):
        print(
            "Cluster {} contains {} samples".format(cluster_number, c[cluster_number])
        )
    print("--------------------")

    class PartialFitPipeline(Pipeline):
        def partial_fit(self, X, y=None):
            Xt = X
            for name, transform in self.steps[:-1]:
                Xt = transform.transform(Xt)
            return self.steps[-1][1].partial_fit(Xt, y=y)

    pipeline = PartialFitPipeline(
        [
            ("feature_extraction", HashingVectorizer()),
            ("clusterer", MiniBatchKMeans(random_state=14, n_clusters=3)),
        ]
    )
    batch_size = 10
    for iteration in range(int(len(documents) / batch_size)):
        start = batch_size * iteration
        end = batch_size * (iteration + 1)
        pipeline.partial_fit(documents[start:end])
    labels = pipeline.predict(documents)
    c = Counter(labels)
    for cluster_number in range(n_clusters):
        print(
            "Cluster {} contains {} samples".format(cluster_number, c[cluster_number])
        )
