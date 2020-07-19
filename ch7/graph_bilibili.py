#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import silhouette_score

if __name__ == "__main__":
    with open("bili.txt", mode="r") as fin:
        temp = json.load(fin)
    users = pd.DataFrame(temp)
    users.columns = ["Id", "Friends"]
    print(users[:5])

    G = nx.DiGraph()
    main_users = list(users["Id"].values)
    for u in main_users:
        G.add_node(u, label=u)
    for user in users.values:
        friends = user[1]
        for friend in friends:
            if friend in main_users:
                G.add_edge(user[0], int(friend))
    print("graph finished")
    # plt.figure(3, figsize=(100, 100))
    # nx.draw(G, alpha=0.1, edge_color='b', with_labels=True, font_size=16, node_size=30, node_color='r')
    # plt.savefig('fix.png')

    friends = {user: set(friends) for user, friends in users.values}

    def compute_similarity(friends1, friends2):
        return len(friends1 & friends2) / len(friends1 | friends2)

    def create_graph(followers, threshold=0.0):
        G = nx.Graph()
        for user1 in friends.keys():
            if len(friends[user1]) == 0:
                continue
            for user2 in friends.keys():
                if len(friends[user2]) == 0:
                    continue
                if user1 == user2:
                    continue
                weight = compute_similarity(friends[user1], friends[user2])
                if weight >= threshold:
                    G.add_node(user1, lable=user1)
                    G.add_node(user2, lable=user2)
                    G.add_edge(user1, user2, weight=weight)
        return G

    # G = create_graph(friends)
    # plt.figure(3, figsize=(100, 100))
    # pos = nx.spring_layout(G)
    # nx.draw_networkx_nodes(G, pos, node_size=30)
    # edgewidth = [d['weight'] for (u, v, d) in G.edges(data=True)]
    # nx.draw_networkx_edges(G, pos, width=edgewidth)
    # plt.savefig('fix2.png')

    G = create_graph(friends, 0.1)
    sub_graphs = nx.connected_components(G)
    for i, sub_graphs in enumerate(sub_graphs):
        n_nodes = len(sub_graphs)
        print("Subgraph{} has {} nodes".format(i, n_nodes))
    print("---------------------")
    G = create_graph(friends, 0.15)
    sub_graphs = nx.connected_components(G)
    for i, sub_graphs in enumerate(sub_graphs):
        n_nodes = len(sub_graphs)
        print("Subgraph{} has {} nodes".format(i, n_nodes))

    sub_graphs = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    n_subgraphs = nx.number_connected_components(G)
    fig = plt.figure(figsize=(20, (n_subgraphs * 3)))
    for i, sub_graph in enumerate(sub_graphs):
        ax = fig.add_subplot(int(n_subgraphs / 3) + 1, 3, i + 1)
        # 将坐标轴标签关掉
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        pos = nx.spring_layout(G)
        nx.draw(
            G=G.subgraph(sub_graph),
            alpha=0.1,
            edge_color="b",
            with_labels=True,
            font_size=16,
            node_size=30,
            node_color="r",
            ax=ax,
        )
    # plt.show()

    def compute_silhouette(threshold, friends):
        G = create_graph(friends, threshold=threshold)
        if len(G.nodes()) < 2:
            return -99
        sub_graphs = nx.connected_components(G)
        if not (2 <= nx.number_connected_components(G) < len(G.nodes()) - 1):
            return -99
        label_dict = {}
        for i, sub_graph in enumerate(sub_graphs):
            for node in sub_graph:
                label_dict[node] = i
        labels = np.array([label_dict[node] for node in G.nodes()])
        X = nx.to_scipy_sparse_matrix(G).todense()
        X = 1 - X
        np.fill_diagonal(X, 0)
        return silhouette_score(X, labels, metric="precomputed")

    def inverted_silhouette(threshold, friends):
        res = compute_silhouette(threshold, friends=friends)
        return -res

    result = minimize(
        inverted_silhouette, 0.1, args=(friends,), options={"maxiter": 10}
    )
    print(result.x)
