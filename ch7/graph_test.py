#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt


if __name__ == "__main__":
    res = []
    with open("data.csv", "r") as fin:
        head = fin.readline()
        for i in range(30):
            # friends = json.loads(','.join(fin.readline().split(',')[9:]))
            temp = json.loads("[" + fin.readline() + "]")
            temp = [int(temp[0]), int(temp[5]), temp[9]]
            res.append(temp)
    tweet_df = pd.DataFrame(data=res, columns=["Id", "FriendCount", "friends"])
    tweet_df = tweet_df.dropna(axis=0, how="any")
    # 只取好友少于150个的
    tweet_df = tweet_df[(tweet_df["FriendCount"] <= 300).values]

    G = nx.DiGraph()
    main_users = list(tweet_df["Id"].values)
    G.add_nodes_from(main_users)
    for user in tweet_df.values:
        friends = user[2]
        for friend in friends:
            # 由于数据是随机选取的，如果加上核心用户条件就会没有边生成
            # if friend in main_users:
            G.add_edge(user[0], int(friend))
    print("graph finished")
    plt.figure(3, figsize=(200, 200))
    nx.draw(G, alpha=0.1, edge_color="b")
    plt.show()
