#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import datetime
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from ch3.nba_test import X_all
from sklearn.model_selection import GridSearchCV  # 网格搜索，找到最佳参数


if __name__ == "__main__":
    """
    - 球队上次打比赛距今有多长时间？
    - 两支球队过去五场比赛结果如何？
    - 球队是不是跟某支特定球队打比赛时发挥更好？
    """
    dataset = pd.read_csv(
        "NBA_data.csv",
        parse_dates=["Date"],
        skiprows=[0,],
        usecols=[0, 2, 3, 4, 5, 6, 7, 9],
    )  # 加载数据集
    dataset.columns = [
        "Date",
        "Visitor Team",
        "VisitorPts",
        "Home Team",
        "HomePts",
        "Score Type",
        "OT?",
        "Notes",
    ]
    dataset["HomeWin"] = dataset["VisitorPts"] < dataset["HomePts"]
    y_true = dataset["HomeWin"].values  # 胜负情况

    # 保存上次打比赛的时间
    last_played_date = defaultdict(datetime.date)
    # 手动为每个球队初始化
    for team in set(dataset["Home Team"]):
        last_played_date[team] = datetime.date(year=2013, month=10, day=25)
    # 两支球队过去的比赛结果，每个球队的数据是[True,False,,,]的序列
    last_five_games = defaultdict(list)

    # 存放Home和Visitor前五次比赛的获胜次数
    dataset["HWinTimes"] = 0
    dataset["VWinTimes"] = 0
    # 存放距离上次比赛的时间间隔，用天计数
    dataset["HLastPlayedSpan"] = 0
    dataset["VLastPlayedSpan"] = 0
    for index, row in dataset.iterrows():
        home_team = row["Home Team"]
        visitor_team = row["Visitor Team"]

        row["HWinTimes"] = sum(last_five_games[home_team][-5:])
        row["VWinTimes"] = sum(last_five_games[visitor_team][-5:])
        row["HLastPlayedSpan"] = (row["Date"].date() - last_played_date[home_team]).days
        row["VLastPlayedSpan"] = (
            row["Date"].date() - last_played_date[visitor_team]
        ).days

        dataset.iloc[index] = row

        last_played_date[home_team] = row["Date"].date()
        last_played_date[visitor_team] = row["Date"].date()
        last_five_games[home_team].append(row["HomeWin"])
        last_five_games[visitor_team].append(not row["HomeWin"])

    X_1 = dataset[
        ["HLastPlayedSpan", "VLastPlayedSpan", "HWinTimes", "VWinTimes"]
    ].values
    clf = DecisionTreeClassifier(random_state=14)
    scores = cross_val_score(clf, X_1, y_true, scoring="accuracy")
    print("DecisionTree: Using time span and win times")
    print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

    clf = RandomForestClassifier(random_state=14, n_jobs=-1)
    scores = cross_val_score(clf, X_1, y_true, scoring="accuracy")
    print("RandomForest: Using time span and win times")
    print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))
    print("---------------------------------")

    X_all = np.hstack([X_1, X_all])

    clf = DecisionTreeClassifier(random_state=14)
    scores = cross_val_score(clf, X_all, y_true, scoring="accuracy")
    print("DecisionTree: Using time span and win times")
    print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

    clf = RandomForestClassifier(random_state=14, n_jobs=-1)
    scores = cross_val_score(clf, X_all, y_true, scoring="accuracy")
    print("RandomForest: Using time span and win times")
    print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))
    print("---------------------------------")
    parameter_space = {
        "max_features": [2, 10, "auto"],
        "n_estimators": [100,],
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": [2, 4, 6],
    }
    grid = GridSearchCV(clf, parameter_space)
    grid.fit(X_all, y_true)
    print("Accuracy: {0:.1f}%".format(grid.best_score_ * 100))
    print(grid.best_estimator_)
