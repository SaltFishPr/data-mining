#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier  # 创建决策树的类
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder  # 能将字符串类型的特征转化成整型
from sklearn.preprocessing import OneHotEncoder  # 将特征转化为二进制数字
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.model_selection import GridSearchCV  # 网格搜索，找到最佳参数

results = pd.read_csv(
    "NBA_data.csv",
    parse_dates=["Date"],
    skiprows=[0,],
    usecols=[0, 2, 3, 4, 5, 6, 7, 9],
)  # 加载数据集
results.columns = [
    "Date",
    "Visitor Team",
    "VisitorPts",
    "Home Team",
    "HomePts",
    "Score Type",
    "OT?",
    "Notes",
]
# results.ix[]已被弃用
print(results.loc[:5])  # 查看数据集前五行
print("--------------------------")

results["HomeWin"] = results["VisitorPts"] < results["HomePts"]
y_true = results["HomeWin"].values  # 胜负情况
# 创建两个新feature，初始值都设为0，保存这场比赛的两个队伍上场比赛的情况
results["HomeLastWin"] = False
results["VisitorLastWin"] = False
won_last = defaultdict(int)

for index, row in results.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    # 这场比赛之前两个球队上次是否获胜保存在result中
    row["HomeLastWin"] = won_last[home_team]
    row["VisitorLastWin"] = won_last[visitor_team]
    results.iloc[index] = row
    # 这场比赛的结果更新won_last中的情况
    won_last[home_team] = row["HomeWin"]
    won_last[visitor_team] = not row["HomeWin"]

X_previouswins = results[["HomeLastWin", "VisitorLastWin"]].values
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_previouswins, y_true, scoring="accuracy")
print("Using just the last result from the home and visitor teams")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))
print("--------------------------")

ladder = pd.read_csv("NBA_standings.csv", skiprows=[0,])
# 创建一个新特征，两个队伍在上个赛季的排名哪个比较高
results["HomeTeamRanksHigher"] = 0
for index, row in results.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    # 这个球队改名了
    if home_team == "New Orleans Pelicans":
        home_team = "New Orleans Hornets"
    elif visitor_team == "New Orleans Pelicans":
        visitor_team = "New Orleans Hornets"
    # 这里源代码无法运行，少加了一个括号 ladder[(ladder["Team"] == home_team)] 表示根据条件获取这一行的数据
    home_row = ladder[(ladder["Team"] == home_team)]
    visitor_row = ladder[(ladder["Team"] == visitor_team)]
    home_rank = home_row["Rk"].values[0]
    visitor_rank = visitor_row["Rk"].values[0]
    row["HomeTeamRanksHigher"] = int(home_rank > visitor_rank)
    results.iloc[index] = row

X_homehigher = results[["HomeLastWin", "VisitorLastWin", "HomeTeamRanksHigher"]].values
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_homehigher, y_true, scoring="accuracy")
print("Using whether the home team is ranked higher")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))
print("--------------------------")

last_match_winner = defaultdict(int)
results["HomeTeamWonLast"] = 0
for index, row in results.iterrows():
    home_team = row["Home Team"]
    visitor_team = row["Visitor Team"]
    # 按照英文字母表排序，不去考虑哪个是主场球队
    teams = tuple(sorted([home_team, visitor_team]))
    # 找到两支球队上次比赛的赢家，更新框中的数据，初始为0
    # 这里的HomeTeamWonLast跟主场客场没有什么关系，也可以叫WhichTeamWonLast，这里为了和源码尽量保持一致使用了源码
    row["HomeTeamWonLast"] = 1 if last_match_winner[teams] == row["Home Team"] else 0
    results.iloc[index] = row
    winner = row["Home Team"] if row["HomeWin"] else row["Visitor Team"]
    # 将两个球队上次遇见比赛的情况存到字典中去
    last_match_winner[teams] = winner

X_home_higher = results[["HomeTeamRanksHigher", "HomeTeamWonLast"]].values
clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_home_higher, y_true, scoring="accuracy")
print("Using whether the home team is ranked higher")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))
print("--------------------------")

# 创建一个转化器实例
encoding = LabelEncoder()
# 将球队名转化为整型
encoding.fit(results["Home Team"].values)
# 抽取所有比赛中主客场球队的球队名，组合起来形成一个矩阵
home_teams = encoding.transform(results["Home Team"].values)
visitor_teams = encoding.transform(results["Visitor Team"].values)
# 建立训练集，[["Home Team Feature"，"Visitor Team Feature"],["Home Team Feature"，"Visitor Team Feature"]...]
X_teams = np.vstack([home_teams, visitor_teams]).T
# 创建转化器实例
onehot = OneHotEncoder()
# 生成转化后的特征
X_teams = onehot.fit_transform(X_teams).todense()

clf = DecisionTreeClassifier(random_state=14)
scores = cross_val_score(clf, X_teams, y_true, scoring="accuracy")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))

clf = RandomForestClassifier(random_state=14, n_jobs=-1)
scores = cross_val_score(clf, X_teams, y_true, scoring="accuracy")
print("Using full team labels is ranked higher")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))
print("--------------------------")

X_all = np.hstack([X_home_higher, X_teams])  # 将上面计算的特征进行组合
print(X_all.shape)
scores = cross_val_score(clf, X_all, y_true, scoring="accuracy")
print("Using whether the home team is ranked higher")
print("Accuracy: {0:.1f}%".format(np.mean(scores) * 100))
print("--------------------------")

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
