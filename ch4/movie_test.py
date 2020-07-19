#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import pandas as pd
from collections import defaultdict
from operator import itemgetter


def get_movie_name(movie_id):
    title_object = movie_name_data[movie_name_data["MovieID"]
                                   == movie_id]["Title"]
    title = title_object.values[0]
    return title


if __name__ == '__main__':
    # header=None 不把第一行当做表头
    all_ratings = pd.read_csv(
        "ml-100k/u.data",
        delimiter="\t",
        header=None,
        names=[
            "UserID",
            "MovieID",
            "Rating",
            "Datetime"])
    # 转化时间戳为datetime
    all_ratings["Datetime"] = pd.to_datetime(all_ratings["Datetime"], unit='s')
    # 输出用户-电影-评分稀疏矩阵
    print(all_ratings[:5])
    # 创建Favorite特征，将评分属性二值化为是否喜欢
    all_ratings["Favorable"] = all_ratings["Rating"] > 3
    # 取用户ID为前200的用户的打分数据
    ratings = all_ratings[all_ratings["UserID"].isin(range(200))]
    favorable_ratings = ratings[ratings["Favorable"]]
    # 创建用户喜欢哪些电影的字典
    favorable_reviews_by_users = dict(
        (k,
         frozenset(
             v.values)) for k,
        v in favorable_ratings.groupby("UserID")["MovieID"])
    # 创建一个数据框，了解每部电影的影迷数量
    num_favorable_by_movie = ratings[[
        "MovieID", "Favorable"]].groupby("MovieID").sum()
    # 查看最受欢迎的五部电影
    print(num_favorable_by_movie.sort_values("Favorable", ascending=False)[:5])

    # 字典保存最新发现的频繁项集
    frequent_itemsets = {}
    min_support = 50

    # 第一步，每一步电影生成只包含它自己的项集
    # frozenset() 返回一个冻结的集合，冻结后集合不能再添加或删除任何元素
    # 普通集合可变，集合中不能有可变的元素，因此普通集合不能被放在集合中；冻结集合不可变，因此可以被放入集合
    frequent_itemsets[1] = dict(
        (frozenset(
            (movie_id,
             )),
         row["Favorable"]) for movie_id,
        row in num_favorable_by_movie.iterrows() if row["Favorable"] > min_support)

    # 会有重复，导致喜欢电影1,50的人分别为50,100但是 {1,50} 的集合有100个
    # 两个原因，第一在current_superset时项集有时候会突然调换位置
    def find_frequent_itemsets(
            favorable_reviews_by_users,
            k_1_itemsets,
            min_support):
        counts = defaultdict(int)
        # 遍历每一个用户，获取其喜欢的电影
        for user, reviews in favorable_reviews_by_users.items():
            # 遍历每个项集
            for itemset in k_1_itemsets:
                if itemset.issubset(reviews):  # 判断itemset是否是用户喜欢的电影的子集
                    # 对用户喜欢的电影中除了这个子集的电影进行遍历
                    for other_reviewed_movie in reviews - itemset:
                        # 将该电影并入项集中
                        current_superset = itemset | frozenset(
                            {other_reviewed_movie})
                        counts[current_superset] += 1  # 这个项集的支持度+1
        # 返回元素数目+1的项集和数量
        res = dict([(itemset, frequency) for itemset,
                                             frequency in counts.items() if frequency >= min_support])
        return res


    for k in range(2, 20):
        cur_frequent_itemsets = find_frequent_itemsets(
            favorable_reviews_by_users, frequent_itemsets[k - 1], min_support)
        frequent_itemsets[k] = cur_frequent_itemsets
        if len(cur_frequent_itemsets) == 0:
            print("Did not find any frequent itemsets of length {}".format(k))
            sys.stdout.flush()  # 将缓冲区内容输出到终端，不宜多用，输出操作带来的计算开销会拖慢程序运行速度
            break
        else:
            print(
                "I found {} frequent itemsets of length {}".format(
                    len(cur_frequent_itemsets), k))
            sys.stdout.flush()
    del frequent_itemsets[1]

    # 规则形式：如果用户喜欢前提中的所有电影，那么他们也会喜欢结论中的电影
    candidate_rules = []
    for itemset_length, itemset_counts in frequent_itemsets.items():
        for itemset in itemset_counts.keys():
            for conclusion in itemset:
                premise = itemset - {conclusion}
                candidate_rules.append((premise, conclusion))
    print(candidate_rules[:5])

    # 计算置信度
    correct_counts = defaultdict(int)
    incorrect_counts = defaultdict(int)

    # 遍历每一个用户，获取其喜欢的电影
    for user, reviews in favorable_reviews_by_users.items():
        # 遍历每个规则
        for candidate_rule in candidate_rules:
            # 获取规则的条件和结论
            premise, conclusion = candidate_rule
            # 如果条件是喜欢电影的子集（条件成立）
            if premise.issubset(reviews):
                # 如果用户也喜欢结论的电影
                if conclusion in reviews:
                    correct_counts[candidate_rule] += 1
                else:
                    incorrect_counts[candidate_rule] += 1
    # 计算置信度，结论发生的次数除以条件发生的次数
    rule_confidence = {
        candidate_rule: correct_counts[candidate_rule] /
        float(
            correct_counts[candidate_rule] +
            incorrect_counts[candidate_rule]) for candidate_rule in candidate_rules}

    # 给置信度排序
    sorted_confidence = sorted(
        rule_confidence.items(),
        key=itemgetter(1),
        reverse=True)
    for index in range(5):
        print("Rule #{}".format(index + 1))
        (premise, conclusion) = sorted_confidence[index][0]
        print(
            "Rule: If a person recommends {} they will also recommand {}".format(
                premise,
                conclusion))
        print(
            "- Confidence: {0:.3f}".format(rule_confidence[(premise, conclusion)]))
        print("--------------------")

    movie_name_data = pd.read_csv(
        "ml-100k/u.item",
        delimiter='|',
        header=None,
        encoding="mac-roman")
    movie_name_data.columns = [
        'MovieID',
        'Title',
        'Release Date',
        'Video Release',
        'IMDB',
        '<UNK>',
        'Action',
        'Adventure',
        'Animation',
        "Children's",
        'Comedy',
        'Crime',
        'Documentary',
        'Drama',
        'Fantasy',
        'Film-Noir',
        'Horror',
        'Musical',
        'Mystery',
        'Romance',
        'Sci-Fi',
        'Thriller',
        'War',
        'Western']

    for index in range(5):
        print('Rule #{0}'.format(index + 1))
        (premise, conclusion) = sorted_confidence[index][0]
        premise_names = ', '.join(get_movie_name(idx) for idx in premise)
        conclusion_name = get_movie_name(conclusion)
        print(
            'Rule: if a person recommends {0} they will also recommend {1}'.format(
                premise_names,
                conclusion_name))
        print(
            ' - Confidence: {0:.3f}'.format(rule_confidence[(premise, conclusion)]))
        print("--------------------")
    print()
    print()
    # 评估测试
    test_dataset = all_ratings[~all_ratings['UserID'].isin(range(200))]
    test_favorable = test_dataset[test_dataset["Favorable"]]
    test_favorable_by_users = dict((k, frozenset(v.values))
                                   for k, v in test_favorable.groupby("UserID")["MovieID"])

    correct_counts = defaultdict(int)
    incorrect_counts = defaultdict(int)
    for user, reviews in test_favorable_by_users.items():
        for candidate_rule in candidate_rules:
            premise, conclusion = candidate_rule
            if premise.issubset(reviews):
                if conclusion in reviews:
                    correct_counts[candidate_rule] += 1
                else:
                    incorrect_counts[candidate_rule] += 1

    test_confidence = {
        candidate_rule: correct_counts[candidate_rule] /
        float(
            correct_counts[candidate_rule] +
            incorrect_counts[candidate_rule]) for candidate_rule in rule_confidence}
    for index in range(5):
        print("Rule #{0}".format(index + 1))
        (premise, conclusion) = sorted_confidence[index][0]
        premise_names = ", ".join(get_movie_name(idx) for idx in premise)
        conclusion_name = get_movie_name(conclusion)
        print(
            'Rule: if a person recommends {0} they will also recommend {1}'.format(
                premise_names,
                conclusion_name))
        print(
            ' - Confidence: {0:.3f}'.format(rule_confidence[(premise, conclusion)]))
        print("--------------------")
