#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from email.parser import Parser  # 邮件解析器
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils import check_random_state  # 随机状态实例
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
import quotequail

enron_data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "maildir")
if __name__ == "__main__":
    p = Parser()

    def get_enron_corpus(
        num_authors=10,
        data_folder=enron_data_folder,
        min_docs_author=10,
        max_docs_author=100,
        random_state=None,
    ):
        random_state = check_random_state(random_state)
        # 随机对得到的邮箱列表进行排序
        email_addresses = sorted(os.listdir(data_folder))
        random_state.shuffle(email_addresses)

        documents = []
        classes = []
        author_num = 0
        authors = {}

        for user in email_addresses:
            users_email_folder = os.path.join(data_folder, user)
            mail_folders = [
                os.path.join(users_email_folder, subfolder)
                for subfolder in os.listdir(users_email_folder)
                if "sent" in subfolder
            ]
            try:
                authored_emails = [
                    open(
                        os.path.join(mail_folder, email_filename), encoding="cp1252"
                    ).read()
                    for mail_folder in mail_folders
                    for email_filename in os.listdir(mail_folder)
                ]
            except IsADirectoryError:
                continue
            # 获得至少十封邮件
            if len(authored_emails) < min_docs_author:
                continue
            # 最多获取前100封邮件
            if len(authored_emails) > max_docs_author:
                authored_emails = authored_emails[:max_docs_author]
            # 解析邮件，获取邮件内容
            contents = [p.parsestr(email)._payload for email in authored_emails]
            documents.extend(contents)
            # 将发件人添加到类列表中，每封邮件添加一次
            classes.extend([author_num] * len(authored_emails))
            # 记录收件人编号，再把编号+1
            authors[user] = author_num
            author_num += 1
            # 收件人数量达到设置的值跳出循环
            if author_num >= num_authors or author_num >= len(email_addresses):
                break
        # 返回邮件数据集以及收件人字典
        return documents, np.array(classes), authors

    documents, classes, authors = get_enron_corpus(
        data_folder=enron_data_folder, random_state=14
    )

    def remove_replies(email_contents):
        r = quotequail.unwrap(email_contents)
        if r is None:
            return email_contents
        if "text_top" in r:
            return r["text_top"]
        elif "text" in r:
            return r["text"]
        return email_contents

    documents = [remove_replies(document) for document in documents]

    parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
    svr = SVC()
    grid = GridSearchCV(svr, parameters)
    pipeline = Pipeline(
        [
            (
                "feature_extraction",
                CountVectorizer(analyzer="char", ngram_range=(3, 3)),
            ),
            ("classifier", grid),
        ]
    )
    scores = cross_val_score(pipeline, documents, classes, scoring="f1_macro")
    print("Score: {:.3f}".format(np.mean(scores)))

    training_documents, test_documents, y_train, y_test = train_test_split(
        documents, classes, random_state=14
    )
    pipeline.fit(training_documents, y_train)
    y_pred = pipeline.predict(test_documents)
    print(pipeline.named_steps["classifier"].best_params_)

    cm = confusion_matrix(y_test, y_pred)
    cm = cm / cm.astype(np.float).sum(axis=1)

    sorted_authors = sorted(authors.keys(), key=lambda x: authors[x])
    plt.figure(figsize=(20, 20))
    plt.imshow(cm, cmap="Blues")
    tick_marks = np.arange(len(sorted_authors))
    plt.xticks(tick_marks, sorted_authors)
    plt.yticks(tick_marks, sorted_authors)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()
