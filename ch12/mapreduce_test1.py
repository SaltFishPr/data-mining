# !/usr/bin/env python
# author: Salt Fish
# -*- coding: utf-8 -*-
from collections import defaultdict
from sklearn.datasets import fetch_20newsgroups

from joblib import Parallel, delayed

import timeit


def map_word_count(document_id, document):
    counts = defaultdict(int)
    for word in document.split():
        counts[word] += 1
    for word in counts:
        yield word, counts[word]


def shuffle_words(results_generators):
    records = defaultdict(list)
    for results in results_generators:
        for word, count in results:
            records[word].append(count)
    for word in records:
        yield word, records[word]


def reduce_counts(word, list_of_counts):
    return word, sum(list_of_counts)


if __name__ == "__main__":
    dataset = fetch_20newsgroups(subset="train")
    documents = dataset.data
    # start = timeit.default_timer()
    # # 生成器，输出(单词，出现次数的键值对)
    # map_results = map(map_word_count, range(len(documents)), documents)
    # shuffle_results = shuffle_words(map_results)
    # reduce_results = [
    #     reduce_counts(word, list_of_counts) for word, list_of_counts in shuffle_results
    # ]
    # end = timeit.default_timer()
    # print(reduce_results[:5])
    # print(len(reduce_results))
    # print("----------", str(end - start))

    def map_word_count(document_id, document):
        counts = defaultdict(int)
        for word in document.split():
            counts[word] += 1
        return list(counts.items())

    start = timeit.default_timer()

    map_results = Parallel(n_jobs=4)(
        delayed(map_word_count)(i, document) for i, document in enumerate(documents)
    )

    shuffle_results = shuffle_words(map_results)
    reduce_results = [
        reduce_counts(word, list_of_counts) for word, list_of_counts in shuffle_results
    ]

    end = timeit.default_timer()

    print(reduce_results[:5])
    print(len(reduce_results))
    print("----------", str(end - start))
