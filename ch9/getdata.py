#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
import os
import time
from collections import defaultdict

titles = {
    "burton": [4657, 2400, 5760, 6036, 7111, 8821, 18506, 4658, 5761, 6886, 7113],
    "dickens": [
        24022,
        1392,
        1414,
        1467,
        2324,
        580,
        786,
        888,
        963,
        27924,
        1394,
        1415,
        15618,
        25985,
        588,
        807,
        914,
        967,
        30127,
        1400,
        1421,
        16023,
        28198,
        644,
        809,
        917,
        968,
        1023,
        1406,
        1422,
        17879,
        30368,
        675,
        810,
        924,
        98,
        1289,
        1413,
        1423,
        17880,
        32241,
        699,
        821,
        927,
    ],
    "doyle": [
        2349,
        11656,
        1644,
        22357,
        2347,
        290,
        34627,
        5148,
        8394,
        26153,
        12555,
        1661,
        23059,
        2348,
        294,
        355,
        5260,
        8727,
        10446,
        126,
        17398,
        2343,
        2350,
        3070,
        356,
        5317,
        903,
        10581,
        13152,
        2038,
        2344,
        244,
        32536,
        423,
        537,
        108,
        139,
        2097,
        2345,
        24951,
        32777,
        4295,
        7964,
        11413,
        1638,
        21768,
        2346,
        2845,
        3289,
        439,
        834,
    ],
    "gaboriau": [1748, 1651, 2736, 3336, 4604, 4002, 2451, 305, 3802, 547],
    "nesbit": [34219, 23661, 28804, 4378, 778, 20404, 28725, 33028, 4513, 794],
    "tarkington": [
        1098,
        15855,
        1983,
        297,
        402,
        5798,
        8740,
        980,
        1158,
        1611,
        2326,
        30092,
        483,
        5949,
        8867,
        13275,
        18259,
        2595,
        3428,
        5756,
        6401,
        9659,
    ],
    "twain": [
        1044,
        1213,
        245,
        30092,
        3176,
        3179,
        3183,
        3189,
        74,
        86,
        1086,
        142,
        2572,
        3173,
        3177,
        3180,
        3186,
        3192,
        76,
        91,
        119,
        1837,
        2895,
        3174,
        3178,
        3181,
        3187,
        3432,
        8525,
    ],
}

assert len(titles) == 7

assert len(titles["tarkington"]) == 22
assert len(titles["dickens"]) == 44
assert len(titles["nesbit"]) == 10
assert len(titles["doyle"]) == 51
assert len(titles["twain"]) == 29
assert len(titles["burton"]) == 11
assert len(titles["gaboriau"]) == 10

url_base = "http://www.gutenberg.org/files/"
url_format = "{url_base}{id}/{id}-0.txt"

# 修复URL
url_fix_format = "http://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

fiexes = defaultdict(list)
# fixes = {}
# fixes[4657] = 'http://www.gutenberg.org/cache/epub/4657/pg4657.txt'

# make parent folder if not exists
# data_folder = os.path.join(os.path.expanduser('~'),'Data','books') #
# 这是在用户user目录中存储
data_folder = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    print(data_folder)

    for author in titles:
        print("Downloading titles from", author)
        # make author's folder if not exists
        author_folder = os.path.join(data_folder, author)
        if not os.path.exists(author_folder):
            os.makedirs(author_folder)
        # download each title to this folder
        for bookid in titles[author]:
            # if bookid in fixes:
            #     print(' - Applying fix to book with id', bookid)
            #     url = fixes[bookid]
            # else:
            #     print(' - Getting book with id', bookid)
            #     url = url_format.format(url_base=url_base, id=bookid)

            url = url_format.format(url_base=url_base, id=bookid)
            print(" - ", url)
            filename = os.path.join(author_folder, "%s.txt" % bookid)
            if os.path.exists(filename):
                print(" - File already exists, skipping")
            else:
                r = requests.get(url)
                if r.status_code == 404:
                    print("url 404:", author, bookid, "add to fixes list")
                    fiexes[author].append(bookid)
                else:
                    txt = r.text
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(txt)
                time.sleep(1)
    print("Download complete")

    print("开始下载修复列表")
    for author in fiexes:
        print("开始下载<%s>的作品" % author)
        author_folder = os.path.join(data_folder, author)
        if not os.path.exists(author_folder):
            os.makedirs(author_folder)

        for bookid in fiexes[author]:
            filename = os.path.join(author_folder, "%s.txt" % bookid)
            if os.path.exists(filename):
                print("文件已经下载，跳过")
            else:
                url_fix = url_fix_format.format(id=bookid)
                print(" - ", url_fix)
                r = requests.get(url_fix)
                if r.status_code == 404:
                    print("又出错了！", author, bookid)
                else:
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(r.text)
                time.sleep(1)
    print("修复列表下载完毕")
