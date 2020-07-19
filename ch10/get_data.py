# !/usr/bin/env python
# author: Salt Fish
# -*- coding: utf-8 -*-
import hashlib
import os
import requests
import json


proxies = {"http": "socks5://127.0.0.1:5760", "https": "socks5://127.0.0.1:5760"}


if __name__ == "__main__":
    base_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(base_folder, "raw")
    with open(os.path.join(base_folder, "stories1.txt"), "r") as f:
        temp = f.readlines()
    stories = []
    for l in temp:
        stories.append(json.loads(l))

    number_errors = 0
    for title, url, score in stories:
        print(url)
        output_filename = hashlib.md5(url.encode()).hexdigest()
        fullpath = os.path.join(data_folder, output_filename + ".txt")
        try:
            response = requests.get(url, proxies=proxies, timeout=10)
            data = response.text
            with open(fullpath, "w") as outf:
                outf.write(data)
        except Exception as e:
            number_errors += 1
            print("出错：{}".format(number_errors))
