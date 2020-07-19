# !/usr/bin/env python
# author: Salt Fish
# -*- coding: utf-8 -*-
import os
from lxml import html, etree

if __name__ == "__main__":
    base_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(base_folder, "raw")
    text_output_folder = os.path.join(base_folder, "textonly")
    filenames = [
        os.path.join(data_folder, filename) for filename in os.listdir(data_folder)
    ]

    skip_node_types = ["script", "head", "style", etree.Comment]

    def get_text_from_file(filename):
        with open(filename, "r") as inf:
            html_tree = html.parse(inf)
        return get_text_from_node(html_tree.getroot())

    def get_text_from_node(node):
        if len(node) == 0:
            # 没有子节点，直接返回内容
            if node.text:
                return node.text
            else:
                return ""
        else:
            # 有子节点，递归调用得到内容
            results = (
                get_text_from_node(child)
                for child in node
                if child.tag not in skip_node_types
            )
        result = str.join("\n", (r for r in results if len(r) > 1))
        if len(result) >= 100:
            return result
        else:
            return ""

    for filename in os.listdir(data_folder):
        text = get_text_from_file(os.path.join(data_folder, filename))
        with open(os.path.join(text_output_folder, filename), "w") as outf:
            outf.write(text)
