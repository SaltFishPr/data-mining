#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import Counter

if __name__ == "__main__":
    s = """Three Rings for the Elven-kings under the sky, 
Seven for the Dwarf-lords in halls of stone, 
Nine for Mortal Men, doomed to die, 
One for the Dark Lord on his dark throne 
In the Land of Mordor where the Shadows lie. 
One Ring to rule them all, One Ring to find them, 
One Ring to bring them all and in the darkness bind them. 
In the Land of Mordor where the Shadows lie""".lower()
    words = s.split()
    c = Counter(words)
    # 输出出现次数最多的前5个词
    print(c.most_common(5))
