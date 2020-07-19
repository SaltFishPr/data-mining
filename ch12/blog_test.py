# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @author   : saltfish
# @Filename : blog_test.py
# @Date     : 2020-03-28
import os
import re
from mrjob.job import MRJob


data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blogs")

# 查看数据集内容
# file_name = os.path.join(data_folder, "1005545.male.25.Engineering.Sagittarius.xml")
# all_posts = []
# with open(file_name, "r") as inf:
#     post_start = False
#     post = []
#     for line in inf:
#         line = line.strip()
#         if line == "<post>":
#             post_start = True
#         elif line == "</post>":
#             post_start = False
#             all_posts.append("\n".join(post))
#             post = []
#         elif post_start:
#             post.append(line)
# print(all_posts[:2])
# print(len(all_posts))
# """
# ["\n\nB-Logs: The Business Blogs Paradox    urlLink HindustanTimes.com  discusses the effects of technology and blogs in particular. According to the article, Blogs are 'a direct, one-to-many vehicle for communicating ideas'. What makes them disruptive in business application is that they allow businesses - which are after all human -- to communicate with a human 'real voice'. On the other hand,  urlLink Webpronews.com  discusses the idea of corporate newsletter publishing via blogs. I found the idea very pragmatic and futuristic. Way to go !!\n\n", '\n\nBohemian Rhapsody : Is it??   I have just come back to my room from the  urlLink IIT  coffee shack. The place, where we have shared laughters and joys, sat for long hours over countless cups of coffee and maggi and chat sessions. Where we have shared our thoughts and fought over stupid topics discussing them passionately. I have just come back from that very place, but with thoughts and feelings so different from what it used to be.  Today the laughter has given way to a look of tension and frustration on everyone\'s face. "Its so frustrating. There is no job. I am applying everywhere. No vacancies." And then on the next table you hear, "I am planning to go abroad on a scholarship. The job scene is really bad and I don\'t think I will get a nice job. Going for higher studies is the only option left in front of me." And then a person comes who does have a job. Meeting him, I congratulate him. But he is too irritated with the job he got. The job is in a core engineering one, in a chemical MNC, but the pay is a punitive six thousand bucks; the only perk being free accomodation in some small village-town.  There are ruminations galore, about the uncertain futures. About the decisions made. About the hard work, before and during IIT years. And to end thus, with a degree in hand but no job to do justice to years of hardships and hardwork, and that too when you are the product of one of the best institutes in the country. It gets too disappointing.  I only wish those people, my dear friends, good luck !! May serendipity happen !!\n\n']
# 80
# """
class ExtractPosts(MRJob):
    post_start = False
    post = []

    def mapper(self, key, value: str):
        """
        映射函数，从文件中读取一行作为输入，最后生成一篇博客的所有内容
        :param key:
        :param value:
        :return:
        """
        filename = os.environ["map_input_file"]
        gender = filename.split(".")
        line = value.strip()


if __name__ == "__main__":
    filename = data_folder
    print(filename)
