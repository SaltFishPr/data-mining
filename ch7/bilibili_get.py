#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import time

import requests
import json


COOKIE = ""


def get_followings(mid, pn, ps):
    """
    获取关注用户信息
    :param mid: 用户ID
    :param pn: 页数
    :param ps: 每页数量
    """
    my_res = []
    try:
        headers = {
            "Connection": "keep-alive",
            "Host": "api.bilibili.com",
            "Referer": "https://space.bilibili.com/" + str(mid),
            "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36",
        }
        url = (
            "https://api.bilibili.com/x/relation/followings?vmid="
            + str(mid)
            + "&pn="
            + str(pn)
            + "&ps="
            + str(ps)
            + "&order=desc&jsonp=jsonp"
        )
        print("获取关注用户信息url:{}".format(url))
        req = requests.get(url, headers=headers, timeout=60)
        if req.status_code == 200:
            code = req.json()
            if code.get("data"):
                glist = code.get("data").get("list")
                for i in glist:
                    result = {"uname": i.get("uname"), "mid": i.get("mid")}
                    my_res.append(result)
            else:
                print("限制只访问前5页")

        else:
            print("获取关注用户信息失败 code:{}".format(req.status_code))

    except ConnectionError as e:
        print("ConnectionError网络异常", e.args)
    return my_res


def get_bilibili(root_id, deep_int):
    mid_user = [root_id]
    root_following = get_followings(root_id, 2, 50)
    users = {
        root_id: {"uname": "**Root**", "distance": 0, "followings": root_following}
    }

    for user_id in mid_user:
        user_info = users[user_id]
        distance = user_info["distance"] + 1
        if distance > deep_int:
            break
        for following in user_info["followings"]:
            uid = following["uid"]
            uname = following["uname"]
            # 如果已经搜索过这个用户
            if uid in mid_user:
                continue
            # 获得这个用户的关注列表
            following_list = get_followings(uid, 2, 50)
            # 将这个用户的信息加入用户列表
            mid_user.append(uid)
            users[uid] = {
                "uname": uname,
                "distance": distance,
                "followings": following_list,
            }
            time.sleep(2)
        return users


if __name__ == "__main__":
    res = get_bilibili(214582845, 5)
    with open("bilibili.txt", "w") as fout:
        json.dump(res, fout)
