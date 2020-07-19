# !/usr/bin/env python
# author: Salt Fish
# -*- coding: utf-8 -*-
import json
import os
import requests
import getpass
import time


CLIENT_ID = "Lj4ADuFxVdsZaw"
CLIENT_SECRET = "1HDkFXNFa5CWpmXcyu6jBNo0Kko"
USER_AGENT = "python:saltfish_ch10 (by /u/saltfishpr)"
USERNAME = "saltfishpr"
PASSWORD = "kR#uW3KWi466C8y"

proxies = {"http": "socks5://127.0.0.1:5760", "https": "socks5://127.0.0.1:5760"}


def login(username, password):
    if password is None:
        password = getpass.getpass(
            "Enter reddit password for user {}: ".format(username)
        )
    headers = {"User-Agent": USER_AGENT}
    # 使用凭据设置身份验证对象
    client_auth = requests.auth.HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)
    post_data = {"grant_type": "password", "username": username, "password": password}
    response = requests.post(
        "https://www.reddit.com/api/v1/access_token",
        proxies=proxies,
        auth=client_auth,
        data=post_data,
        headers=headers,
    )
    return response.json()


if __name__ == "__main__":
    # token = login(USERNAME, PASSWORD)
    # print(token)

    token = {
        "access_token": "470243193994-OXyind4NFJB5fM-nN03UkLZ69SA",
        "token_type": "bearer",
        "expires_in": 3600,
        "scope": "*",
    }

    def get_links(subreddit, token, n_pages=5):
        stories = []
        after = None
        for page_number in range(n_pages):
            # 进行调用之前等待，以避免超过API限制
            print("等待2s...")
            time.sleep(2)
            # Setup headers and make call, just like in the login function
            headers = {
                "Authorization": "bearer {}".format(token["access_token"]),
                "User-Agent": USER_AGENT,
            }
            url = "https://oauth.reddit.com/r/{}/top?limit=100".format(subreddit)
            if after:
                # Append cursor for next page, if we have one
                url += "&after={}".format(after)
            while True:
                try:
                    response = requests.get(
                        url, proxies=proxies, headers=headers, timeout=10
                    )
                    result = response.json()
                    print(response)
                    # Get the new cursor for the next loop
                    after = result["data"]["after"]
                except:
                    print("requests出错等待...")
                    time.sleep(2)
                else:
                    break
            # 将所有新闻项添加到story列表中
            for story in result["data"]["children"]:
                stories.append(
                    (
                        story["data"]["title"],
                        story["data"]["url"],
                        story["data"]["score"],
                    )
                )
        return stories

    stories = get_links("worldnews", token)
    base_folder = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(base_folder, "raw")
    with open(os.path.join(base_folder, "stories2.txt"), "w") as f:
        for link in stories:
            f.write(json.dumps(list(link)))
            f.write("\n")
