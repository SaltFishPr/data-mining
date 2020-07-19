#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import twitter

consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""
authorization = twitter.OAuth(
    access_token, access_token_secret, consumer_key, consumer_secret
)

if __name__ == "__main__":
    output_filename = "python_tweets.json"

    t = twitter.Twitter(auth=authorization)

    n_output = 0

    with open(output_filename, "a") as output_file:
        search_results = t.search.tweets(q="python", count=100)["statuses"]
        for tweet in search_results:
            if "text" in tweet:
                output_file.write(json.dumps(tweet))
                output_file.write("\n\n")
                n_output += 1

    print("Saved {} entries".format(n_output))
    pass
