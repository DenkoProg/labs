#!/usr/bin/env python3
import json
import sys


def flush_block(user, subs):
    if user is None:
        return
    for s in subs:
        merged = {**s, **user}
        # final denormed record
        print(json.dumps(merged))


current_key = None
user_record = None
sub_list = []

for line in sys.stdin:
    body = line.rstrip("\n")
    key, json_str = body.split("\t", 1)
    rec = json.loads(json_str)
    tag = rec["tag"]
    data = rec["data"]

    if current_key and key != current_key:
        flush_block(user_record, sub_list)
        user_record = None
        sub_list = []

    current_key = key
    if tag == "USR":
        user_record = data
    else:  # SUB
        sub_list.append(data)

# flush last key
if current_key:
    flush_block(user_record, sub_list)
