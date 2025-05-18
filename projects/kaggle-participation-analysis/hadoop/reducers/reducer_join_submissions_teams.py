#!/usr/bin/env python3
import json
import sys


def flush_block(team, subs):
    if team is None:
        return
    for s in subs:
        merged = {**s, **team}
        # Emit keyed by UserId for next stage
        print(f"{merged['UserId']}\t{json.dumps(merged)}")


current_key = None
team_record = None
sub_list = []

for line in sys.stdin:
    line = line.rstrip("\n")
    key, body = line.split("\t", 1)
    rec = json.loads(body)
    tag = rec["tag"]
    data = rec["data"]

    if current_key and key != current_key:
        flush_block(team_record, sub_list)
        team_record = None
        sub_list = []

    current_key = key
    if tag == "TEAM":
        team_record = data
    else:  # SUB
        sub_list.append(data)

# flush last group
if current_key:
    flush_block(team_record, sub_list)
