#!/usr/bin/env python3
import sys

current_user = None
membership = None
submissions = []

for line in sys.stdin:
    parts = line.strip().split("\t")
    if len(parts) < 3:
        continue

    user_id, tag, data = parts

    if user_id != current_user:
        if membership:
            for sub in submissions:
                print(f"{user_id}\t{sub}\t{membership}")
        current_user = user_id
        membership = None
        submissions = []

    if tag == "MEM":
        membership = data
    elif tag == "SUB":
        submissions.append(data)

if membership:
    for sub in submissions:
        print(f"{current_user}\t{sub}\t{membership}")
