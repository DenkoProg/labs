#!/usr/bin/env python3
import sys

current_user_id = None
user_data = None
submissions = []


def emit_all():
    if user_data:
        for sub, membership in submissions:
            print(f"{current_user_id}\t{sub}\t{membership}\t{user_data}")


for line in sys.stdin:
    parts = line.strip().split("\t", 3)
    if len(parts) < 3:
        continue

    user_id = parts[0]
    tag = parts[1]

    if user_id != current_user_id:
        emit_all()
        current_user_id = user_id
        user_data = None
        submissions = []

    if tag == "USER":
        user_data = parts[2]
    elif tag == "SUB" and len(parts) == 4:
        submissions.append((parts[2], parts[3]))

emit_all()
