#!/usr/bin/env python3
import sys

current_id = None
team_data = None
entries = []

for line in sys.stdin:
    parts = line.strip().split("\t")
    key, tag = parts[0], parts[1]

    if key != current_id:
        if team_data:
            for entry in entries:
                print(f"{entry[0]}\t{entry[1]}\t{entry[2]}\t{team_data}")
        current_id = key
        team_data = None
        entries = []

    if tag == "TEAM":
        team_data = parts[2]
    elif tag == "JOINED":
        entries.append(parts[2:5])

if team_data:
    for entry in entries:
        print(f"{entry[0]}\t{entry[1]}\t{entry[2]}\t{team_data}")
