#!/usr/bin/env python3
import sys

current_id = None
comp_data = None
joined = []

for line in sys.stdin:
    parts = line.strip().split("\t")
    key, tag = parts[0], parts[1]

    if key != current_id:
        if comp_data:
            for j in joined:
                print(f"{j[0]}\t{j[1]}\t{j[2]}\t{j[3]}\t{comp_data}")
        current_id = key
        comp_data = None
        joined = []

    if tag == "COMP":
        comp_data = parts[2]
    elif tag == "JOINED":
        joined.append(parts[2:6])

if comp_data:
    for j in joined:
        print(f"{j[0]}\t{j[1]}\t{j[2]}\t{j[3]}\t{comp_data}")
