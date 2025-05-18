#!/usr/bin/env python3
import csv
import json
import sys

for line in sys.stdin:
    line = line.rstrip("\n")
    if "\t" in line:
        # This is the TSV from previous reducer: UserId\tjson
        key, body = line.split("\t", 1)
        join_data = json.loads(body)
        payload = {"tag": "SUB", "data": join_data}
    else:
        # users.csv
        row = next(csv.DictReader([line]))
        user_id = row["Id"]
        user_data = {
            "UserName": row["UserName"],
            "DisplayName": row["DisplayName"],
            "Country": row["Country"],
            "PerformanceTier": row["PerformanceTier"],
        }
        payload = {"tag": "USR", "data": user_data}
        key = user_id

    print(f"{key}\t{json.dumps(payload)}")
