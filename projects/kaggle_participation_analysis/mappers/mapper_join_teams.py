#!/usr/bin/env python3
import sys
import csv

for line in sys.stdin:
    parts = line.strip().split("\t")
    if len(parts) == 3:
        user_id, submission, user = parts
        team_id = submission.split(",")[2]
        print(f"{team_id}\tJOINED\t{user_id}\t{submission}\t{user}")

for line in sys.stdin:
    row = next(csv.reader([line.strip()]))
    if len(row) >= 3:
        print(f"{row[0]}\tTEAM\t{','.join(row)}")
