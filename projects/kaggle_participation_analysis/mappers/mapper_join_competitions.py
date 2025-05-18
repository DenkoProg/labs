#!/usr/bin/env python3
import sys
import csv

for line in sys.stdin:
    parts = line.strip().split("\t")
    if len(parts) == 4:
        user_id, submission, user, team = parts
        competition_id = team.split(",")[1]
        print(f"{competition_id}\tJOINED\t{user_id}\t{submission}\t{user}\t{team}")

for line in sys.stdin:
    row = next(csv.reader([line.strip()]))
    if len(row) >= 1:
        print(f"{row[0]}\tCOMP\t{','.join(row)}")
