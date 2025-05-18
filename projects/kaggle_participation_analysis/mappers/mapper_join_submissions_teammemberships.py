#!/usr/bin/env python3
import sys
import csv

for line in sys.stdin:
    row = next(csv.reader([line.strip()]))
    if not row or len(row) < 2:
        continue

    if len(row) >= 5 and row[0].isdigit():  # submissions.csv
        submitted_user_id = row[1]
        print(f"{submitted_user_id}\tSUB\t{','.join(row)}")
    elif len(row) >= 3 and row[0].isdigit():  # teammemberships.csv
        user_id = row[2]
        print(f"{user_id}\tMEM\t{','.join(row)}")
