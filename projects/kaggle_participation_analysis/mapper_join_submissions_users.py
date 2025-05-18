#!/usr/bin/env python3
import sys
import csv


def is_user_csv_line(row):
    return row[0].isdigit() and len(row) > 3


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue

    if "\t" in line:
        # This is from out0 (joined submission+membership)
        parts = line.split("\t")
        if len(parts) == 3:
            user_id, submission, membership = parts
            print(f"{user_id}\tSUB\t{submission}\t{membership}")
    else:
        # Might be a CSV line (users.csv)
        try:
            row = next(csv.reader([line]))
            if is_user_csv_line(row):
                user_id = row[0]
                print(f"{user_id}\tUSER\t{','.join(row)}")
        except Exception:
            continue
