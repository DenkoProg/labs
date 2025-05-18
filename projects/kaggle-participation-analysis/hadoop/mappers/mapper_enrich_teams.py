#!/usr/bin/env python3
import csv, sys, json

comp_map = {}

try:
    with open("competitions.csv", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(line.replace("\x00", "") for line in f)
        for row in reader:
            if row.get("Id") and row.get("Title"):
                comp_map[row["Id"]] = {
                    "CompetitionTitle": row["Title"],
                    "HostSegmentTitle": row.get("HostSegmentTitle", ""),
                    "EnabledDate": row.get("EnabledDate", ""),
                    "DeadlineDate": row.get("DeadlineDate", ""),
                }
except Exception as e:
    print(f"ERROR: Failed to read competitions.csv - {e}", file=sys.stderr)
    sys.exit(1)

# Read teams.csv from stdin
reader = csv.DictReader(sys.stdin)
for row in reader:
    comp_info = comp_map.get(row["CompetitionId"], {})
    if not comp_info:
        continue  # skip if competition not found

    merged = {**row, **comp_info}
    payload = {"tag": "TEAM", "data": merged}
    print(f"{row['Id']}\t{json.dumps(payload)}")
