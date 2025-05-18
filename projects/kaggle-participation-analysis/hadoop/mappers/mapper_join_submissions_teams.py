#!/usr/bin/env python3
import csv
import json
import sys

for line in sys.stdin:
    line = line.rstrip("\n")
    # --- detect whether this is a submission CSV row (no tab) or enriched-team TSV (has tab) ---
    if "\t" not in line:
        # submissions.csv line
        row = next(csv.DictReader([line]))
        team_id = row["TeamId"]
        sub_data = {
            "SubmissionId": row["Id"],
            "UserId": row["SubmittedUserId"],
            "Score": row["PublicScoreLeaderboardDisplay"],
            "SubmissionDate": row["SubmissionDate"],
        }
        payload = {"tag": "SUB", "data": sub_data}
    else:
        # teams_enriched part: key\tjson
        key, body = line.split("\t", 1)
        team_id = key
        team_rec = json.loads(body)["data"]
        payload = {"tag": "TEAM", "data": team_rec}

    print(f"{team_id}\t{json.dumps(payload)}")
