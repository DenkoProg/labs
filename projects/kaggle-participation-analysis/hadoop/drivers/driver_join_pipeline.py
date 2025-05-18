#!/usr/bin/env python3
import subprocess

# Path to your hadoop-streaming jar
JAR = "/opt/homebrew/Cellar/hadoop/3.4.1/libexec/share/hadoop/tools/lib/hadoop-streaming-3.4.1.jar"

# HDFS directories
INPUT = "/user/data/input"
OUT_TEAMS = "/user/data/out0_teams_enriched"
OUT_SUB_TEAM = "/user/data/out1_submissions_teams"
OUT_FINAL = "/user/data/out2_submissions_users"


def run_map_only(mapper, input_path, output_path, files=None):
    """Map-only job: ships `files` into each mapper and emits <key>\t<value>."""
    subprocess.run(["hdfs", "dfs", "-rm", "-r", "-f", output_path], check=True)
    cmd = [
        "hadoop",
        "jar",
        JAR,
        "-input",
        input_path,
        "-output",
        output_path,
        "-mapper",
        f"python3 {mapper}",
        "-numReduceTasks",
        "0",
        "-file",
        mapper,
    ]
    if files:
        for f in files:
            cmd += ["-file", f]
    subprocess.run(cmd, check=True)


def run(mapper, reducer, input_paths, output_path, files=None):
    """Classic map-reduce: ships mapper+reducer (and any extra files), takes multiple inputs."""
    subprocess.run(["hdfs", "dfs", "-rm", "-r", "-f", output_path], check=True)
    cmd = [
        "hadoop",
        "jar",
        JAR,
        "-mapper",
        f"python3 {mapper}",
        "-reducer",
        f"python3 {reducer}",
        "-file",
        mapper,
        "-file",
        reducer,
    ]
    if files:
        for f in files:
            cmd += ["-file", f]
    for path in input_paths:
        cmd += ["-input", path]
    cmd += ["-output", output_path]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    # 1) Enrich teams with competition metadata (map-side join)
    run_map_only(
        "mapper_enrich_teams.py",
        f"{INPUT}/teams.csv",
        OUT_TEAMS,
        files=[f"./data/competitions.csv"],
    )

    # 2) Join submissions ↔ enriched teams
    run(
        "mapper_join_submissions_teams.py",
        "reducer_join_submissions_teams.py",
        [f"{INPUT}/submissions.csv", OUT_TEAMS],
        OUT_SUB_TEAM,
    )

    # 3) Join (submissions+teams) ↔ users
    run(
        "mapper_join_submissions_users.py",
        "reducer_join_submissions_users.py",
        [OUT_SUB_TEAM, f"{INPUT}/users.csv"],
        OUT_FINAL,
    )

    print("✅ Join pipeline complete:", OUT_FINAL)
