import subprocess

JAR = "/opt/homebrew/Cellar/hadoop/3.4.1/libexec/share/hadoop/tools/lib/hadoop-streaming-3.4.1.jar"
INPUT = "/user/data/input"
OUT0 = "/user/data/out0"
OUT1 = "/user/data/out1"
OUT2 = "/user/data/out2"
OUT3 = "/user/data/out3"


def run(mapper, reducer, input_path, output_path):
    subprocess.run(["hdfs", "dfs", "-rm", "-r", "-f", output_path])
    subprocess.run(
        [
            "hadoop",
            "jar",
            JAR,
            "-input",
            input_path,
            "-output",
            output_path,
            "-mapper",
            f"python3 {mapper}",
            "-reducer",
            f"python3 {reducer}",
            "-file",
            mapper,
            "-file",
            reducer,
        ]
    )


# run(
#     "mappers/mapper_join_submissions_teammemberships.py",
#     "reducers/reducer_join_submissions_teammemberships.py",
#     f"{INPUT}/submissions_teammemberships",
#     OUT0,
# )

run(
    "mappers/mapper_join_submissions_users.py",
    "reducers/reducer_join_submissions_users.py",
    OUT0,
    OUT1,
)

# run("mappers/mapper_join_teams.py", "reducers/reducer_join_teams.py", OUT1, OUT2)

# run("mappers/mapper_join_competitions.py", "reducers/reducer_join_competitions.py", OUT2, OUT3)

# print("✅ Join pipeline complete:", OUT3)
