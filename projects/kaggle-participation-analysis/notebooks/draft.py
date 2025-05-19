!pip3 install pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("KaggleParticipation").config(
    "spark.driver.memory", "4g"
).config("spark.executor.memory", "4g").getOrCreate()

users = spark.read.csv(
    "/Users/denys.koval/Labs/projects/kaggle-participation-analysis/data/users.csv",
    header=True,
    inferSchema=True,
)
competitions = spark.read.csv(
    "/Users/denys.koval/Labs/projects/kaggle-participation-analysis/data/competitions.csv",
    header=True,
    inferSchema=True,
)
teams = spark.read.csv(
    "/Users/denys.koval/Labs/projects/kaggle-participation-analysis/data/teams.csv",
    header=True,
    inferSchema=True,
)
submissions = spark.read.csv(
    "/Users/denys.koval/Labs/projects/kaggle-participation-analysis/data/submissions.csv",
    header=True,
    inferSchema=True,
)

for df, name in [
    (users, "users"),
    (competitions, "competitions"),
    (teams, "teams"),
    (submissions, "submissions"),
]:
    print(f"—— schema of {name} ——")
    df.printSchema()

submissions_df = submissions.alias("s")
teams_df = teams.alias("t")
users_df = users.alias("u")

from pyspark.sql.functions import col, to_date, regexp_replace, trim


clean_comp = (
    spark.read.option("header", True)
    .option("multiLine", True)
    .option("escape", '"')
    .csv(
        "/Users/denys.koval/Labs/projects/kaggle-participation-analysis/data/competitions.csv"
    )
)

clean_comp = clean_comp.withColumn(
    "SlugClean", regexp_replace(col("Slug"), "<[^>]*>", "")
)
clean_comp = clean_comp.withColumn("SlugClean", trim(col("SlugClean")))

clean_comp_filtered = (
    clean_comp.filter(col("Id").rlike("^[0-9]+$"))
    .withColumn("CompetitionId", col("Id"))
    .withColumn("DeadlineDate", to_date("DeadlineDate", "yyyy-MM-dd"))
    .select("CompetitionId", "SlugClean", "DeadlineDate")
    .dropDuplicates(["CompetitionId"])
    .alias("c")
)

clean_comp_filtered.show(20, truncate=False)

sub_with_team = submissions.alias("s").join(
    teams.alias("t"), col("s.TeamId") == col("t.Id"), "left"
)

sub_with_team_comp = sub_with_team.join(
    clean_comp_filtered.alias("c"),
    col("t.CompetitionId") == col("c.CompetitionId"),
    "left",
)

sub_with_user = sub_with_team_comp.join(
    users.alias("u"), col("s.SubmittedUserId") == col("u.Id"), "left"
)

sub_with_user.select(
    col("s.Id").alias("SubmissionId"),
    col("u.UserName"),
    col("t.TeamName"),
    col("c.SlugClean").alias("Slug"),
    col("s.SubmissionDate"),
).show(5, truncate=False)

from pyspark.sql.functions import row_number
from pyspark.sql.window import Window

competition_dim = (
    clean_comp_filtered
    .withColumn("CompetitionKey", row_number().over(Window.orderBy("CompetitionId")))
    .alias("cd")
)

competition_dim.printSchema()
competition_dim.show(5, truncate=False)

from pyspark.sql.functions import dayofmonth, month, year, hour, date_format

user_dim = (
    users.select("Id", "UserName", "Country", "PerformanceTier")
    .dropDuplicates()
    .withColumn("UserKey", row_number().over(Window.orderBy("Id")))
    .alias("ud")
)
user_dim.show(5, truncate=False)

team_dim = (
    teams.select("Id", "TeamName", "IsBenchmark")
    .dropDuplicates()
    .withColumn("TeamKey", row_number().over(Window.orderBy("Id")))
    .alias("td")
)
team_dim.show(5, truncate=False)

time_dim = (
    submissions.alias("s")
    .select("SubmissionDate")
    .dropDuplicates()
    .withColumn("TimeKey", to_date("SubmissionDate", "MM/dd/yyyy"))
    .filter(col("TimeKey").isNotNull())
    .withColumn("Day", dayofmonth("TimeKey"))
    .withColumn("Month", month("TimeKey"))
    .withColumn("Year", year("TimeKey"))
    .withColumn("Hour", hour("SubmissionDate"))
    .withColumn("DayOfWeek", date_format("TimeKey", "EEEE"))
    .select("TimeKey", "Day", "Month", "Year", "Hour", "DayOfWeek")
    .alias("tdim")
)
time_dim.show(5, truncate=False)

facts = (
    sub_with_user.join(user_dim, col("s.SubmittedUserId") == col("ud.Id"), "left")
    .join(team_dim, col("s.TeamId") == col("td.Id"), "left")
    .join(competition_dim, col("t.CompetitionId") == col("cd.CompetitionId"), "left")
    .join(
        time_dim,
        to_date(col("s.SubmissionDate"), "MM/dd/yyyy") == col("tdim.TimeKey"),
        "left",
    )
    .select(
        col("s.Id").alias("SubmissionId"),
        col("tdim.TimeKey"),
        col("ud.UserKey"),
        col("td.TeamKey"),
        col("cd.CompetitionKey"),
        col("s.PublicScoreLeaderboardDisplay"),
        col("s.PrivateScoreLeaderboardDisplay"),
        col("s.IsAfterDeadline"),
    )
    .repartition(10)
)

user_dim.write.json(
    "/Users/denys.koval/Labs/projects/kaggle-participation-analysis/output/final/UserDim"
)
team_dim.write.json(
    "/Users/denys.koval/Labs/projects/kaggle-participation-analysis/output/final/TeamDim"
)
competition_dim.write.json(
    "/Users/denys.koval/Labs/projects/kaggle-participation-analysis/output/final/CompetitionDim"
)
time_dim.write.json(
    "/Users/denys.koval/Labs/projects/kaggle-participation-analysis/output/final/TimeDim"
)
facts.write.mode("overwrite").parquet(
    "/Users/denys.koval/Labs/projects/kaggle-participation-analysis/output/final/SubmissionFact"
)

df = spark.read.parquet(
    "/Users/denys.koval/Labs/projects/kaggle-participation-analysis/output/final/SubmissionFact/part-00000-39e907ca-3957-457c-acda-c96fba993cb0-c000.snappy.parquet"
)
df.show()
df.printSchema()

# user_dim = spark.read.json(
#     "/Users/denys.koval/Labs/projects/kaggle-participation-analysis/output/UserDim"
# )

# team_dim = spark.read.json(
#     "/Users/denys.koval/Labs/projects/kaggle-participation-analysis/output/TeamDim"
# )

# competition_dim = spark.read.json(
#     "/Users/denys.koval/Labs/projects/kaggle-participation-analysis/output/CompetitionDim"
# )

# time_dim = spark.read.json(
#     "/Users/denys.koval/Labs/projects/kaggle-participation-analysis/output/TimeDim"
# )

# facts = spark.read.parquet(
#     "/Users/denys.koval/Labs/projects/kaggle-participation-analysis/output/SubmissionFact"
# )

## User Dim
Metric: Average Public Score per User
import pyspark.sql.functions as F

agg_user = (
    facts.groupBy("UserKey")
    .agg(F.avg("PublicScoreLeaderboardDisplay").alias("AvgPublicScore"))
    .filter(F.col("AvgPublicScore") < 1)
)
agg_user.show()

## Team Dim
Metric: Total number of submissions per Team
agg_team = facts.groupBy("TeamKey").agg(F.count("*").alias("SubmissionCount"))
agg_team.show()

## Competition Dim
Metric: Average private score per competition
agg_comp = (
    facts.groupBy("CompetitionKey")
    .agg(
        F.avg("PrivateScoreLeaderboardDisplay")
        .alias("AvgPrivateScore")
    )
    .filter(F.col("AvgPrivateScore") < 1)
)
agg_comp.show()

## Time Dim
Metric: Number of submissions per year-month
from pyspark.sql.functions import year, month

agg_time = (
    facts.withColumn("Year", year("TimeKey"))
    .withColumn("Month", month("TimeKey"))
    .groupBy("Year", "Month")
    .agg(F.count("*").alias("SubmissionCount"))
    .orderBy("Year", "Month")
)
agg_time.show()
