from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from sqlalchemy import create_engine
import pandas as pd
import os

DB_CONN = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"
CSV_DIR = "data"
SQL_DIR = "sql"

default_args = {
    "owner": "airflow",
}

dag = DAG(
    dag_id="etl_sales_pipeline",
    description="Повний ETL: завантаження, агрегація, збереження",
    default_args=default_args,
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
)

datasets = [
    {
        "name": "dim_customer",
        "csv": "dim_customer.csv",
        "sql": "create_dim_customer.sql",
    },
    {"name": "dim_product", "csv": "dim_product.csv", "sql": "create_dim_product.sql"},
    {
        "name": "dim_shipmode",
        "csv": "dim_shipmode.csv",
        "sql": "create_dim_shipmode.sql",
    },
    {"name": "dim_date", "csv": "dim_date.csv", "sql": "create_dim_date.sql"},
    {"name": "sales_fact", "csv": "sales_fact.csv", "sql": "create_sales_fact.sql"},
]

aggregations = [
    {"task_id": "aggregate_by_category", "sql": "aggregate_sales_by_category.sql"},
    {"task_id": "aggregate_by_region", "sql": "aggregate_sales_by_region.sql"},
]


def load_csv_to_postgres(csv_file, table_name):
    def _load():
        df = pd.read_csv(os.path.join(CSV_DIR, csv_file))
        engine = create_engine(DB_CONN)
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        print(f"✅ Завантажено {table_name} ({len(df)} рядків)")

    return _load


previous_task = None
for table in datasets:
    create_task = PostgresOperator(
        task_id=f"create_{table['name']}",
        postgres_conn_id="postgres_default",
        sql=f"{SQL_DIR}/{table['sql']}",
        dag=dag,
    )

    load_task = PythonOperator(
        task_id=f"load_{table['name']}",
        python_callable=load_csv_to_postgres(table["csv"], table["name"]),
        dag=dag,
    )

    if previous_task:
        previous_task >> create_task
    create_task >> load_task
    previous_task = load_task

for agg in aggregations:
    agg_task = PostgresOperator(
        task_id=agg["task_id"],
        postgres_conn_id="postgres_default",
        sql=f"{SQL_DIR}/{agg['sql']}",
        dag=dag,
    )
    previous_task >> agg_task
    previous_task = agg_task
