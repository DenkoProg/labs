import os
import sys

import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.dates import days_ago

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.datasets import (
    dim_customer,
    dim_date,
    dim_product,
    dim_shipmode,
    sales_fact,
)
from config.logger import logger
from config.settings import config
from managers.db import db_manager

SQL_DIR = "sql"

default_args = {
    "owner": "airflow",
}

dag = DAG(
    dag_id="etl_sales_pipeline",
    description="Complete ETL: loading, aggregation, saving",
    default_args=default_args,
    start_date=days_ago(1),
    # Changed to be triggered by all datasets
    schedule=[dim_customer, dim_date, dim_product, dim_shipmode, sales_fact],
    catchup=False,
)

datasets = [
    {
        "name": "dim_customer",
        "csv": "dim_customer.csv",
        "sql": "create_dim_customer.sql",
        "dataset": dim_customer,
    },
    {
        "name": "dim_product",
        "csv": "dim_product.csv",
        "sql": "create_dim_product.sql",
        "dataset": dim_product,
    },
    {
        "name": "dim_shipmode",
        "csv": "dim_shipmode.csv",
        "sql": "create_dim_shipmode.sql",
        "dataset": dim_shipmode,
    },
    {
        "name": "dim_date",
        "csv": "dim_date.csv",
        "sql": "create_dim_date.sql",
        "dataset": dim_date,
    },
    {
        "name": "sales_fact",
        "csv": "sales_fact.csv",
        "sql": "create_sales_fact.sql",
        "dataset": sales_fact,
    },
]

aggregations = [
    {"task_id": "aggregate_by_category", "sql": "aggregate_sales_by_category.sql"},
    {"task_id": "aggregate_by_region", "sql": "aggregate_sales_by_region.sql"},
]


def load_csv_to_postgres(csv_file, table_name):
    def _load():
        logger.info(f"Loading {table_name} from {csv_file}")
        csv_path = os.path.join(config.DATA_OUTPUT_DIR, csv_file)

        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        engine = db_manager.get_engine()
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        logger.info(f"✅ Loaded {table_name} ({len(df)} rows)")

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
        inlets=[table["dataset"]],
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
        inlets=[sales_fact],
        dag=dag,
    )
    previous_task >> agg_task
    previous_task = agg_task
