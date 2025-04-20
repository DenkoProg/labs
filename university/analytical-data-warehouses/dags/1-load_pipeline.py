import os
import sys
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.dates import days_ago

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.settings import config
from managers.db import db_manager
from config.logger import logger

# Constants
SQL_DIR = "sql"

default_args = {
    "owner": "airflow",
}

dag = DAG(
    dag_id="load_csv_to_postgres",
    default_args=default_args,
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    description="Load dimensional tables and fact table into PostgreSQL from CSV",
)

tables = [
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


for table in tables:
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

    create_task >> load_task
