from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
}

dag = DAG(
    dag_id="build_data_marts",
    description="Побудова вітрин даних раз на місяць",
    default_args=default_args,
    start_date=days_ago(1),
    schedule_interval="0 0 1 * *",  # Кожного 1 числа місяця в 00:00
    catchup=False,
    tags=["data_mart", "monthly", "aggregation"],
)

create_dwh_schema = PostgresOperator(
    task_id="create_dwh_schema",
    postgres_conn_id="postgres_default",
    sql="""
    CREATE SCHEMA IF NOT EXISTS dwh;
    """,
    dag=dag,
)

create_sales_by_category_month = PostgresOperator(
    task_id="create_sales_by_category_month",
    postgres_conn_id="postgres_default",
    sql="sql/sales_by_category_month.sql",
    dag=dag,
)

create_sales_by_region_month = PostgresOperator(
    task_id="create_sales_by_region_month",
    postgres_conn_id="postgres_default",
    sql="sql/sales_by_region_month.sql",
    dag=dag,
)

create_top_customers_by_profit = PostgresOperator(
    task_id="create_top_customers_by_profit",
    postgres_conn_id="postgres_default",
    sql="sql/top_customers_by_profit.sql",
    dag=dag,
)

(
    create_dwh_schema
    >> [create_sales_by_category_month, create_sales_by_region_month]
    >> create_top_customers_by_profit
)
