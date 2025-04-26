from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.dates import days_ago
from config.datasets import (
    sales_fact,
)

dag = DAG(
    dag_id="aggregate_sales_pipeline",
    description="Агрегація даних по категоріях та регіонах",
    start_date=days_ago(1),
    schedule=[sales_fact],
    catchup=False,
)

agg_by_category = PostgresOperator(
    task_id="aggregate_by_category",
    postgres_conn_id="postgres_default",
    sql="sql/aggregate_sales_by_category.sql",
    inlets=[sales_fact],
    dag=dag,
)

agg_by_region = PostgresOperator(
    task_id="aggregate_by_region",
    postgres_conn_id="postgres_default",
    sql="sql/aggregate_sales_by_region.sql",
    inlets=[sales_fact],
    dag=dag,
)

agg_by_category >> agg_by_region
