from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import os
import pandas as pd

default_args = {
    "owner": "airflow",
}

dag = DAG(
    "data_pipeline",
    default_args=default_args,
    description="Pipeline: check file, load xlsx, split into dimensional tables",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
)

FILE_PATH = "data/Store.xlsx"
OUTPUT_DIR = "data/"


# === TASK 1 ===
def check_file_exists():
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"Файл {FILE_PATH} не знайдено!")
    print("Файл знайдено ✅")


# === TASK 2 ===
def load_and_split_data():
    df = pd.read_excel(FILE_PATH)

    # Dim_Date
    dim_date = df[["Order Date", "Ship Date"]].copy()
    dim_date = dim_date.drop_duplicates().reset_index(drop=True)
    dim_date["Date_ID"] = dim_date.index + 1
    dim_date = dim_date[["Date_ID", "Order Date", "Ship Date"]]
    dim_date.to_csv(f"{OUTPUT_DIR}/dim_date.csv", index=False)

    # Dim_Customer
    dim_customer = (
        df[
            [
                "Customer ID",
                "Customer Name",
                "Segment",
                "Country",
                "City",
                "State",
                "Postal Code",
                "Region",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    dim_customer.to_csv(f"{OUTPUT_DIR}/dim_customer.csv", index=False)

    # Dim_Product
    dim_product = (
        df[["Product ID", "Category", "Sub-Category", "Product Name"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    dim_product.to_csv(f"{OUTPUT_DIR}/dim_product.csv", index=False)

    # Dim_ShipMode
    dim_shipmode = df[["Ship Mode"]].drop_duplicates().reset_index(drop=True)
    dim_shipmode["ShipMode_ID"] = dim_shipmode.index + 1
    dim_shipmode = dim_shipmode[["ShipMode_ID", "Ship Mode"]]
    dim_shipmode.to_csv(f"{OUTPUT_DIR}/dim_shipmode.csv", index=False)

    # Sales_Fact
    sales_fact = df.copy()
    sales_fact = sales_fact.merge(dim_date, on=["Order Date", "Ship Date"], how="left")
    sales_fact = sales_fact.merge(dim_shipmode, on="Ship Mode", how="left")
    sales_fact = sales_fact[
        [
            "Order ID",
            "Date_ID",
            "Customer ID",
            "Product ID",
            "ShipMode_ID",
            "Sales",
            "Quantity",
            "Discount",
            "Profit",
        ]
    ]
    sales_fact.insert(0, "Fact_ID", range(1, len(sales_fact) + 1))
    sales_fact.to_csv(f"{OUTPUT_DIR}/sales_fact.csv", index=False)

    print("Таблиці створені та збережені у data/ ✅")


# === Operators ===
check_file_task = PythonOperator(
    task_id="check_file",
    python_callable=check_file_exists,
    dag=dag,
)

load_split_task = PythonOperator(
    task_id="load_and_split",
    python_callable=load_and_split_data,
    dag=dag,
)

check_file_task >> load_split_task
