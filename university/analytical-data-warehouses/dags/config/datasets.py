import json
import os

from airflow import Dataset
from airflow.hooks.base import BaseHook

output_path = json.loads(BaseHook.get_connection("datasets_folder").extra)["path"]

dim_customer = Dataset(uri=os.path.join(output_path, "dim_customer.csv"))
dim_date = Dataset(uri=os.path.join(output_path, "dim_date.csv"))
dim_product = Dataset(uri=os.path.join(output_path, "dim_product.csv"))
dim_shipmode = Dataset(uri=os.path.join(output_path, "dim_shipmode.csv"))
sales_fact = Dataset(uri=os.path.join(output_path, "sales_fact.csv"))
