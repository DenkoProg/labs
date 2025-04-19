from airflow import DAG
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta


dag = DAG(
    dag_id="dummy_dag",
    schedule="*/5 * * * *",
    start_date=datetime(2023, 1, 1),
    catchup=False,
    dagrun_timeout=timedelta(minutes=60),
)
dummy_task_1 = DummyOperator(
    task_id='dummy_task_1',
    dag=dag
)

dummy_task_2 = DummyOperator(
    task_id='dummy_task_2',
    dag=dag
)

dummy_task_3 = DummyOperator(
    task_id='dummy_task_3',
    dag=dag
)

dummy_task_4 = DummyOperator(
    task_id='dummy_task_4',
    dag=dag
)

dummy_task_1 >> [dummy_task_2, dummy_task_3] >> dummy_task_4
