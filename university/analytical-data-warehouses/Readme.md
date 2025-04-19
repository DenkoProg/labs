# Initializing Environment
Before starting Airflow for the first time, you need to prepare your environment, i.e. create the necessary files, directories and initialize the database.

## Setting the right Airflow user (only for UNIX based OS)
```
echo -e "AIRFLOW_UID=$(id -u)" > .env
```

## Initialize the database
On all operating systems, you need to run database migrations and create the first user account. To do this, run.
```
docker compose up airflow-init
```

After initialization is complete, you should see a message like this:
```
airflow-init_1       | Upgrades done
airflow-init_1       | Admin user airflow created
airflow-init_1       | 2.5.1
start_airflow-init_1 exited with code 0
```

The account created has the login **airflow** and the password **airflow**.

## Cleaning-up the environment
The docker-compose environment we have prepared is a “quick-start” one.
To clean-up run:
```
docker compose down --volumes --remove-orphans
```

## Running Airflow
Now you can start all services:
```
docker compose up
```

## Admin webpage
Admin webpage is located: 
[http://localhost:8080/](http://localhost:8080/)

Credentials.
```
user: airflow
password: airflow
```