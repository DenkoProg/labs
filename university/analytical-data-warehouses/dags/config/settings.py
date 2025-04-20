from dotenv import load_dotenv
import os
from pathlib import Path

ENV_PATH = Path(__file__).parents[2] / ".env"

load_dotenv(ENV_PATH)


class Config:
    # Database
    DB_HOST = os.getenv("DB_HOST", "postgres")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "airflow")
    DB_USER = os.getenv("DB_USER", "airflow")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "airflow")

    # Paths
    DATA_INPUT_PATH = os.getenv("DATA_INPUT_PATH", "data/Store.xlsx")
    DATA_OUTPUT_DIR = os.getenv("DATA_OUTPUT_DIR", "data/")

    @property
    def database_url(self):
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


config = Config()
