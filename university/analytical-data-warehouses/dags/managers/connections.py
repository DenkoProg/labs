import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, load_only, Query

bi_engine = create_engine(
    os.getenv('DATABASE_URL'),
)

_bi_internal_session = sessionmaker(bind=bi_engine)
bi_session = _bi_internal_session()
