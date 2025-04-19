from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base

public_metadata = (
    MetaData(schema='public')
)

PublicBase = declarative_base(metadata=public_metadata)
