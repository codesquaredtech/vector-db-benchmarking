from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base

VectorDB = declarative_base(metadata=MetaData(schema="vectordb"))
