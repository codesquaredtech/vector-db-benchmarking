from .base import VectorDB
from sqlalchemy import Column, Integer, String, TIMESTAMP, func


class Directory(VectorDB):
    __tablename__ = "directory"

    id = Column("id", Integer, primary_key=True)
    path = Column("path", String(768), nullable=False)
    inserted_at = Column("inserted_at", TIMESTAMP, server_default=func.now())
