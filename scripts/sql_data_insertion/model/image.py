from .base import VectorDB
from sqlalchemy import (
    Column,
    Integer,
    String,
    TIMESTAMP,
    func,
    ForeignKey,
    JSON,
    UniqueConstraint,
)


class Image(VectorDB):
    __tablename__ = "image"

    id = Column("id", Integer, primary_key=True)
    name = Column("name", String(128), nullable=False)
    tags = Column("tags", JSON, nullable=True)
    inserted_at = Column("inserted_at", TIMESTAMP, server_default=func.now())
    directory_id = Column(
        "directory_id", Integer, ForeignKey("directory.id"), nullable=False
    )

    __table_args__ = (
        UniqueConstraint("name", name="image_unique"),
    )
