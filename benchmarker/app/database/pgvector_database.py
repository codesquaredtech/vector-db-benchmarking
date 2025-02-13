import psycopg2
from app.database.vector_database import VectorDatabase
from app.logger import get_logger
from pgvector.psycopg2 import register_vector
from psycopg2 import sql
from sqlalchemy import create_engine

logger = get_logger()


class PGVectorDatabase(VectorDatabase):
    def __init__(self):
        self.connection = None

    def connect(
        self,
        dbname="vbenchmarkdb",
        user="vbenchmarkusr",
        password="vbenchmarkpass",
        host="pgvector_db",
        port=5432,
    ):
        try:
            self.connection = psycopg2.connect(
                dbname=dbname, user=user, password=password, host=host, port=port
            )

            register_vector(self.connection)
            logger.info("Connection successful.")
        except Exception as e:
            logger.error(f"Error connecting to the database: {e}")

    def drop_collection(self, collection_name: str):
        with self.connection.cursor() as cur:
            try:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(
                        sql.Identifier(collection_name)
                    )
                )
                logger.info(f"Table '{collection_name}' dropped successfully.")
                self.connection.commit()
            except Exception as e:
                logger.info(f"Error dropping table: {e}")
                self.connection.rollback()

    def create_collection(self, collection_name: str, vector_size: int):
        columns = [
            "id bigserial PRIMARY KEY",
            f"embedding vector({vector_size})",
            "image_path varchar(255)",
        ]

        columns_string = ",".join(columns)
        with self.connection.cursor() as cur:
            try:
                cur.execute(
                    sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
                        sql.Identifier(collection_name), sql.SQL(columns_string)
                    )
                )
                cur.execute(
                    sql.SQL(
                        "CREATE INDEX ON {} USING hnsw (embedding vector_cosine_ops)"
                    ).format(sql.Identifier(collection_name))
                )
                logger.info(f"Table '{collection_name}' created successfully.")
                self.connection.commit()
            except Exception as e:
                logger.error(f"Error creating table: {e}")
                self.connection.rollback()

    def insert(self, collection_name: str, data):
        self.sqlalchemy_engine = create_engine(
            "postgresql+psycopg2://", creator=lambda: self.connection
        )
        try:
            data.to_sql(
                collection_name,
                con=self.sqlalchemy_engine,
                if_exists="append",
                index=False,
            )
            logger.info(
                f"Data from DataFrame inserted into '{collection_name}' successfully."
            )
        except Exception as e:
            logger.error(f"Error inserting DataFrame: {e}")

    def delete(self, collection_name: str):
        with self.connection.cursor() as cur:
            try:
                query = sql.SQL("DELETE FROM {}").format(
                    sql.Identifier(collection_name)
                )
                cur.execute(query)
                self.connection.commit()
                logger.info("Data deleted successfully.")
            except Exception as e:
                logger.error(f"Error deleting data: {e}")
                self.connection.rollback()

    def search(self, collection_name: str, embedding: list, params: dict):
        results = []

        search_string = f"""
with image_score as (
	select f.image_path, (1 - (f.embedding <=> %s)) as score
	from "{collection_name}" f
)
select image_path, score
from image_score
where score >= {params.get("certainty", 0)}
order by score desc
limit {params.get("limit", 1600)};
"""

        try:
            with self.connection.cursor() as cur:
                cur.execute(search_string, (embedding,))
                results = cur.fetchall()
                logger.info("#" * 25)
                logger.info("Executed Query:")
                logger.info(cur.query)
                logger.info("-" * 15)
                logger.info("Fetched Results:")
                logger.info(results)
                self.connection.commit()
        except Exception as e:
            logger.error(f"Error searching data: {e}")
            self.connection.rollback()

        return results

    def parse_search_results(self, results: list):
        similar_embeddings = []
        for result in results:
            logger.info(f"Image path: {result[0]}, Score: {result[1]}")
            similar_embeddings.append(result[0].split("/")[-1])
        return similar_embeddings
