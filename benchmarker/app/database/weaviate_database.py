from app.database.vector_database import VectorDatabase
import weaviate

from app.logger import get_logger

logger = get_logger()


class WeaviateDatabase(VectorDatabase):
    def connect(self, host="weaviate", port=8080):
        client = weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=50051,
        )
        logger.info(f"Weavite client is ready: {client.is_ready()}")

    def drop_collection(self, collection_name: str):
        pass

    def create_collection(self, collection_name: str):
        pass

    def insert(self, collection_name: str, data):
        pass

    def delete(self, collection_name: str):
        pass

    def search(self, collection_name: str, embedding: list, params: dict):
        pass

    def parse_search_results(self, results: list):
        pass
