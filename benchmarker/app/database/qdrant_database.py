from app.database.vector_database import VectorDatabase
from app.logger import get_logger

logger = get_logger()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, HnswConfig, VectorParams


class QdrantDatabase(VectorDatabase):
    def __init__(self):
        self.client = None

    def connect(self, host="qdrant", port="6333"):
        self.client = QdrantClient(url=f"http://{host}:{port}")

    def drop_collection(self, collection_name: str):
        pass
    
    def create_collection(self, collection_name: str):
        self.client.create_collection(
            collection_name=f"{collection_name}",
            vectors_config={
                "dense": VectorParams(size=1280, 
                                            distance=Distance.COSINE),
            },
            hnsw_config={"m": 16, "ef_construct": 200} # Trebalo bi razmisliti i o ostalim potencijalnim konfiguracijama
        )


    def insert(self, collection_name: str, data):
        pass

    def delete(self, collection_name: str):
        pass

    def search(self, collection_name: str, embedding: list, params: dict):
       pass

    def parse_search_results(self, results: list):
        pass
