import chromadb
from chromadb.config import Settings

from app.database.vector_database import VectorDatabase
from app.logger import get_logger


logger = get_logger()


class ChromaDatabase(VectorDatabase):
    def __init__(self):
        self.client = None

    def connect(self, host="chroma_db", port="8000"):
        try:
            self.client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(allow_reset=True, anonymized_telemetry=False),
            )

            logger.info(f"Successfully connected to Chroma at {host}:{port}")

        except Exception as e:
            logger.error(f"Error connecting to Chroma at {host}:{port}: {e}")
            self.client = None

    def drop_collection(self, collection_name: str):
        if self.client is None:
            raise ConnectionError("Chroma client is not connected.")

        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Collection '{collection_name}' deleted successfully.")

        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")

    def create_collection(self, collection_name: str, vector_size: str):
        if self.client is None:
            raise ConnectionError("Chroma client is not connected.")

        try:
            self.client.create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Collection '{collection_name}' created successfully.")

        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {e}")

    def insert(self, collection_name: str, data):
        if self.client is None:
            raise ConnectionError("Chroma client is not connected.")

        if data.empty:
            logger.warning("Data is empty. Skipping insert.")
            return

        try:
            collection = self.client.get_or_create_collection(name=collection_name)

            ids = data.index.tolist()
            str_ids = [str(element) for element in ids]
            embeddings = data["embedding"].tolist()
            image_paths = [{"image_path": path} for path in data["image_path"]]

            collection.add(embeddings=embeddings, metadatas=image_paths, ids=str_ids)

        except Exception as e:
            logger.error(f"Failed to insert data into '{collection_name}': {e}")

    def delete(self, collection_name: str):
        if self.client is None:
            raise ConnectionError("Chroma client is not connected.")

        try:
            collection = self.client.get_collection(name=collection_name)
            collection.delete(where={"id": {"$ne": ""}})
            logger.info(f"Deleted all records from collection '{collection_name}'.")

        except Exception as e:
            logger.error(f"Failed to delete data from '{collection_name}': {e}")

    def search(self, collection_name: str, embedding: list, params: dict):
        if self.client is None:
            logger.error("Chroma client is not connected.")
            return []

        collection = self.client.get_or_create_collection(name=collection_name)

        limit = params.get("limit", 16000)
        threshold = params.get("threshold ", 0.5)

        results = collection.query(query_embeddings=[embedding], n_results=limit)

        distances = results.get("distances", [])[0]
        ids = results.get("ids", [])[0]
        metadatas = results.get("metadatas", [])[0]

        filtered_results = [
            {"id": ids[i], "distance": distances[i], "metadata": metadatas[i]}
            for i in range(len(distances))
            if distances[i] <= threshold
        ]

        return filtered_results

    def parse_search_results(self, results: list):
        similar_embeddings = []

        for result in results:
            image_path = result["metadata"].get("image_path")
            score = result["distance"]

            logger.info(f"Image path: {image_path}, Score: {score}")

            if image_path:
                similar_embeddings.append(image_path.split("/")[-1])

        return similar_embeddings
