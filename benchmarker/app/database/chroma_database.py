import math
import time

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

    def insert(
        self,
        collection_name: str,
        data,
        batch_size: int = 500,
        retries: int = 5,
        delay: float = 10,
    ):
        if self.client is None:
            raise ConnectionError("Chroma client is not connected.")

        if data.empty:
            logger.warning("Data is empty. Skipping insert.")
            return

        def reconnect():
            logger.info("Attempting to reconnect to Chroma...")
            self.connect()
            if self.client is None:
                raise ConnectionError("Reconnection to Chroma failed.")

        try:
            try:
                collection = self.client.get_or_create_collection(name=collection_name)
            except Exception as e:
                logger.warning(f"Error during collection access: {e}")
                reconnect()
                collection = self.client.get_or_create_collection(name=collection_name)

            total_rows = len(data)
            num_batches = math.ceil(total_rows / batch_size)

            for i in range(num_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, total_rows)
                batch = data.iloc[start:end]

                ids = batch.index.tolist()
                str_ids = [str(i) for i in ids]
                embeddings = batch["embedding"].tolist()
                image_paths = [{"image_path": path} for path in batch["image_path"]]

                logger.info(
                    f"Inserting batch {i + 1}/{num_batches} with {len(str_ids)} points into {collection_name}"
                )

                attempt = 0
                while attempt <= retries:
                    try:
                        collection.add(
                            embeddings=embeddings, metadatas=image_paths, ids=str_ids
                        )
                        break  # Success, exit retry loop
                    except Exception as e:
                        logger.warning(
                            f"Error inserting batch {i + 1}: {e} (attempt {attempt + 1}/{retries})"
                        )
                        attempt += 1

                        # Try reconnecting if it's a connection-related error
                        if "ConnectionError" in str(type(e)) or "Connection" in str(e):
                            try:
                                reconnect()
                                collection = self.client.get_or_create_collection(
                                    name=collection_name
                                )
                            except Exception as reconnect_error:
                                logger.error(
                                    f"Reconnect attempt failed: {reconnect_error}"
                                )
                                raise reconnect_error

                        if attempt > retries:
                            logger.error(f"Final attempt failed for batch {i + 1}: {e}")
                            raise e
                        time.sleep(delay)

        except Exception as e:
            logger.error(f"Failed to insert data into '{collection_name}': {e}")
            raise e

    def delete(self, collection_name: str):
        if self.client is None:
            raise ConnectionError("Chroma client is not connected.")

        try:
            # collection = self.client.get_collection(name=collection_name)
            self.client.delete_collection(name=collection_name)
            # collection.delete(where={"id": {"$ne": ""}})
            logger.info(f"Deleted all records from collection '{collection_name}'.")

        except Exception as e:
            logger.error(f"Failed to delete data from '{collection_name}': {e}")

    def search(self, collection_name: str, embedding: list, params: dict):
        if self.client is None:
            logger.error("Chroma client is not connected.")
            return []

        collection = self.client.get_or_create_collection(name=collection_name)

        limit = params.get("limit", 16000)
        threshold = params.get("threshold", 0.5)

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

            # logger.info(f"Image path: {image_path}, Score: {score}")

            if image_path:
                similar_embeddings.append(image_path.split("/")[-1])

        return similar_embeddings
