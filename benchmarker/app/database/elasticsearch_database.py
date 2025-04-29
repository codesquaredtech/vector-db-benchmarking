from app.database.vector_database import VectorDatabase
from elasticsearch import Elasticsearch, helpers
from app.logger import get_logger
import time

logger = get_logger()


class ElasticsearchDatabase(VectorDatabase):
    def __init__(self):
        self.client = None

    def connect(self, host="elasticsearch_db", port=9200, retries=10, delay=5):
        attempt = 0
        while attempt < retries:
            try:
                self.client = Elasticsearch(
                    [
                        {
                            "host": host,
                            "port": port,
                            "scheme": "http",
                        }
                    ],
                    timeout=18000,
                )
                if self.client.ping():
                    logger.info(
                        f"Successfully connected to Elasticsearch at {host}:{port}"
                    )
                    return
                else:
                    raise ConnectionError(
                        f"Could not connect to Elasticsearch at {host}:{port}"
                    )

            except Exception as e:
                attempt += 1
                logger.warning(
                    f"Attempt {attempt}/{retries} failed to connect to Elasticsearch at {host}:{port}: {e}"
                )
                self.client = None
                if attempt < retries:
                    time.sleep(delay)
                else:
                    logger.error(
                        f"Failed to connect to Elasticsearch after {retries} attempts."
                    )

    def drop_collection(self, collection_name: str):
        if self.client is None:
            logger.error("Elasticsearch client is not connected.")
            return

        index_name = collection_name.lower()  # Has to be lowercase for es
        if self.client.indices.exists(index=index_name):
            self.client.indices.delete(index=index_name)
            logger.info(f"Index '{collection_name}' deleted successfully.")
        else:
            logger.info(f"Index '{collection_name}' does not exist. Skipping drop.")

    def create_collection(self, collection_name: str, vector_size: int):
        if self.client is None:
            logger.error("Elasticsearch client is not connected.")
            return

        index_name = collection_name.lower()

        mapping = {
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": vector_size,
                        "index": "true",
                        "similarity": "cosine",  # cosine is default
                    },
                    "image_path": {"type": "keyword"},
                }
            }
        }

        if not self.client.indices.exists(index=index_name):
            self.client.indices.create(index=index_name, body=mapping)
            logger.info(f"Index '{collection_name}' created successfully.")
        else:
            logger.warning(
                f"Index '{collection_name}' already exists. Skipping index creation."
            )

    def _generate_actions(self, data, index_name):
        for idx, row in data.iterrows():
            # logger.info(f"Idx, row: {idx}, {row}")
            yield {
                "_index": index_name,
                "_id": idx,
                "_source": {
                    "embedding": row["embedding"],
                    "image_path": row["image_path"],
                },
            }

    def insert(self, collection_name: str, data, batch_size: int = 500):
        if self.client is None:
            logger.error("Elasticsearch client is not connected.")
            return

        if data.empty:
            logger.warning("Data is empty. Skipping insert.")
            return

        index_name = collection_name.lower()
        total = len(data)

        logger.info(
            f"Inserting {total} documents into '{collection_name}' in batches of {batch_size}."
        )

        try:
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                batch = data.iloc[start:end]

                # Convert batch to actions without loading everything into memory
                actions = (
                    {
                        "_index": index_name,
                        "_id": idx,
                        "_source": {
                            "embedding": row["embedding"],
                            "image_path": row["image_path"],
                        },
                    }
                    for idx, row in batch.iterrows()
                )

                helpers.bulk(self.client, actions)
                logger.info(
                    f"Inserted batch {start // batch_size + 1} ({len(batch)} documents) into '{collection_name}'."
                )

        except Exception as e:
            logger.error(f"Error inserting data into Elasticsearch: {e}")

    def delete(self, collection_name: str):
        if self.client is None:
            logger.error("Elasticsearch client is not connected.")
            return

        try:
            self.client.indices.delete(index=collection_name.lower())
        except Exception as e:
            logger.error(f"Error deleting data from Elasticsearch: {e}")

    def search(self, collection_name: str, embedding: list, params: dict):
        if self.client is None:
            logger.error("Elasticsearch client is not connected.")
            return []

        index_name = collection_name.lower()
        limit = params.get("limit", 10)
        certainty = params.get("certainty", 0.5)
        num_candidates = params.get("num_candidates", 100)
        results = []

        knn = {
            "field": "embedding",
            "query_vector": embedding,
            "k": limit,
            "num_candidates": num_candidates,
        }

        body = {
            "size": limit,
            "knn": knn,
            "_source": ["image_path"],
            "min_score": certainty,
        }

        try:
            response = self.client.search(index=index_name, body=body)

            results = [
                {**hit["_source"], "score": hit["_score"]}
                for hit in response["hits"]["hits"]
            ]

            return results

        except Exception as e:
            logger.error(f"Error occured while searching in Elasticsearch: {e}")
            return []

    def parse_search_results(self, results: list):
        if self.client is None:
            logger.error("Elasticsearch client is not connected.")
            return

        similar_embeddings = []

        for result in results:
            image_path = result["image_path"]
            score = result["score"]

            # logger.info(f"Image path: {image_path}, Score: {score}")

            if image_path:
                similar_embeddings.append(image_path.split("/")[-1])

        return similar_embeddings
