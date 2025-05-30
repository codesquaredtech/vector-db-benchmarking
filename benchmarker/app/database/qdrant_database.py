import math
from app.database.vector_database import VectorDatabase
from app.logger import get_logger

logger = get_logger()

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Batch,
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    Range,
    SearchParams,
    VectorParams,
)

VECTOR_NAME = "image_vector"


class QdrantDatabase(VectorDatabase):
    def __init__(self):
        self.client = None

    def connect(self, host="qdrant", port="6333"):
        logger.info("Initializing QD client")
        try:
            self.client = QdrantClient(url=f"http://{host}:{port}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e

    def drop_collection(self, collection_name: str):
        logger.info(f"Dropping {collection_name} collection")
        try:
            self.client.delete_collection(collection_name)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e

    def create_collection(self, collection_name: str, vector_size: int):
        logger.info(f"Creating {collection_name} collection")
        try:
            self.client.create_collection(
                collection_name=f"{collection_name}",
                vectors_config={
                    VECTOR_NAME: VectorParams(
                        size=vector_size, distance=Distance.COSINE
                    ),
                },
                hnsw_config={
                    "m": 16,
                    "ef_construct": 10000,
                },
            )

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e

    def insert(self, collection_name: str, data, batch_size: int = 500):
        total_rows = len(data)
        num_batches = math.ceil(total_rows / batch_size)

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, total_rows)
            batch = data.iloc[start:end]

            ids = batch.index.tolist()
            embeddings = batch["embedding"].tolist()
            image_paths = batch["image_path"].tolist()
            images_as_payload = [{"image_path": path} for path in image_paths]

            logger.info(
                f"Inserting batch {i + 1}/{num_batches} with {len(ids)} points into {collection_name}"
            )

            try:
                self.client.upsert(
                    collection_name=collection_name,
                    points=Batch(
                        ids=ids,
                        payloads=images_as_payload,
                        vectors={VECTOR_NAME: embeddings},
                    ),
                )
            except Exception as e:
                logger.error(f"Error inserting batch {i + 1}: {e}")
                raise e

    def delete(self, collection_name: str):
        logger.info(f"Deleting everything from {collection_name}")

        try:
            self.client.delete(
                collection_name=f"{collection_name}",
                points_selector=FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="id",
                                range=Range(
                                    gt=0,
                                    gte=None,
                                    lt=None,
                                    lte=None,
                                ),
                            ),
                        ],
                    )
                ),
            )
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e

    def search(self, collection_name: str, embedding: list, params: dict):
        logger.info(f"Searching in collection {collection_name}")
        response = []
        try:
            response = self.client.query_points(
                collection_name=f"{collection_name}",
                query=embedding,
                search_params=SearchParams(exact=False),
                score_threshold=params.get("certainty", 0),
                limit=params.get("limit", 1600),
                with_payload=["image_path"],
                using=VECTOR_NAME,
            )

        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e

        return response

    def parse_search_results(self, results: list):
        logger.info(f"Parsing search results: {results}")
        similar_embeddings = []
        for point in results.points:
            #            logger.info(
            #                f"ID: {point.id}, Image path: {point.payload['image_path']}, Score: {point.score}"
            #            )
            similar_embeddings.append(point.payload["image_path"].split("/")[-1])
        return similar_embeddings
