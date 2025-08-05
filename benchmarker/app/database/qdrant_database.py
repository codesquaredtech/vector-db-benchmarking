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
from qdrant_client.http.exceptions import UnexpectedResponse

import math
import os
import csv
import time
from app.database.vector_database import VectorDatabase
from app.logger import get_logger

logger = get_logger()

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
        logger.info(f"Creating fresh '{collection_name}' collection")

        try:
            existing_collections = self.client.get_collections().collections
            existing_names = [col.name for col in existing_collections]

            if collection_name in existing_names:
                logger.info(
                    f"Collection '{collection_name}' already exists. Dropping it."
                )
                self.client.delete_collection(collection_name=collection_name)

            # Create a new collection
            self.client.create_collection(
                collection_name=collection_name,
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

            logger.info(f"Collection '{collection_name}' created successfully.")

        except Exception as e:
            logger.error(
                f"An error occurred while resetting collection '{collection_name}': {e}"
            )
            raise

    def insert(
        self,
        collection_name: str,
        data,
        batch_size: int = 500,
        timing_csv_path: str = "results/batch_times_qdrant.csv",
    ):
        total_rows = len(data)
        num_batches = math.ceil(total_rows / batch_size)

        batch_times = []

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

            batch_num = i + 1
            batch_start_time = time.time()

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
                logger.error(f"Error inserting batch {batch_num}: {e}")
                raise e

            batch_end_time = time.time()
            elapsed = batch_end_time - batch_start_time
            logger.info(f"Batch {batch_num} inserted in {elapsed:.2f} seconds.")
            batch_times.append({"batch": batch_num, "time_sec": elapsed})

        # Append batch times to CSV file
        write_header = not os.path.exists(timing_csv_path)
        with open(timing_csv_path, mode="a", newline="") as csvfile:
            fieldnames = ["batch", "time_sec"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for entry in batch_times:
                writer.writerow(entry)

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
