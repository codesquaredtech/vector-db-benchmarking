from app.database.vector_database import VectorDatabase
import weaviate
from weaviate.classes.data import DataObject
import uuid
import math
from weaviate.classes.query import MetadataQuery
import numpy as np

from weaviate.classes.config import (
    Configure,
    Property,
    DataType,
    VectorDistances,
)

from app.logger import get_logger

logger = get_logger()

global_client = None


class WeaviateDatabase(VectorDatabase):
    def connect(self, host="weaviate", port=8080):
        global global_client
        global_client = weaviate.connect_to_local(
            host=host,
            port=port,
            grpc_port=50051,
        )
        logger.info(f"Weavite client is ready: {global_client.is_ready()}")

    def drop_collection(self, collection_name: str):
        global global_client
        global_client.collections.delete(collection_name)

    def create_collection(self, collection_name: str, vector_size: int):
        global global_client
        properties = [
            # No need for ID - automatically gets generated as an UUID
            Property(
                name="image_path",
                data_type=DataType.TEXT,
                description="Path to the image",
                max_length=255,
            ),
        ]
        # TODO: Compare M with maxConnections: https://milvus.io/docs/v2.0.x/index.md and https://weaviate.io/developers/academy/py/vector_index/hnsw
        global_client.collections.create(
            collection_name,
            properties=properties,
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
                ef_construction=200,
                max_connections=16,
            ),
        )

        # Info about the specification
        # response = global_client.collections.list_all(simple=False)
        # logger.info(f"Weaviate specification: {response}")

    def insert(self, collection_name: str, data, batch_size: int = 500):
        global global_client
        collection = global_client.collections.get(collection_name)

        # Verify the dataframe has the required columns
        if "embedding" not in data.columns or "image_path" not in data.columns:
            raise ValueError("Data must contain 'embedding' and 'image_path' columns")

        total_rows = len(data)
        num_batches = math.ceil(total_rows / batch_size)

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, total_rows)
            batch = data.iloc[start:end]

            objects = []
            for _, row in batch.iterrows():
                if not isinstance(row["embedding"], (list, np.ndarray)):
                    raise ValueError(
                        f"Embedding must be a list or numpy array, got {type(row['embedding'])}"
                    )

                objects.append(
                    DataObject(
                        uuid=str(uuid.uuid4()),
                        properties={"image_path": row["image_path"]},
                        vector=list(row["embedding"]),
                    )
                )

            try:
                result = collection.data.insert_many(objects)
                logger.debug(f"Insert result: {result}")
            except Exception as e:
                logger.error(f"Failed to insert batch {i}: {e}")
                raise

    def delete(self, collection_name: str):
        # If we don't want to drop the collection, we can do it this way.
        # But, max amount of deleted objects in one query is 10000 (default).
        # There is a configurable maximum limit (QUERY_MAXIMUM_RESULTS) on the number of objects that can be deleted in a single query (default 10,000).
        # collection = client.collections.get(collection_name)
        # collection.data.delete_many()
        global global_client
        global_client.collections.delete(collection_name)
        global_client.close()

    def search(self, collection_name: str, embedding: list, params: dict):
        # https://weaviate.io/developers/weaviate/search/similarity
        global global_client

        limit = params["limit"]
        if limit is None:
            limit = 10000

        collection = global_client.collections.get(collection_name)
        response = collection.query.near_vector(
            near_vector=embedding,
            certainty=params["certainty"],
            limit=params["limit"],  # todo: include a case where limit is not present
            return_metadata=MetadataQuery(certainty=True),
        )

        # for o in response.objects:
        #    logger.info(o.properties)
        #    logger.info(o.metadata.certainty)

        return response.objects

    def parse_search_results(self, results: list):
        similar_embeddings = []
        for result in results:
            # logger.info(
            #    f"Image path: {result.properties.get('image_path')}, Score: {result.metadata.certainty}"
            # )
            similar_embeddings.append(
                result.properties.get("image_path").split("/")[-1]
            )
        return similar_embeddings
