import math

from app.database.vector_database import VectorDatabase
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)
from app.logger import get_logger

logger = get_logger()


class MilvusDatabase(VectorDatabase):
    def connect(self, host="milvus", port="19530"):
        connections.connect("default", host=host, port=port)

    def drop_collection(self, collection_name: str):
        if utility.has_collection(collection_name):
            collection = Collection(name=collection_name)
            collection.load()
            utility.drop_collection(collection_name)

    def create_collection(self, collection_name: str, vector_size: int):
        id_field = FieldSchema(
            name="id", dtype=DataType.INT64, is_primary=True, auto_id=False
        )
        embedding_field = FieldSchema(
            name="embedding", dtype=DataType.FLOAT_VECTOR, dim=vector_size
        )
        image_path_field = FieldSchema(
            name="image_path", dtype=DataType.VARCHAR, max_length=255
        )

        schema = CollectionSchema(
            fields=[id_field, embedding_field, image_path_field],
            description="Image database schema",
        )

        collection = Collection(name=collection_name, schema=schema)
        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        }
        collection.create_index(field_name="embedding", index_params=index_params)

    def insert(self, collection_name: str, data, batch_size: int = 500):
        collection = Collection(name=collection_name)

        total_rows = len(data)
        num_batches = math.ceil(total_rows / batch_size)

        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, total_rows)

            batch = data.iloc[start:end]

            ids = batch.index.tolist()
            embeddings = batch["embedding"].tolist()
            image_paths = batch["image_path"].tolist()

            collection.insert([ids, embeddings, image_paths])

    def delete(self, collection_name: str):
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)

    def search(self, collection_name: str, embedding: list, params: dict):
        collection = Collection(name=collection_name)
        collection.load()

        search_params = {
            "metric_type": params.get("metric_type", "COSINE"),
            "params": params.get("index_params", {"ef": 64}),
        }

        limit = params.get("limit")
        """
        offset - Number of entities to skip during the search.
        The sum of this parameter and limit of the search method should be less than 16384.
        """
        if limit is None:
            limit = 16000

        search_args = {
            "data": [embedding],
            "anns_field": params["anns_field"],
            "param": search_params,
            "limit": limit,
            "expr": params.get("expr"),
            "output_fields": params.get("output_fields", []),
        }

        raw_results = collection.search(**search_args)

        threshold = params.get("threshold", 0.5)

        filtered_results = []
        for result in raw_results:
            filtered_result = [item for item in result if item.score >= threshold]
            filtered_results.append(filtered_result)

        return filtered_results

    def parse_search_results(self, results: list):
        similar_embeddings = []
        for result in results:
            for hit in result:
                # logger.info(
                #    f"ID: {hit.entity.get('id')}, Image path: {hit.entity.get('image_path')}, Score: {hit.score}"
                # )
                similar_embeddings.append(hit.entity.get("image_path").split("/")[-1])
        return similar_embeddings
