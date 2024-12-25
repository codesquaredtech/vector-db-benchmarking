from app.vector_database import VectorDatabase
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)


class MilvusDatabase(VectorDatabase):
    def connect(self, host="milvus", port="19530"):
        connections.connect("default", host=host, port=port)

    def drop_collection(self, collection_name: str):
        if utility.has_collection(collection_name):
            collection = Collection(name=collection_name)
            collection.load()
            utility.drop_collection(collection_name)

    def create_collection(self, collection_name: str):
        id_field = FieldSchema(
            name="id", dtype=DataType.INT64, is_primary=True, auto_id=False
        )
        embedding_field = FieldSchema(
            name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1280
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

    def insert(self, collection_name: str, data):
        ids = data.index.tolist()
        embeddings = data["embedding"].tolist()
        image_paths = data["image_path"].tolist()

        collection = Collection(name=collection_name)
        collection.insert([ids, embeddings, image_paths])

    def search(self, collection_name: str, embedding: list, params: dict):
        collection = Collection(name=collection_name)
        collection.load()

        search_params = {
            "metric_type": params.get("metric_type", "COSINE"),
            "params": params.get("index_params", {"ef": 64}),
        }

        results = collection.search(
            data=[embedding],
            anns_field=params["anns_field"],
            param=search_params,
            limit=params.get("limit", 10),
            expr=params.get("expr"),
            output_fields=params.get("output_fields", []),
        )

        # Parse and return results
        similar_embeddings = []
        for result in results:
            for hit in result:
                similar_embeddings.append(
                    {
                        "id": hit.entity.get("id"),
                        "image_path": hit.entity.get("image_path"),
                        "score": hit.score,
                    }
                )
        return similar_embeddings
