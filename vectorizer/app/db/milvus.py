from pymilvus import DataType, MilvusClient

# ToDo: investigate this: https://milvus.io/api-reference/pymilvus/v2.4.x/MilvusClient/Client/MilvusClient.md
milvus_client = MilvusClient(uri="http://milvus:19530")


def create_collection():
    collection_name = "faces_from_events"
    if collection_name not in milvus_client.list_collections():
        # Schema
        schema = milvus_client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=1280
        )
        schema.add_field(
            field_name="image_path", datatype=DataType.VARCHAR, max_length=255
        )
        # schema.add_field(
        #     field_name="picture_id", datatype=DataType.VARCHAR, max_length=255
        # )
        # schema.add_field(
        #     field_name="event_name", datatype=DataType.VARCHAR, max_length=255
        # )
        # schema.add_field(field_name="user_id", datatype=DataType.INT64)

        # Index
        index_params = milvus_client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            # TODO: Discuss - there are other index_type values (not all of them are available for milvus-lite)
            index_type="FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )

        # Collection
        milvus_client.create_collection(
            collection_name=collection_name, schema=schema, index_params=index_params
        )


create_collection()
