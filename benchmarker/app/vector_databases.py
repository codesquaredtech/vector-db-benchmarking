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

COLLECTION_NAME = "faces_collection"


def handle_exceptions(logger):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"An error occurred in {func.__name__}: {e}")

        return wrapper

    return decorator


# TODO: Add support for other vector databases


@handle_exceptions(logger)
def connect_to_vector_database(vector_database_name="MILVUS"):
    if vector_database_name == "MILVUS":
        connections.connect("default", host="milvus", port="19530")
        logger.info("Successfully connected to MILVUS")
    else:
        logger.error(f"Unsupported vector database: {vector_database_name}")


@handle_exceptions(logger)
def drop_already_existing_collection(vector_database_name="MILVUS"):
    if vector_database_name == "MILVUS":
        if utility.has_collection(COLLECTION_NAME):
            logger.info(f"Collection '{COLLECTION_NAME}' already exists. Dropping it.")
            utility.drop_collection(COLLECTION_NAME)
            logger.info(f"Collection '{COLLECTION_NAME}' dropped.")
    else:
        logger.error(f"Unsupported vector database: {vector_database_name}")


@handle_exceptions(logger)
def initialise_vector_database(vector_database_name="MILVUS"):
    if vector_database_name == "MILVUS":
        id_field = FieldSchema(
            name="id", dtype=DataType.INT64, is_primary=True, auto_id=False
        )
        embedding_field = FieldSchema(
            name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1280
        )
        image_path_field = FieldSchema(
            name="image_path", dtype=DataType.VARCHAR, max_length=255
        )

        logger.info("Initializing schema")
        schema = CollectionSchema(
            fields=[id_field, embedding_field, image_path_field],
            description="Image database schema",
        )

        logger.info("Creating collection")
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        logger.info(f"Collection '{COLLECTION_NAME}' created!")

        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 200},
        }
        collection.create_index(field_name="embedding", index_params=index_params)
        logger.info("HNSW index created!")
    else:
        logger.error(f"Unsupported vector database: {vector_database_name}")


@handle_exceptions(logger)
def insert_embeddings_into_vector_database(
    embeddings_df, vector_database_name="MILVUS"
):
    if vector_database_name == "MILVUS":
        collection = Collection(name=COLLECTION_NAME)

        ids = embeddings_df.index.tolist()
        embeddings = embeddings_df["embedding"].tolist()
        image_paths = embeddings_df["image_path"].tolist()

        collection.insert([ids, embeddings, image_paths])
        logger.info(
            f"Inserted {len(embeddings_df)} records into the collection '{COLLECTION_NAME}'."
        )
    else:
        logger.error(f"Unsupported vector database: {vector_database_name}")


# TODO: Add functions for search
