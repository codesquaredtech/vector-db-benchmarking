import pandas as pd
import datetime
from deepface import DeepFace

# Milvus
from app.milvus_database import MilvusDatabase
from app.face_detection import (
    extract_faces_from_deepface_detections,
    initialise_face_embedder,
    create_embedding,
)
from app.logger import get_logger
from app.images import convert_bytes_to_image

OUTPUT_FILE_PATH = "./output/embeddings_2024-12-17_09-53-28.parquet"
COLLECTION_NAME = "faces_collection"


def get_vector_database(db_type: str):
    if db_type == "MILVUS":
        return MilvusDatabase()
    else:
        raise ValueError(f"Unsupported vector database: {db_type}")


def retrieve_embeddings_from_parquet_file(file_path):
    try:
        embeddings = pd.read_parquet(file_path, engine="pyarrow")
        return embeddings
    except Exception as e:
        logger.error(
            f"An error occurred while retrieving embeddings from {file_path}: {e}"
        )


def process_input_image(image_to_compare_with_path):
    face_embedder = initialise_face_embedder()

    numpy_image, _ = convert_bytes_to_image(image_to_compare_with_path)
    detected_faces = DeepFace.extract_faces(
        img_path=numpy_image,
        enforce_detection=False,
        detector_backend="retinaface",
        align=True,
    )
    face_images = extract_faces_from_deepface_detections(detected_faces)

    image_embedding = create_embedding(face_images[0], face_embedder)
    return image_embedding


def insert_embeddings(db):
    logger.info("Retrieving extracted embeddings")
    embeddings = retrieve_embeddings_from_parquet_file(OUTPUT_FILE_PATH)
    logger.info("Embeddings retrieved successfully")

    logger.info("Connecting to the vector database")
    db.connect()
    db.drop_collection(COLLECTION_NAME)
    logger.info("Successfully connected to the vector database")

    vector_db_initialisation_start_datetime = datetime.datetime.now()
    logger.info(
        f'Starting vector database initialisation at: {vector_db_initialisation_start_datetime.strftime("%Y-%m-%d_%H-%M-%S")}'
    )
    db.create_collection(COLLECTION_NAME)

    vector_db_initialisation_end_datetime = datetime.datetime.now()
    logger.info(
        f'Finished vector database initialisation at: {vector_db_initialisation_end_datetime.strftime("%Y-%m-%d_%H-%M-%S")}'
    )
    logger.info(
        f"Total vector database initialisation time: {(vector_db_initialisation_end_datetime-vector_db_initialisation_start_datetime).total_seconds()}"
    )

    vector_db_insertion_start_datetime = datetime.datetime.now()
    logger.info(
        f'Starting vector database insertion at: {vector_db_insertion_start_datetime.strftime("%Y-%m-%d_%H-%M-%S")}'
    )
    db.insert(COLLECTION_NAME, embeddings)

    vector_db_insertion_end_datetime = datetime.datetime.now()
    logger.info(
        f'Finished vector database insertion at: {vector_db_insertion_end_datetime.strftime("%Y-%m-%d_%H-%M-%S")}'
    )
    logger.info(
        f"Total vector database insertion time: {(vector_db_insertion_end_datetime-vector_db_insertion_start_datetime).total_seconds()}"
    )


def search_similar_embeddings(db, image_to_compare_with_path, search_params):
    logger.info(f"Processing image: {image_to_compare_with_path}")
    image_embedding = process_input_image(image_to_compare_with_path)
    logger.info(f"Image: {image_to_compare_with_path} converted to the embedding")

    similar_embedding_search_start_time = datetime.datetime.now()
    logger.info(
        f'Starting similar embedding search at: {similar_embedding_search_start_time.strftime("%Y-%m-%d_%H-%M-%S")}'
    )
    similar_embeddings = db.search(COLLECTION_NAME, image_embedding, search_params)

    similar_embedding_search_end_time = datetime.datetime.now()
    logger.info(f"Similar embeddings: {similar_embeddings}")
    logger.info(
        f'Finished similar embedding search at: {similar_embedding_search_end_time.strftime("%Y-%m-%d_%H-%M-%S")}'
    )
    logger.info(
        f"Total similar embedding search time: {(similar_embedding_search_end_time-similar_embedding_search_start_time).total_seconds()}"
    )


if __name__ == "__main__":
    logger = get_logger()

    db = get_vector_database("MILVUS")

    insert_embeddings(db)

    search_params = search_params = {
        "anns_field": "embedding",
        "metric_type": "COSINE",
        "index_params": {"ef": 64},
        "limit": 5,
        "output_fields": ["id", "image_path"],
    }

    search_similar_embeddings(db, "./app/test-benchmarking-face.jpg", search_params)
