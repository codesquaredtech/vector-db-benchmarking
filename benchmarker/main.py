import pandas as pd
import numpy as np
import psutil
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
VECTOR_STORING_BENCHMARKING_RESULTS_BASE_FILE_PATH = "./results/vector_storing_results_"

COLLECTION_NAME = "faces_collection"
NUM_ITERATIONS = 10


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


def insert_embeddings(db, num_iterations=NUM_ITERATIONS):
    logger.info("Retrieving extracted embeddings")
    embeddings = retrieve_embeddings_from_parquet_file(OUTPUT_FILE_PATH)
    logger.info("Embeddings retrieved successfully")

    benchmark_data = []

    for i in range(num_iterations):
        logger.info(f"Starting benchmark iteration {i + 1}/{num_iterations}")

        process = psutil.Process()

        logger.info("Connecting to the vector database")
        db.connect()
        db.drop_collection(COLLECTION_NAME)
        logger.info("Successfully connected to the vector database")

        # Measure memory in MB
        memory_before_init = process.memory_info().rss / (1024**2)
        vector_db_initialisation_start = datetime.datetime.now()
        db.create_collection(COLLECTION_NAME)
        vector_db_initialisation_end = datetime.datetime.now()
        memory_after_init = process.memory_info().rss / (1024**2)

        initialisation_time = (
            vector_db_initialisation_end - vector_db_initialisation_start
        ).total_seconds()

        memory_before_insert = process.memory_info().rss / (1024**2)
        vector_db_insertion_start = datetime.datetime.now()
        db.insert(COLLECTION_NAME, embeddings)
        vector_db_insertion_end = datetime.datetime.now()
        memory_after_insert = process.memory_info().rss / (1024**2)

        insertion_time = (
            vector_db_insertion_end - vector_db_insertion_start
        ).total_seconds()

        memory_usage_initialisation = memory_after_init - memory_before_init
        memory_usage_insertion = memory_after_insert - memory_before_insert

        logger.info(
            f"Iteration {i + 1} - Initialisation Time: {initialisation_time}s, Insertion Time: {insertion_time}s, "
            f"Memory Usage (Init): {memory_usage_initialisation} MB, Memory Usage (Insert): {memory_usage_insertion} MB"
        )

        benchmark_data.append(
            {
                "iteration": i + 1,
                "initialisation_time": initialisation_time,
                "insertion_time": insertion_time,
                "memory_usage_initialisation": memory_usage_initialisation,
                "memory_usage_insertion": memory_usage_insertion,
            }
        )

    benchmark_df = pd.DataFrame(benchmark_data)
    complete_file_path = (
        VECTOR_STORING_BENCHMARKING_RESULTS_BASE_FILE_PATH
        + f"_size_{len(embeddings)}__"
        + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        + ".csv"
    )
    benchmark_df.to_csv(complete_file_path, index=False)

    initialisation_times = benchmark_df["initialisation_time"]
    insertion_times = benchmark_df["insertion_time"]
    memory_usage_initialisation = benchmark_df["memory_usage_initialisation"]
    memory_usage_insertion = benchmark_df["memory_usage_insertion"]

    stats = {
        "initialisation_mean": np.mean(initialisation_times),
        "initialisation_std": np.std(initialisation_times),
        "insertion_mean": np.mean(insertion_times),
        "insertion_std": np.std(insertion_times),
        "initialisation_p90": np.percentile(initialisation_times, 90),
        "insertion_p90": np.percentile(insertion_times, 90),
        "memory_usage_init_mean": np.mean(memory_usage_initialisation),
        "memory_usage_init_std": np.std(memory_usage_initialisation),
        "memory_usage_insert_mean": np.mean(memory_usage_insertion),
        "memory_usage_insert_std": np.std(memory_usage_insertion),
    }

    stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
    stats_df.to_csv(complete_file_path, mode="a", header=False, index=False)

    logger.info(f"Benchmark results saved to {complete_file_path}")


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
