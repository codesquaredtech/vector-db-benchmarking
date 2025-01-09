import pandas as pd
import numpy as np
import psutil
import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.metrics import precision_score, recall_score, f1_score
from deepface import DeepFace

# Milvus
from app.database.milvus_database import MilvusDatabase
from app.face_detection import (
    extract_faces_from_deepface_detections,
    initialise_face_embedder,
    create_embedding,
)
from app.logger import get_logger
from app.images import convert_bytes_to_image

"""
Modify global variables if needed.
"""

INPUT_FILE_PATH = "./input/embeddings_2024-12-17_09-53-28.parquet"
VECTOR_STORING_AND_DELETION_BENCHMARKING_RESULTS_BASE_FILE_PATH = (
    "./results/vector_storing_and_deletion_results_"
)
VECTOR_SEARCH_BENCHMARKING_RESULTS_BASE_FILE_PATH = "./results/vector_search_results_"
IMAGE_TO_COMPARE_WITH_PATH = "./app/search_data/test_1.jpg"
LABELED_DATASET_PATH = "./app/search_data/labeled_pictures.csv"

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
    embeddings = retrieve_embeddings_from_parquet_file(INPUT_FILE_PATH)
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

        vector_db_deletion_start = datetime.datetime.now()
        db.delete(COLLECTION_NAME)
        vector_db_deletion_end = datetime.datetime.now()

        deletion_time = (
            vector_db_deletion_end - vector_db_deletion_start
        ).total_seconds()

        logger.info(f"Iteration {i + 1} - Deletion Time: {deletion_time}s")

        benchmark_data.append(
            {
                "iteration": i + 1,
                "initialisation_time": initialisation_time,
                "insertion_time": insertion_time,
                "deletion_time": deletion_time,
                "memory_usage_initialisation": memory_usage_initialisation,
                "memory_usage_insertion": memory_usage_insertion,
            }
        )

    benchmark_df = pd.DataFrame(benchmark_data)
    complete_file_path = (
        VECTOR_STORING_AND_DELETION_BENCHMARKING_RESULTS_BASE_FILE_PATH
        + f"_size_{len(embeddings)}__"
        + str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        + ".csv"
    )
    benchmark_df.to_csv(complete_file_path, index=False)

    initialisation_times = benchmark_df["initialisation_time"]
    insertion_times = benchmark_df["insertion_time"]
    memory_usage_initialisation = benchmark_df["memory_usage_initialisation"]
    memory_usage_insertion = benchmark_df["memory_usage_insertion"]
    deletion_times = benchmark_df["deletion_time"]

    stats = {
        "initialisation_mean": np.mean(initialisation_times),
        "initialisation_std": np.std(initialisation_times),
        "insertion_mean": np.mean(insertion_times),
        "insertion_std": np.std(insertion_times),
        "deletion_mean": np.mean(deletion_times),
        "deletion_std": np.std(deletion_times),
        "initialisation_p90": np.percentile(initialisation_times, 90),
        "insertion_p90": np.percentile(insertion_times, 90),
        "deletion_p90": np.percentile(deletion_times, 90),
        "initialisation_p95": np.percentile(initialisation_times, 95),
        "insertion_p95": np.percentile(insertion_times, 95),
        "deletion_p95": np.percentile(deletion_times, 95),
        "initialisation_p99": np.percentile(initialisation_times, 99),
        "insertion_p99": np.percentile(insertion_times, 99),
        "deletion_p99": np.percentile(deletion_times, 99),
        "memory_usage_init_mean": np.mean(memory_usage_initialisation),
        "memory_usage_init_std": np.std(memory_usage_initialisation),
        "memory_usage_insert_mean": np.mean(memory_usage_insertion),
        "memory_usage_insert_std": np.std(memory_usage_insertion),
    }

    stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
    stats_df.to_csv(complete_file_path, mode="a", header=False, index=False)

    logger.info(f"Benchmark results saved to {complete_file_path}")

    db.insert(COLLECTION_NAME, embeddings)


def search_embedding(db, embedding, search_params):
    start_time = time.perf_counter()
    raw_predicted_results = db.search(COLLECTION_NAME, embedding, search_params)
    end_time = time.perf_counter()

    # parsed_predicted_results - list of unique image names where the faces has been found
    parsed_predicted_results = db.parse_search_results(raw_predicted_results)

    real_results = pd.read_csv(LABELED_DATASET_PATH)
    target_picture = IMAGE_TO_COMPARE_WITH_PATH.split("/")[-1]
    real_results[f"predicted_{target_picture}"] = real_results.apply(
        lambda row: 1 if row["picture_name"] in parsed_predicted_results else 0,
        axis=1,
    )

    y_true = real_results[target_picture]
    y_pred = real_results[f"predicted_{target_picture}"]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    logger.info(f"Precision {precision} recall {recall} f1 {f1}")

    return end_time - start_time, precision, recall, f1


def search_similar_embeddings(
    db,
    image_to_compare_with_path,
    search_params,
    num_threads=10,
    num_iterations=100,
):
    logger.info(f"Processing image: {image_to_compare_with_path}")
    image_embedding = process_input_image(image_to_compare_with_path)
    logger.info(f"Image: {image_to_compare_with_path} converted to embedding.")

    benchmark_data = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(search_embedding, db, image_embedding, search_params)
            for _ in range(num_iterations)
        ]

        start_time = time.perf_counter()
        successful_requests = 0

        for i, future in enumerate(as_completed(futures)):
            try:
                search_time, precision, recall, f1 = future.result()
                successful_requests += 1

                benchmark_data.append(
                    {
                        "iteration": i + 1,
                        "search_time": search_time,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1,
                    }
                )
                logger.info(
                    f"Iteration {i + 1} - Search Time: {search_time}s, Precision: {precision}, Recall: {recall}, F1: {f1}"
                )
            except Exception as e:
                logger.error(f"Error during search on iteration {i + 1}: {e}")

        end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    rps = successful_requests / elapsed_time if elapsed_time > 0 else 0

    search_times = [data["search_time"] for data in benchmark_data]

    stats = {
        "search_time_mean": np.mean(search_times),
        "search_time_std": np.std(search_times),
        "search_time_p90": np.percentile(search_times, 90),
        "search_time_p95": np.percentile(search_times, 95),
        "search_time_p99": np.percentile(search_times, 99),
        "rps": rps,
        "total_time": elapsed_time,
        "successful_requests": successful_requests,
    }

    benchmark_df = pd.DataFrame(benchmark_data)
    stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])

    complete_file_path = (
        f"{VECTOR_SEARCH_BENCHMARKING_RESULTS_BASE_FILE_PATH}threads_{num_threads}_iterations_{num_iterations}_"
        + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + ".csv"
    )

    benchmark_df.to_csv(complete_file_path, index=False)
    stats_df.to_csv(complete_file_path, mode="a", header=False, index=False)

    logger.info(f"Search benchmark results and stats saved to {complete_file_path}")
    logger.info(
        f"RPS: {rps}, Total Time: {elapsed_time}s, Successful Requests: {successful_requests}"
    )

    return stats


"""
Modify code below for the purposes of other vector database benchmarking.
"""

if __name__ == "__main__":
    logger = get_logger()

    db = get_vector_database("MILVUS")

    """
    Insert + Delete benchmarking
    """

    insert_embeddings(db)

    """
    Search benchmarking
    """

    search_params = search_params = {
        "anns_field": "embedding",
        "metric_type": "COSINE",
        "index_params": {"ef": 64},
        "limit": None,
        "threshold": 0.8,
        "output_fields": ["id", "image_path"],
    }

    search_similar_embeddings(
        db,
        IMAGE_TO_COMPARE_WITH_PATH,
        search_params,
        num_threads=50,
        num_iterations=100,
    )
