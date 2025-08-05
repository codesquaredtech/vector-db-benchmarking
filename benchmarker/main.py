import datetime
import json
import time
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Milvus
from app.database.milvus_database import MilvusDatabase

# PGVector
from app.database.pgvector_database import PGVectorDatabase

# Qdrant
from app.database.qdrant_database import QdrantDatabase

# Weaviate
from app.database.weaviate_database import WeaviateDatabase

# Chroma
from app.database.chroma_database import ChromaDatabase
from app.logger import get_logger

# Elasticsearch
from app.database.elasticsearch_database import ElasticsearchDatabase

"""
Modify global variables if needed.
"""
INPUT_FOLDER_PATH = "./input/insightface"
VECTOR_STORING_AND_DELETION_BENCHMARKING_RESULTS_BASE_FILE_PATH = (
    "./results/vector_storing_and_deletion_results_"
)
VECTOR_SEARCH_BENCHMARKING_RESULTS_BASE_FILE_PATH = "./results/vector_search_results_"
EMBEDDINGS_TO_COMPARE_WITH_PATH = [
    "./app/search_data/embedding_insightface_man.csv",
    "./app/search_data/embedding_insightface_woman.csv",
]
LABELED_DATASET_PATHS = {
    "man.JPG": "./app/search_data/small_man_updated.csv",
    "woman.JPG": "./app/search_data/small_woman_updated.csv",
}


def retrieve_embeddings_from_parquet_folder(folder_path):
    embeddings_list = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".parquet"):
            file_path = os.path.join(folder_path, file_name)
            try:
                embeddings = pd.read_parquet(file_path, engine="pyarrow")
                embeddings_list.append(embeddings)
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

    if not embeddings_list:
        raise ValueError(f"No valid parquet embeddings found in folder: {folder_path}")

    all_embeddings = pd.concat(embeddings_list, ignore_index=True)
    return all_embeddings


def retrieve_embedding_from_csv_file(embedding_path: str):
    df = pd.read_csv(embedding_path)

    df["embedding"] = df["embedding"].apply(lambda x: np.array(json.loads(x)))

    # Always one row only (one face)
    for _, row in df.iterrows():
        embedding_array = row["embedding"]
        image_path = row["image_path"]
    return embedding_array, image_path


# TODO: This isn't correct; Should be modified to follow docker's memory.
def measure_memory(process):
    """Returns memory usage in MB."""
    return process.memory_info().rss / (1024**2)


def prepare_database(db, collection_name, vector_size):
    """Connects and prepares the collection."""
    db.connect()
    db.drop_collection(collection_name)
    db.create_collection(collection_name, vector_size)


def run_single_iteration(
    db, collection_name, vector_size, embeddings, process, iteration_num
):
    """Runs one benchmark iteration (init, insert, delete)."""
    logger.info(f"Starting benchmark iteration {iteration_num}")

    prepare_database(db, collection_name, vector_size)

    memory_before_init = measure_memory(process)
    init_start = datetime.datetime.now()
    db.create_collection(collection_name, vector_size)
    init_end = datetime.datetime.now()
    memory_after_init = measure_memory(process)

    memory_before_insert = measure_memory(process)
    insert_start = datetime.datetime.now()
    db.insert(collection_name, embeddings)
    insert_end = datetime.datetime.now()
    memory_after_insert = measure_memory(process)

    delete_start = datetime.datetime.now()
    db.delete(collection_name)
    delete_end = datetime.datetime.now()

    return {
        "iteration": iteration_num,
        "initialisation_time": (init_end - init_start).total_seconds(),
        "insertion_time": (insert_end - insert_start).total_seconds(),
        "deletion_time": (delete_end - delete_start).total_seconds(),
        "memory_usage_initialisation": memory_after_init - memory_before_init,
        "memory_usage_insertion": memory_after_insert - memory_before_insert,
    }


def summarize_benchmark_results(
    benchmark_data, output_prefix, database_name, embedding_count
):
    """Saves raw and statistical benchmark results to CSV."""
    benchmark_df = pd.DataFrame(benchmark_data)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    complete_file_path = f"{output_prefix}_size_{embedding_count}__database_{database_name}_{timestamp}.csv"

    benchmark_df.to_csv(complete_file_path, index=False)

    stats = {
        "initialisation_mean": np.mean(benchmark_df["initialisation_time"]),
        "initialisation_std": np.std(benchmark_df["initialisation_time"]),
        "insertion_mean": np.mean(benchmark_df["insertion_time"]),
        "insertion_std": np.std(benchmark_df["insertion_time"]),
        "deletion_mean": np.mean(benchmark_df["deletion_time"]),
        "deletion_std": np.std(benchmark_df["deletion_time"]),
        "initialisation_p90": np.percentile(benchmark_df["initialisation_time"], 90),
        "insertion_p90": np.percentile(benchmark_df["insertion_time"], 90),
        "deletion_p90": np.percentile(benchmark_df["deletion_time"], 90),
        "initialisation_p95": np.percentile(benchmark_df["initialisation_time"], 95),
        "insertion_p95": np.percentile(benchmark_df["insertion_time"], 95),
        "deletion_p95": np.percentile(benchmark_df["deletion_time"], 95),
        "initialisation_p99": np.percentile(benchmark_df["initialisation_time"], 99),
        "insertion_p99": np.percentile(benchmark_df["insertion_time"], 99),
        "deletion_p99": np.percentile(benchmark_df["deletion_time"], 99),
        "memory_usage_init_mean": np.mean(benchmark_df["memory_usage_initialisation"]),
        "memory_usage_init_std": np.std(benchmark_df["memory_usage_initialisation"]),
        "memory_usage_insert_mean": np.mean(benchmark_df["memory_usage_insertion"]),
        "memory_usage_insert_std": np.std(benchmark_df["memory_usage_insertion"]),
    }

    stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
    stats_df.to_csv(complete_file_path, mode="a", header=False, index=False)

    logger.info(f"Benchmark results saved to {complete_file_path}")


def insert_embeddings(db, num_iterations, collection_name, vector_size, database_name):
    """Top-level orchestration of the embedding insertion benchmark."""

    logger.info("Retrieving extracted embeddings")
    embeddings = retrieve_embeddings_from_parquet_folder(INPUT_FOLDER_PATH)
    logger.info(f"Embeddings retrieved successfully, total rows: {len(embeddings)}")

    benchmark_data = []
    process = psutil.Process()

    for i in range(num_iterations):
        result = run_single_iteration(
            db, collection_name, vector_size, embeddings, process, i + 1
        )
        logger.info(
            f"Iteration {i + 1} - Init: {result['initialisation_time']}s, "
            f"Insert: {result['insertion_time']}s, Delete: {result['deletion_time']}s, "
            f"Mem Init: {result['memory_usage_initialisation']} MB, Mem Insert: {result['memory_usage_insertion']} MB"
        )
        benchmark_data.append(result)

    summarize_benchmark_results(
        benchmark_data,
        output_prefix=VECTOR_STORING_AND_DELETION_BENCHMARKING_RESULTS_BASE_FILE_PATH,
        database_name=database_name,
        embedding_count=len(embeddings),
    )


def search_embedding(db, collection_name, embedding, search_params, image_path):
    start_time = time.perf_counter()
    raw_predicted_results = db.search(collection_name, embedding, search_params)
    end_time = time.perf_counter()

    # parsed_predicted_results - list of unique image names where the faces have been found
    parsed_predicted_results = db.parse_search_results(raw_predicted_results)

    target_picture = image_path.split("/")[-1]
    logger.info(f"target picture {target_picture}")
    real_results = pd.read_csv(LABELED_DATASET_PATHS[target_picture])
    real_results[f"predicted_{target_picture}"] = real_results.apply(
        lambda row: 1 if row["picture_name"] in parsed_predicted_results else 0,
        axis=1,
    )

    y_true = real_results[target_picture]
    y_pred = real_results[f"predicted_{target_picture}"]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # mAP would require us to use the limit parameter instead of the cosine similarity threshold everywhere
    specificity = tn / (tn + fp)
    far = fp / (fp + tn)
    frr = fn / (fn + tp)

    logger.info(
        f"Precision {precision} recall {recall} f1 {f1} Specificity {specificity} far {far} frr {frr}"
    )

    return end_time - start_time, precision, recall, f1, specificity, far, frr


def prepare_database_with_embeddings(db, collection_name, vector_size):
    embeddings = retrieve_embeddings_from_parquet_folder(INPUT_FOLDER_PATH)
    db.connect()
    db.create_collection(collection_name, vector_size)
    db.insert(collection_name, embeddings)


def run_search_iteration(
    db, collection_name, image_embedding, search_params, image_path, iteration
):
    try:
        search_time, precision, recall, f1, specificity, far, frr = search_embedding(
            db, collection_name, image_embedding, search_params, image_path
        )
        return {
            "iteration": iteration,
            "search_time": search_time,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "specificity": specificity,
            "far": far,
            "frr": frr,
        }
    except Exception as e:
        logger.error(f"Error during search on iteration {iteration}: {e}")
        return None


def benchmark_search_for_embedding(
    db,
    collection_name,
    image_embedding,
    image_path,
    search_params,
    num_threads,
    num_iterations,
):
    logger.info(f"Running benchmark for image: {image_path}")
    benchmark_data = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(
                run_search_iteration,
                db,
                collection_name,
                image_embedding,
                search_params,
                image_path,
                i + 1,
            )
            for i in range(num_iterations)
        ]

        start_time = time.perf_counter()
        for future in as_completed(futures):
            result = future.result()
            if result:
                benchmark_data.append(result)
        end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    rps = len(benchmark_data) / elapsed_time if elapsed_time > 0 else 0

    return benchmark_data, elapsed_time, rps


def summarize_search_results(
    benchmark_data, database_name, num_threads, num_iterations
):
    benchmark_df = pd.DataFrame(benchmark_data)
    search_times = benchmark_df["search_time"]

    stats = {
        "search_time_mean": np.mean(search_times),
        "search_time_std": np.std(search_times),
        "search_time_p90": np.percentile(search_times, 90),
        "search_time_p95": np.percentile(search_times, 95),
        "search_time_p99": np.percentile(search_times, 99),
        "rps": len(benchmark_data) / search_times.sum()
        if search_times.sum() > 0
        else 0,
        "total_time": search_times.sum(),
        "successful_requests": len(benchmark_data),
    }

    stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    complete_file_path = (
        f"{VECTOR_SEARCH_BENCHMARKING_RESULTS_BASE_FILE_PATH}threads_{num_threads}_iterations_{num_iterations}_database_{database_name}_"
        f"{timestamp}.csv"
    )

    benchmark_df.to_csv(complete_file_path, index=False)
    stats_df.to_csv(complete_file_path, mode="a", header=False, index=False)

    logger.info(f"Search benchmark results and stats saved to {complete_file_path}")
    logger.info(
        f"RPS: {stats['rps']}, Total Time: {stats['total_time']}s, Successful Requests: {stats['successful_requests']}"
    )


def search_similar_embeddings(
    db,
    collection_name,
    vector_size,
    search_params,
    database_name,
    num_threads=10,
    num_iterations=100,
):
    prepare_database_with_embeddings(db, collection_name, vector_size)

    for embedding_path in EMBEDDINGS_TO_COMPARE_WITH_PATH:
        logger.info(f"Processing query embedding: {embedding_path}")
        image_embedding, image_path = retrieve_embedding_from_csv_file(embedding_path)

        benchmark_data, _, _ = benchmark_search_for_embedding(
            db,
            collection_name,
            image_embedding,
            image_path,
            search_params,
            num_threads,
            num_iterations,
        )

        summarize_search_results(
            benchmark_data,
            database_name=database_name,
            num_threads=num_threads,
            num_iterations=num_iterations,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Vector DB Benchmarking Script")

    parser.add_argument(
        "--collection-name", type=str, default=os.getenv("COLLECTION_NAME", "Faces")
    )
    parser.add_argument(
        "--num-iterations", type=int, default=int(os.getenv("NUM_ITERATIONS", 3))
    )
    parser.add_argument(
        "--database", type=str, default=os.getenv("DATABASE", "ELASTICSEARCH")
    )
    parser.add_argument(
        "--vector-size", type=int, default=int(os.getenv("VECTOR_SIZE", 512))
    )

    return parser.parse_args()


def get_default_search_params(database_name):
    db = database_name.upper()
    if db == "WEAVIATE" or db == "QDRANT" or db == "PGVECTOR":
        return {"certainty": 0.6, "limit": 10000}
    elif db == "MILVUS":
        return {
            "anns_field": "embedding",
            "metric_type": "COSINE",
            "index_params": {"ef": 10000},
            "limit": None,
            "threshold": 0.6,
            "output_fields": ["id", "image_path"],
        }
    elif db == "CHROMA":
        return {"threshold": 0.6, "limit": 10000}
    elif db == "ELASTICSEARCH":
        return {"certainty": 0.6, "limit": 10000, "num_candidates": 10000}
    else:
        raise ValueError(f"No default search parameters defined for {db}")


def get_vector_database(db_type: str):
    if db_type == "MILVUS":
        return MilvusDatabase()
    elif db_type == "WEAVIATE":
        return WeaviateDatabase()
    elif db_type == "PGVECTOR":
        return PGVectorDatabase()
    elif db_type == "QDRANT":
        return QdrantDatabase()
    elif db_type == "ELASTICSEARCH":
        return ElasticsearchDatabase()
    elif db_type == "CHROMA":
        return ChromaDatabase()
    else:
        raise ValueError(f"Unsupported vector database: {db_type}")


"""
Modify code below for the purposes of other vector database benchmarking.
"""

if __name__ == "__main__":
    args = parse_args()
    logger = get_logger()

    COLLECTION_NAME = args.collection_name
    NUM_ITERATIONS = args.num_iterations
    DATABASE_FOR_BENCHMARKING = args.database.upper()
    VECTOR_SIZE = args.vector_size

    logger.info(
        f"Running benchmark with config: DATABASE={DATABASE_FOR_BENCHMARKING}, COLLECTION={COLLECTION_NAME}, ITERATIONS={NUM_ITERATIONS}, VECTOR_SIZE={VECTOR_SIZE}"
    )

    db = get_vector_database(DATABASE_FOR_BENCHMARKING)

    insert_embeddings(
        db, NUM_ITERATIONS, COLLECTION_NAME, VECTOR_SIZE, DATABASE_FOR_BENCHMARKING
    )
