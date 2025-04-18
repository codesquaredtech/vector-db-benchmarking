import datetime
import json
import time
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

INPUT_FILE_PATHS = [
    "./input/embeddings_dino_Krstenje - 21. jun 2020. Bogdan_2025-04-05_11-16-39.parquet",
    "./input/embeddings_dino_Krstenje - 8. oktobar 2020. - Krtolica - Indjija_2025-04-05_11-16-39.parquet",
    "./input/embeddings_dino_Svadba - 11. Januar 2020. - Nikola_2025-04-05_11-16-39.parquet",
    "./input/embeddings_dino_Svadba - 11. Oktobar 2020. - Jelena i Stefan Bo Inside_2025-04-05_11-16-39.parquet",
    "./input/embeddings_dino_Svadba - 11. Septembar 2020. - Alaska Terasa - Mirjana i Aleksandar_2025-04-05_11-16-39.parquet",
    "./input/embeddings_dino_Svadba - 11. septembar 2020. - Jelena i Damir - Greenday_2025-04-05_11-16-39.parquet",
    "./input/embeddings_dino_Svadba - 12. jun 2021. - Ivana, Vidikovac_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 12. Septembar 2020. - Jasmina i Bojan - Kristal_2025-04-05_11-16-39.parquet",
    "./input/embeddings_dino_Svadba - 12. Septembar 2020. - Jelena i Milan - Piknik_2025-04-05_22-18-03.parquet",
    "./input/embeddings_dino_Svadba - 12. Septembar 2020. - Jelena i Srdjan - restoran Dunav_2025-04-06_10-39-26.parquet",
    "./input/embeddings_dino_Svadba - 13. jun 2020. - Malinovicevi_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 13. Septembar 2020. - Katarina i Veljko - Zal za mladost_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 13. Septembar 2020. - Kovilj_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 13. septembar 2020. - Marija i Darko - Vrdnicka kula_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 14. Novembar 2020. - Zeljka i Nebojsa_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 15. maj 2021. Sandra i Dragan_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 15. Novembar 2020. - Danijel i Dijana_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 16. Oktobar 2020. - Tamara i Sinisa - GreenDay_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 17. April 2021. - Brana i Nemanja_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 17. Oktobar 2020. - Nina i Nenad - RiverSide pool_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 18. oktobar 2020. - Ana Marija i Juda - Vidikovac_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 2. februar 2020. - Milica i Jovan - Vidikovac_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 2. Oktobar 2020. - Maja (marijana) i Bojan_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 20. septembar 2020. Ubovic_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 21. Avgust 2021. - Natasa i Predrag, Marina_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 21. Avgust 2021. - Nevena i Nemanja, Sombor_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 21. Jun 2021. - Reset, Jelena i_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 21. maj 2021. Tamara i Nikola - salac Bulac_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 22. Novembar 2020. - Jovana i Vladimir_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 23. Januar 2021. - Teodora i D_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 23. Oktobar 2021. - Jovana i Milos_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 24. Septembar 2021. - Dijana i Milos_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 25. April 2020.  - Tijana i Petar - samo maticar_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 25. Jun 2021. - Teodora i Nemanja Jovic - Vidikovac_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 25. Septembar 2020. - Kesten_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 25. septembar 2021. - Jelena i Dusan - Fontana, B. Palanka_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 25. Septembar 2021. - Stasa i Igor, Alaska Barka_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 26. jul 2020. - Marko Marinkovic_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 26. jul 2020. - Slobodanka i Ilija_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 26. jun 2021. - Zorana i Nikola_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 26. septembar 2021. - Milana i Sretko, Kum_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 26. septembar 2021. - Milica i Aleksandar - Alaska barka_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 27. jun 2020. - Eksluziv - Teodora i Bojan_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 27. Jun 2021. - Gordana i Momir - Zal za Mladost_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 28. Avgust 2021. - Marijana i Milos, Subotica_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 28. jun 2020. - Duska i Marko - Kum_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 28. jun 2020. - Vidikovac - Sandra i Miroslav_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 28. jun 2021. - Jovana i Vladimir_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 29. Avgust 2020 - Bojana i Marko - Sasin Salas_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 29. avgust 2020. - Iva i Mihailo_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 29. avgust 2020. - Zal za Mladost - Ana i Aleksandar_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 29. maj 2021. Jelena i Vladan - B. Petrovac_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 3. jul 2020. - Sanja i Nemanja - Alaska Barka_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 3. Jul 2021. - Bojana i Mladen_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 3. oktobar 2020. - Aleksandar - Alaska Barka_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 3. Oktobar 2020. - Tijana i Zarko_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 31. Maj 2020. - Kostresevic_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 4. jun 2020. - Sonja i Milos - yellow house_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 4. Oktobar 2020. - Katarina i Srdjan - Ada_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 5. jun 2021. - Andjela, West Exit_2025-03-29_16-25-13.parquet",
    "./input/embeddings_dino_Svadba - 5. Septembar 2020. - Dragana i Nikola_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 6. jun 2020. - Beskraj_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadba - 6. Septembar 2020. - Bojana_2025-04-06_12-01-18.parquet",
    "./input/embeddings_dino_Svadbe - 16. maj 2020. - Biljana i Damir - Session_2025-04-06_12-01-18.parquet",
]


VECTOR_STORING_AND_DELETION_BENCHMARKING_RESULTS_BASE_FILE_PATH = (
    "./results/vector_storing_and_deletion_results_"
)
VECTOR_SEARCH_BENCHMARKING_RESULTS_BASE_FILE_PATH = "./results/vector_search_results_"
EMBEDDINGS_TO_COMPARE_WITH_PATH = [
    "./app/search_data/embedding_dino_man.csv",
    "./app/search_data/embedding_dino_woman.csv",
]
LABELED_DATASET_PATHS = {
    "man.JPG": "./app/search_data/medium_man_updated.csv",
    "woman.JPG": "./app/search_data/medium_woman_updated.csv",
}

COLLECTION_NAME = "Faces"
NUM_ITERATIONS = 5
DATABASE_FOR_BENCHMARKING = "ELASTICSEARCH"
VECTOR_SIZE = 768  # 1280 for mediapipe, 512 for insightface, 768 for dino


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


def retrieve_embeddings_from_parquet_files(file_paths):
    embeddings_list = []
    for file_path in file_paths:
        try:
            embeddings = pd.read_parquet(file_path, engine="pyarrow")
            embeddings_list.append(embeddings)
        except Exception as e:
            logger.error(
                f"An error occurred while retrieving embeddings from {file_path}: {e}"
            )

    all_embeddings = pd.concat(embeddings_list, ignore_index=True)
    # logger.info(all_embeddings.loc[0])
    return all_embeddings


def retrieve_embedding_from_csv_file(embedding_path: str):
    df = pd.read_csv(embedding_path)

    df["embedding"] = df["embedding"].apply(lambda x: np.array(json.loads(x)))

    # Always one row only (one face)
    for _, row in df.iterrows():
        embedding_array = row["embedding"]
        image_path = row["image_path"]
    return embedding_array, image_path


def insert_embeddings(db, num_iterations=NUM_ITERATIONS):
    logger.info("Retrieving extracted embeddings")
    embeddings = retrieve_embeddings_from_parquet_files(INPUT_FILE_PATHS)
    logger.info(f"Embeddings retrieved successfully, total rows: {len(embeddings)}")

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
        db.create_collection(COLLECTION_NAME, VECTOR_SIZE)
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
        + f"_size_{len(embeddings)}__database_{DATABASE_FOR_BENCHMARKING}_"
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


def search_embedding(db, embedding, search_params, image_path):
    start_time = time.perf_counter()
    raw_predicted_results = db.search(COLLECTION_NAME, embedding, search_params)
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


def search_similar_embeddings(
    db,
    search_params,
    num_threads=10,
    num_iterations=100,
):
    embeddings = retrieve_embeddings_from_parquet_files(INPUT_FILE_PATHS)
    db.connect()
    db.create_collection(COLLECTION_NAME, VECTOR_SIZE)
    db.insert(COLLECTION_NAME, embeddings)
    for embedding_path in EMBEDDINGS_TO_COMPARE_WITH_PATH:
        logger.info(f"Processing {embedding_path} path...")
        image_embedding, image_path = retrieve_embedding_from_csv_file(embedding_path)
        benchmark_data = []

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(
                    search_embedding, db, image_embedding, search_params, image_path
                )
                for _ in range(num_iterations)
            ]

            start_time = time.perf_counter()
            successful_requests = 0

            for i, future in enumerate(as_completed(futures)):
                try:
                    search_time, precision, recall, f1, specificity, far, frr = (
                        future.result()
                    )
                    successful_requests += 1

                    benchmark_data.append(
                        {
                            "iteration": i + 1,
                            "search_time": search_time,
                            "precision": precision,
                            "recall": recall,
                            "f1_score": f1,
                            "specificity": specificity,
                            "far": far,
                            "frr": frr,
                        }
                    )
                    logger.info(
                        f"Iteration {i + 1} - Search Time: {search_time}s, Precision: {precision}, Recall: {recall}, F1: {f1}, Specificity: {specificity}, FAR: {far}, FRR: {frr}"
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
            f"{VECTOR_SEARCH_BENCHMARKING_RESULTS_BASE_FILE_PATH}threads_{num_threads}_iterations_{num_iterations}_database_{DATABASE_FOR_BENCHMARKING}_"
            + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + ".csv"
        )

        benchmark_df.to_csv(complete_file_path, index=False)
        stats_df.to_csv(complete_file_path, mode="a", header=False, index=False)

        logger.info(f"Search benchmark results and stats saved to {complete_file_path}")
        logger.info(
            f"RPS: {rps}, Total Time: {elapsed_time}s, Successful Requests: {successful_requests}"
        )


"""
Modify code below for the purposes of other vector database benchmarking.
"""

if __name__ == "__main__":
    logger = get_logger()
    db = get_vector_database(DATABASE_FOR_BENCHMARKING)
    search_params = {"certainty": 0.6, "limit": 10000, "num_candidates": 10000}

    search_similar_embeddings(
        db,
        search_params,
        num_threads=10,
        num_iterations=100,
    )
"""
    benchmark_data = []
    benchmark_data.append(
        {
            "iteration": 1,
            "initialisation_time": 0.260406,
            "insertion_time": 6121.465655,
            "deletion_time": 4.865525,
            "memory_usage_initialisation": 0.38671875,
            "memory_usage_insertion": -1819.5390625,
        }
    )
    benchmark_data.append(
        {
            "iteration": 2,
            "initialisation_time": 0.316565,
            "insertion_time": 6336.585194,
            "deletion_time": 2.594859,
            "memory_usage_initialisation": 0.0,
            "memory_usage_insertion": -1802.453125,
        }
    )
    benchmark_data.append(
        {
            "iteration": 3,
            "initialisation_time": 0.361071,
            "insertion_time": 6211.31512,
            "deletion_time": 4.722099,
            "memory_usage_initialisation": 0.0,
            "memory_usage_insertion": 38.80859375,
        }
    )
    benchmark_data.append(
        {
            "iteration": 4,
            "initialisation_time": 0.347883,
            "insertion_time": 6192.77448,
            "deletion_time": 0.178218,
            "memory_usage_initialisation": 0.20703125,
            "memory_usage_insertion": 16.37109375,
        }
    )
    benchmark_data.append(
        {
            "iteration": 5,
            "initialisation_time": 0.302349,
            "insertion_time": 6274.726453,
            "deletion_time": 0.192311,
            "memory_usage_initialisation": 0.0,
            "memory_usage_insertion": -1184.583152,
        }
    )
    benchmark_df = pd.DataFrame(benchmark_data)
    complete_file_path = (
        VECTOR_STORING_AND_DELETION_BENCHMARKING_RESULTS_BASE_FILE_PATH
        + "_size_1732832__database_QDRANT_"
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
"""
