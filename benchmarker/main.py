import pandas as pd
import datetime

from app.vector_databases import (
    connect_to_vector_database,
    initialise_vector_database,
    insert_embeddings_into_vector_database,
    drop_already_existing_collection,
)
from app.logger import get_logger

OUTPUT_FILE_PATH = "./output/embeddings_2024-12-16_17-30-59.parquet"


def retrieve_embeddings_from_parquet_file(file_path):
    try:
        embeddings = pd.read_parquet(file_path, engine="pyarrow")
        return embeddings
    except Exception as e:
        logger.error(
            f"An error occurred while retrieving embeddings from {file_path}: {e}"
        )


def main(vector_database="MILVUS"):
    logger.info("Retrieving extracted embeddings")
    embeddings = retrieve_embeddings_from_parquet_file(OUTPUT_FILE_PATH)
    logger.info("Embeddings retrieved successfully")

    logger.info(f"Connecting to the {vector_database} vector database")
    connect_to_vector_database(vector_database)
    drop_already_existing_collection(vector_database)
    logger.info(f"Successfully connected to the {vector_database} vector database")

    vector_db_initialisation_start_datetime = datetime.datetime.now()
    logger.info(
        f'Starting vector database initialisation at: {vector_db_initialisation_start_datetime.strftime("%Y-%m-%d_%H-%M-%S")}'
    )
    initialise_vector_database(vector_database)

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

    insert_embeddings_into_vector_database(embeddings, vector_database)

    vector_db_insertion_end_datetime = datetime.datetime.now()
    logger.info(
        f'Finished vector database insertion at: {vector_db_insertion_end_datetime.strftime("%Y-%m-%d_%H-%M-%S")}'
    )
    logger.info(
        f"Total vector database insertion time: {(vector_db_insertion_end_datetime-vector_db_insertion_start_datetime).total_seconds()}"
    )


if __name__ == "__main__":
    logger = get_logger("MILVUS")
    logger.info("Hello!")
    main()
