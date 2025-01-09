from itertools import chain
from multiprocessing import Pool

import pandas as pd
import json
from deepface import DeepFace

from app.face_detection import (
    create_embedding,
    extract_faces_from_deepface_detections,
    initialise_face_embedder,
)
from app.images import convert_bytes_to_image, get_image_paths
from app.logger import get_logger

SUPPORTED_IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
REFERENT_IMAGE_DIRECTORY = "./images/NORTHSTORM/2024/"
IMAGE_TO_COMPARE_WITH_PATH = "./images/comparison/test_1.jpg"
OUTPUT_FILE_PATH = "./output/embeddings_{current_datetime}.parquet"
OUTPUT_EMBEDDING_TO_COMPARE_WITH_PATH = "./output/embedding_compare_with.csv"
CHUNK_SIZE = 1
POOL_PROCESSES = 1


def process_image(image_path):
    logger.info(f"Processing image: {image_path}")

    vectors_to_insert = []
    face_embedder = initialise_face_embedder()

    numpy_image, _ = convert_bytes_to_image(image_path)
    detected_faces = DeepFace.extract_faces(
        img_path=numpy_image,
        enforce_detection=False,
        detector_backend="retinaface",
        align=True,
    )
    face_images = extract_faces_from_deepface_detections(detected_faces)

    logger.info(f"Number of faces on the image: {len(face_images)}")
    for image in face_images:
        vectors_to_insert.append(
            {
                "embedding": create_embedding(image, face_embedder),
                "image_path": image_path,
            }
        )

    return vectors_to_insert


def process_images_in_directory(directory_path, current_datetime, chunk_size=100):
    image_files = get_image_paths(
        directory_path=directory_path, supported_image_types=SUPPORTED_IMAGE_TYPES
    )
    logger.info(f"len of image_files: {len(image_files)}")

    with Pool(processes=POOL_PROCESSES) as pool:
        list_of_lists = pool.map(process_image, image_files, chunksize=chunk_size)
        flattened_list = list(chain.from_iterable(list_of_lists))
        # logger.info(f"Flattened list: {flattened_list}")

        df = pd.DataFrame(flattened_list)
        df.to_parquet(
            OUTPUT_FILE_PATH.format(
                current_datetime=current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
            ),
            compression="snappy",
        )


def process_comparison_image():
    embedding_with_path = process_image(IMAGE_TO_COMPARE_WITH_PATH)[0]
    df = pd.DataFrame(
        [
            {
                "embedding": json.dumps(embedding_with_path["embedding"].tolist()),
                "image_path": embedding_with_path["image_path"],
            }
        ]
    )

    df.to_csv(OUTPUT_EMBEDDING_TO_COMPARE_WITH_PATH, index=False)
    logger.info(
        f"Saved comparison embedding with path to {OUTPUT_EMBEDDING_TO_COMPARE_WITH_PATH}"
    )


def main():
    import datetime

    """
    start_datetime = datetime.datetime.now()
    logger.info(
        f'Starting vectorizing at: {start_datetime.strftime("%Y-%m-%d_%H-%M-%S")}'
    )

    process_images_in_directory(
        REFERENT_IMAGE_DIRECTORY, current_datetime=start_datetime, chunk_size=CHUNK_SIZE
    )

    end_datetime = datetime.datetime.now()
    logger.info(
        f'Finished vectorizing at: {end_datetime.strftime("%Y-%m-%d_%H-%M-%S")}'
    )
    logger.info(
        f"Total processing time: {(end_datetime-start_datetime).total_seconds()}"
    )
    """
    logger.info("Starting vectorisation of the comparison image...")
    process_comparison_image()
    logger.info("Successfully vectorised the comparison image")


if __name__ == "__main__":
    logger = get_logger()
    main()
