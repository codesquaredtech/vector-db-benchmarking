from multiprocessing import Pool

from deepface import DeepFace

from app.db.milvus import milvus_client
from app.face_detection import (create_embedding,
                                extract_faces_from_deepface_detections,
                                initialise_face_detector,
                                initialise_face_embedder)
from app.images import convert_bytes_to_image, get_image_paths
from app.logger import get_logger

SUPPORTED_IMAGE_TYPES = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
REFERENT_IMAGE_DIRECTORY = './images/'
CHUNK_SIZE = 100
POOL_PROCESSES = 4


def process_image(image_path):
    logger.info(f"Processing image: {image_path}")

    vectors_to_insert = []
    face_detector = initialise_face_detector(is_short_range=True)
    face_embedder = initialise_face_embedder()

    numpy_image, mediapipe_image = convert_bytes_to_image(image_path)
    detected_faces = DeepFace.extract_faces(
                                img_path=numpy_image,
                            )
    face_images = extract_faces_from_deepface_detections(
        detected_faces
    )

    logger.info(f"Number of faces on the image: {len(face_images)}")
    for image in face_images:
        vectors_to_insert.append(
            {
                "embedding": create_embedding(image, face_embedder),
                "image_path": image_path
            }
        )
    
    milvus_client.insert("faces_from_events", vectors_to_insert)


def process_images_in_directory(directory_path, chunk_size=100):
    image_files = get_image_paths(directory_path=directory_path, supported_image_types=SUPPORTED_IMAGE_TYPES)
    print(f"len of image_files: {len(image_files)}")

    with Pool(processes=POOL_PROCESSES) as pool:
        pool.map(process_image, image_files, chunksize=chunk_size)


def main():    
    import datetime
    start_datetime = datetime.datetime.now()
    logger.info(f'Starting vectorizing at: {start_datetime.strftime("%Y-%m-%d_%H-%M-%S")}')

    process_images_in_directory(REFERENT_IMAGE_DIRECTORY, chunk_size=CHUNK_SIZE)

    end_datetime = datetime.datetime.now()
    logger.info(f'Finished vectorizing at: {end_datetime.strftime("%Y-%m-%d_%H-%M-%S")}')
    logger.info(f'Total processing time: {(end_datetime-start_datetime).total_seconds()}')


if __name__ == "__main__":
    logger = get_logger()
    logger.info("Hello!")
    main()
