from itertools import chain
from multiprocessing import Pool

import pandas as pd
import numpy as np
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from deepface import DeepFace
import insightface
import timm

from app.face_detection import (
    create_embedding,
    extract_faces_from_deepface_detections,
    initialise_face_embedder,
)
from app.images import convert_bytes_to_image, get_image_paths
from app.logger import get_logger

SUPPORTED_IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
REFERENT_IMAGE_DIRECTORIES = ["./images/NORTHSTORM/2024"]
IMAGE_TO_COMPARE_WITH_PATH = "./images/comparison/test_1.jpg"
OUTPUT_EMBEDDING_TO_COMPARE_WITH_PATH = "./output/embedding_compare_with.csv"
CHUNK_SIZE = 1
POOL_PROCESSES = 1
FACE_EXTRACTION_MODEL = "insightface"  # mediapipe, insightface, dino
OUTPUT_FILE_PATH = (
    "./output/embeddings_{face_extraction_model}_{current_datetime}.parquet"
)
GPU_ENABLED = torch.cuda.is_available()


def process_image(image_path):
    logger.info(f"Processing image: {image_path}")

    numpy_image, _ = convert_bytes_to_image(image_path)
    vectors_to_insert = []

    logger.info(f"Extracting face embeddings with {FACE_EXTRACTION_MODEL}...")
    if FACE_EXTRACTION_MODEL == "mediapipe":
        logger.info("Initialising face embedder...")
        face_embedder = initialise_face_embedder()
        logger.info("Successfully initialised face embedder")

        logger.info("Extracting faces...")
        detected_faces = DeepFace.extract_faces(
            img_path=numpy_image,
            enforce_detection=False,
            detector_backend="retinaface",
            align=True,
        )
        logger.info("Successfully extracted faces")

        logger.info("Converting extracted faces...")
        face_images = extract_faces_from_deepface_detections(detected_faces)
        for image in face_images:
            vectors_to_insert.append(
                {
                    "embedding": create_embedding(image, face_embedder),
                    "image_path": image_path,
                }
            )
        logger.info("Successfully converted extracted faces")
        logger.info(f"Number of faces on the image: {len(face_images)}")

    elif FACE_EXTRACTION_MODEL == "insightface":
        logger.info("Initialising the model...")
        if GPU_ENABLED:
            model = insightface.app.FaceAnalysis(
                name="buffalo_l", providers=["CUDAExecutionProvider"]
            )
            model.prepare(ctx_id=0)
        else:
            model = insightface.app.FaceAnalysis(name="buffalo_l")
            model.prepare(ctx_id=-1)
        logger.info("Successfully initialised the model")

        logger.info("Extracting faces...")
        faces = model.get(numpy_image)

        if faces:
            logger.info("Successfully extracted the faces")
            for face in faces:
                logger.info("Converting the extracted faces...")
                embedding = face.normed_embedding
                vectors_to_insert.append(
                    {
                        "embedding": embedding,
                        "image_path": image_path,
                    }
                )
            logger.info("Successfully converted the extracted faces")
            logger.info(f"Number of faces on the image: {len(faces)}")
        else:
            logger.warning("No face detected in the image.")

    elif FACE_EXTRACTION_MODEL == "dino":
        device = torch.device("cuda" if GPU_ENABLED else "cpu")
        model = timm.create_model("vit_base_patch16_224", pretrained=True).to(device)
        model.eval()
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        logger.info("Extracting faces with DeepFace (RetinaFace)...")
        detected_faces = DeepFace.extract_faces(
            img_path=numpy_image,
            enforce_detection=False,
            detector_backend="retinaface",
            align=True,
        )
        face_images = extract_faces_from_deepface_detections(detected_faces)
        for face_image in face_images:
            face_image = (face_image * 255).astype(np.uint8)
            pil_image = Image.fromarray(face_image)
            input_tensor = transform(pil_image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.forward_features(input_tensor)[:, 0, :]
            embedding = embedding.cpu().numpy().flatten()
            vectors_to_insert.append(
                {
                    "embedding": embedding,
                    "image_path": image_path,
                }
            )
        logger.info("DINO embeddings generated.")

    else:
        raise ValueError(f"Unsupported face extraction model: {FACE_EXTRACTION_MODEL}")

    return vectors_to_insert


def process_images_in_directory(directory_paths, current_datetime, chunk_size=100):
    for directory_path in directory_paths:
        image_files = get_image_paths(
            directory_path=directory_path, supported_image_types=SUPPORTED_IMAGE_TYPES
        )
        logger.info(f"len of image_files: {len(image_files)}")

        with Pool(processes=POOL_PROCESSES) as pool:
            list_of_lists = pool.map(process_image, image_files, chunksize=chunk_size)
            flattened_list = list(chain.from_iterable(list_of_lists))
            # logger.info(f"Flattened list: {flattened_list}")

            df = pd.DataFrame(flattened_list)
            logger.info(f"Saving parquet file for {directory_path}...")
            df.to_parquet(
                OUTPUT_FILE_PATH.format(
                    current_datetime=current_datetime.strftime("%Y-%m-%d_%H-%M-%S"),
                    face_extraction_model=FACE_EXTRACTION_MODEL,
                ),
                compression="snappy",
            )
            logger.info(f"Saved parquet file for {directory_path}")


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

    logger.info(f"GPU enabled: {GPU_ENABLED}")
    start_datetime = datetime.datetime.now()
    logger.info(
        f"Starting vectorizing at: {start_datetime.strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    process_images_in_directory(
        REFERENT_IMAGE_DIRECTORIES,
        current_datetime=start_datetime,
        chunk_size=CHUNK_SIZE,
    )

    end_datetime = datetime.datetime.now()
    logger.info(
        f"Finished vectorizing at: {end_datetime.strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    logger.info(
        f"Total processing time: {(end_datetime - start_datetime).total_seconds()}"
    )
    # converting comparison image to an embedding
    logger.info("Starting vectorisation of the comparison image...")
    process_comparison_image()
    logger.info("Successfully vectorised the comparison image")


if __name__ == "__main__":
    logger = get_logger()
    main()
