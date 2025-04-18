from itertools import chain

import insightface
import numpy as np
import pandas as pd
import timm
import json
import torch
import torchvision.transforms as transforms
from app.face_detection import create_embedding, initialise_face_embedder
from app.images import convert_bytes_to_image, get_image_paths
from app.logger import get_logger
from PIL import Image

SUPPORTED_IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
REFERENT_IMAGE_DIRECTORIES = ["./images/NORTHSTORM/2024"]
OUTPUT_FILE_PATH = "./output/embeddings_{face_extraction_model}_{image_name}_{current_datetime}.parquet"
FACE_EXTRACTION_MODEL = "dino"  # mediapipe, insightface, dino
GPU_ENABLED = torch.cuda.is_available()
IMAGE_TO_COMPARE_WITH_PATH = "./images/comparison/woman.JPG"
OUTPUT_EMBEDDING_TO_COMPARE_WITH_PATH = "./output/embedding_dino_woman.csv"
logger = get_logger()

# Global Model Storage
models = {}


def initialize_models():
    """Initialize models once and store in global state."""
    global models

    if FACE_EXTRACTION_MODEL == "mediapipe":
        models["mediapipe"] = initialise_face_embedder()

    elif FACE_EXTRACTION_MODEL == "insightface":
        logger.info("Initializing InsightFace model...")
        model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider"] if GPU_ENABLED else None,
        )
        model.prepare(ctx_id=0 if GPU_ENABLED else -1)
        models["insightface"] = model

    elif FACE_EXTRACTION_MODEL == "dino":
        logger.info("Initializing DINO model...")
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
        models["dino"] = (model, transform, device)

    # Face detection model
    detection_model = insightface.app.FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider"] if GPU_ENABLED else None,
    )
    detection_model.prepare(ctx_id=0 if GPU_ENABLED else -1)
    models["detection"] = detection_model


def process_image(image_path):
    """Process a single image, extracting face embeddings."""
    logger.info(f"Processing image: {image_path}")
    numpy_image, _ = convert_bytes_to_image(image_path)
    vectors_to_insert = []

    if FACE_EXTRACTION_MODEL == "mediapipe":
        face_embedder = models["mediapipe"]
        detector = models["detection"]

        img = np.array(numpy_image)
        detected_faces = detector.get(img)

        for face in detected_faces:
            bbox = face.bbox.astype(int)
            face_image = img[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            uint8_face_image = np.array(face_image, dtype=np.uint8)

            embedding = create_embedding(uint8_face_image, face_embedder)

            vectors_to_insert.append(
                {
                    "embedding": embedding,
                    "image_path": image_path,
                }
            )

    elif FACE_EXTRACTION_MODEL == "insightface":
        model = models["insightface"]
        faces = model.get(numpy_image)

        for face in faces:
            vectors_to_insert.append(
                {"embedding": face.normed_embedding, "image_path": image_path}
            )

    elif FACE_EXTRACTION_MODEL == "dino":
        model, transform, device = models["dino"]

        detection_model = models["detection"]
        faces = detection_model.get(numpy_image)

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)

            cropped_face = numpy_image[y1:y2, x1:x2]

            pil_image = Image.fromarray(cropped_face)
            pil_image = pil_image.convert("RGB")
            input_tensor = transform(pil_image).unsqueeze(0).to(device)

            with torch.no_grad():
                embedding = (
                    model.forward_features(input_tensor)[:, 0, :]
                    .cpu()
                    .numpy()
                    .flatten()
                )
                vectors_to_insert.append(
                    {"embedding": embedding, "image_path": image_path}
                )
    return vectors_to_insert


def process_images_in_directory(directory_paths, current_datetime):
    """Processes all images in given directories sequentially."""
    initialize_models()

    for directory_path in directory_paths:
        image_name = directory_path.split("/")[-1]
        image_files = get_image_paths(
            directory_path=directory_path, supported_image_types=SUPPORTED_IMAGE_TYPES
        )
        logger.info(f"Processing {len(image_files)} images in {directory_path}...")

        all_embeddings = []
        for image_path in image_files:
            all_embeddings.extend(process_image(image_path))

        df = pd.DataFrame(all_embeddings)
        df.to_parquet(
            OUTPUT_FILE_PATH.format(
                current_datetime=current_datetime.strftime("%Y-%m-%d_%H-%M-%S"),
                face_extraction_model=FACE_EXTRACTION_MODEL,
                image_name=image_name,
            ),
            compression="snappy",
        )
        logger.info(f"Saved parquet file for {directory_path}")


def process_comparison_image():
    initialize_models()
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
    """Main function to execute the face embedding pipeline."""
    """
    import datetime

    logger.info(f"GPU enabled: {GPU_ENABLED}")
    start_datetime = datetime.datetime.now()
    logger.info(
        f"Starting vectorizing at: {start_datetime.strftime('%Y-%m-%d_%H-%M-%S')}"
    )

    process_images_in_directory(REFERENT_IMAGE_DIRECTORIES, start_datetime)

    end_datetime = datetime.datetime.now()
    logger.info(
        f"Finished vectorizing at: {end_datetime.strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    logger.info(
        f"Total processing time: {(end_datetime - start_datetime).total_seconds()} seconds"
    )
    """
    logger.info("Starting vectorisation of the comparison image...")
    process_comparison_image()
    logger.info("Successfully vectorised the comparison image")


if __name__ == "__main__":
    main()
