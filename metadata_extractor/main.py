from app.images import get_image_paths, convert_bytes_to_image
from app.logger import get_logger

from app.extractor.face_characteristics_extractor import FaceCharacteristicsExtractor
from app.extractor.face_position_extractor import FacePositionExtractor
from app.extractor.image_description_extractor import ImageDescriptionExtractor

import insightface
import numpy as np

DEFAULT_IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

logger = get_logger()


# TODO - Modify the logic to incorporate DB (all of this will be changed)
def load_faces_without_extracted_metadata(directory_path: str):
    image_paths = get_image_paths(
        directory_path=directory_path,
        supported_image_types=DEFAULT_IMAGE_TYPES,
    )
    gpu_enabled = False
    detection_model = insightface.app.FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider"] if gpu_enabled else None,
    )
    detection_model.prepare(ctx_id=0 if gpu_enabled else -1)

    cropped_faces = []
    images = []
    for image_path in image_paths:
        img = convert_bytes_to_image(image_path)
        img = np.array(img)
        images.append(img)
        faces = detection_model.get(img)
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            crop = img[y1:y2, x1:x2].astype(np.uint8)
            cropped_faces.append(crop)
    return cropped_faces, images


# TODO: Modify the input and the flow
# Order in which faces were detected on a certain picture are necessary to detect these characteristics
def extract_metadata(cropped_faces: list, images: list):
    for cropped_face in cropped_faces:
        face_characteristics_metadata = FaceCharacteristicsExtractor().extract(
            cropped_face
        )
        logger.info(f"Face characteristics metadata: {face_characteristics_metadata}")
    for image in images:
        # face_position_metadata = FacePositionExtractor().extract(image)
        image_description_metadata = ImageDescriptionExtractor().extract(image)
        logger.info(f"Image description metadata: {image_description_metadata}")
        break


# TODO: Add the insertion of the metadata to the DB
def save_metadata():
    pass


def main():
    cropped_faces, images = load_faces_without_extracted_metadata(
        "./images/NORTHSTORM/2024"
    )
    extract_metadata(cropped_faces, images)
    save_metadata()


if __name__ == "__main__":
    main()
