import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def initialise_face_detector(is_short_range=False):
    # TODO: Add the long range model when it comes out
    path_to_model = "app/ml_models/blaze_face_long_range_face_detection_model.tflite"
    if is_short_range:
        path_to_model = (
            "app/ml_models/blaze_face_short_range_face_detection_model.tflite"
        )
    base_options = python.BaseOptions(model_asset_path=path_to_model)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)
    return detector


def extract_faces_from_mediapipe_detections(image_np: np.ndarray, detection_result):
    """
    Extract faces from an image based on detection results.

    :param image_np: Original image as a NumPy array (H, W, C)
    :param detection_result: The output from the MediaPipe face detector.

    :return: List of cropped face regions as NumPy arrays.
    """
    image_height, image_width, _ = image_np.shape
    face_images = []

    for detection in detection_result.detections:
        bbox = detection.bounding_box

        x_min = int(bbox.origin_x)
        y_min = int(bbox.origin_y)
        width = int(bbox.width)
        height = int(bbox.height)

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image_width, x_min + width)
        y_max = min(image_height, y_min + height)

        face_image = image_np[y_min:y_max, x_min:x_max]

        face_images.append(face_image)

    return face_images


def extract_faces_from_deepface_detections(detected_faces):
    face_images = []

    for face in detected_faces:
        face_image = face["face"]
        face_images.append(face_image)

    return face_images


def initialise_face_embedder():
    base_options = python.BaseOptions(
        model_asset_path="app/ml_models/picture_embeddings_model.tflite"
    )
    options = vision.ImageEmbedderOptions(
        base_options=base_options, l2_normalize=True, quantize=True
    )
    embedder = vision.ImageEmbedder.create_from_options(options)
    return embedder


def create_embedding(np_array, embedder):
    uint8_array = (np_array * 255).astype("uint8")
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=uint8_array)

    embedding_object = embedder.embed(mp_image)

    embedding = embedding_object.embeddings[0].embedding
    float_embedding = np.array(embedding, dtype=np.float32)

    return float_embedding
