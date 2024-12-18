import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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
    normalised_float_embedding = (float_embedding - float_embedding.min()) / (
        float_embedding.max() - float_embedding.min()
    )

    return normalised_float_embedding
