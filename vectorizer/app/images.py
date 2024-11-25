import os

import mediapipe as mp
import numpy as np
from PIL import Image


def get_image_paths(directory_path, supported_image_types) -> list[str]:
    paths = []

    for root, _, files in os.walk(directory_path):
        for filename in files:
            if filename.lower().endswith(supported_image_types):
                paths.append(os.path.join(root, filename))
    return paths



def convert_bytes_to_image(image_path):
    image_pil = Image.open(image_path)
    numpy_image = np.array(image_pil)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
    return numpy_image, mp_image
