import os
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
    try:
        image_pil = Image.open(image_path)
    except Exception as e:
        print(f"An error occurred: {e}. Continuing...")
        return []
    numpy_image = np.array(image_pil)
    return numpy_image
