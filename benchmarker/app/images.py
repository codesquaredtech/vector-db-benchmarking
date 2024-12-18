import mediapipe as mp
import numpy as np
from PIL import Image


def convert_bytes_to_image(image_path):
    image_pil = Image.open(image_path)
    numpy_image = np.array(image_pil)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
    return numpy_image, mp_image
