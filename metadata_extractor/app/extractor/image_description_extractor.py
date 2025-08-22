from .metadata_extractor import MetadataExtractor

import io, base64, json
import numpy as np
from PIL import Image
from openai import OpenAI

client = OpenAI()


def numpy_rgb_to_data_url(arr: np.ndarray, fmt="JPEG", quality=90) -> str:
    if arr.dtype != np.uint8:
        raise ValueError("Expected uint8 image array in range 0–255.")
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected shape (H, W, 3) RGB array.")

    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    save_kwargs = {"format": fmt.upper()}
    if fmt.upper() == "JPEG":
        save_kwargs.update({"quality": quality, "optimize": True})
    img.save(buf, **save_kwargs)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def get_schema():
    return {
        "type": "object",
        "properties": {
            "scored_tags": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    },
                    "required": ["tag", "confidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["scored_tags"],
        "additionalProperties": False,
    }


def get_prompt():
    return (
        "Look at the image and generate 8–15 concise search tags (nouns or short noun phrases). "
        "Assign each tag a confidence score between 0 and 1. "
        "Return tags in english."
        "Respond with JSON only."
    )


def get_response_format():
    return {
        "type": "json_schema",
        "json_schema": {"name": "image_tags", "schema": get_schema()},
    }


def get_system_message():
    return {
        "role": "system",
        "content": "You extract search tags from images and return JSON only.",
    }


def get_user_message(prompt, data_url):
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }


def retrieve_confident_tags(resp, confidence_score):
    scored = json.loads(resp.choices[0].message.content)["scored_tags"]
    filtered_tags = [t["tag"] for t in scored if t["confidence"] >= confidence_score]
    return filtered_tags


class ImageDescriptionExtractor(MetadataExtractor):
    def extract(self, image):
        data_url = numpy_rgb_to_data_url(image, fmt="JPEG")

        prompt = get_prompt()

        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            response_format=get_response_format(),
            messages=[
                get_system_message(),
                get_user_message(prompt, data_url),
            ],
        )

        filtered_tags = retrieve_confident_tags(resp, 0.6)
        return filtered_tags
