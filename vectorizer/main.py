import argparse
import datetime
import pandas as pd
import json
import torch

from app.embedder.face_embedder import FaceEmbedder
from app.embedder.insightface_embedder import InsightfaceEmbedder
from app.embedder.dino_embedder import DINOEmbedder
from app.embedder.mediapipe_embedder import MediapipeEmbedder
from app.embedder.facenet_embedder import FacenetEmbedder
from app.embedder.clip_embedder import CLIPEmbedder
from app.images import get_image_paths
from app.logger import get_logger

DEFAULT_IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
DEFAULT_OUTPUT_TEMPLATE = "./output/embeddings_{face_extraction_model}_{image_name}_{current_datetime}.parquet"
DEFAULT_COMPARISON_PATH = "./images/comparison/woman.JPG"
DEFAULT_COMPARISON_OUT = "./output/embedding_{face_extraction_model}_comparison.csv"

logger = get_logger()
models = {}


def initialize_models(model_name: str, gpu_enabled: bool):
    """Initialize exactly the requested embedder and store it in `models`."""
    global models
    models.clear()
    embedder = get_embedder(model_name)
    logger.info(f"Initializing {model_name} embedder…")
    embedder.init_model(gpu_enabled)
    models[model_name] = embedder


def process_image(model_name: str, image_path: str):
    """Returns list of {"embedding", "image_path"} dicts."""
    return models[model_name].process_image(image_path)


def get_embedder(name: str) -> FaceEmbedder:
    if name == "mediapipe":
        return MediapipeEmbedder()
    elif name == "insightface":
        return InsightfaceEmbedder()
    elif name == "dino":
        return DINOEmbedder()
    elif name == "facenet":
        return FacenetEmbedder()
    elif name == "clip":
        return CLIPEmbedder()
    else:
        raise ValueError(f"Unknown embedder: {name}")


def process_images_in_directory(model_name, gpu_enabled, directories, out_template):
    initialize_models(model_name, gpu_enabled)

    for directory_path in directories:
        image_name = directory_path.rstrip("/").split("/")[-1]
        image_files = get_image_paths(
            directory_path=directory_path,
            supported_image_types=DEFAULT_IMAGE_TYPES,
        )
        logger.info(f"Processing {len(image_files)} images in {directory_path}…")

        all_embeddings = []
        for img in image_files:
            all_embeddings.extend(process_image(model_name, img))

        logger.info(f"Embeddings size: {len(all_embeddings[0]['embedding'])}")

        df = pd.DataFrame(all_embeddings)
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_path = out_template.format(
            current_datetime=now,
            face_extraction_model=model_name,
            image_name=image_name,
        )
        df.to_parquet(out_path, compression="snappy")
        logger.info(f"Saved parquet file for {directory_path} → {out_path}")


def process_comparison_image(model_name, gpu_enabled, comp_path, out_path):
    initialize_models(model_name, gpu_enabled)
    embedding = process_image(model_name, comp_path)[0]
    df = pd.DataFrame(
        [
            {
                "embedding": json.dumps(embedding["embedding"].tolist()),
                "image_path": embedding["image_path"],
            }
        ]
    )
    final_out = out_path.format(face_extraction_model=model_name)
    df.to_csv(final_out, index=False)
    logger.info(f"Saved comparison embedding → {final_out}")


def main():
    parser = argparse.ArgumentParser(description="Face-vectorizer")
    parser.add_argument(
        "-m",
        "--model",
        choices=["mediapipe", "insightface", "dino", "facenet", "clip"],
        default="dino",
        help="Which face-embedding backend to use",
    )
    parser.add_argument(
        "-d",
        "--dir",
        nargs="+",
        required=True,
        help="One or more image directories to vectorize",
    )
    parser.add_argument(
        "--compare",
        metavar="PATH",
        help="Also vectorize a single comparison image at this path",
    )
    parser.add_argument(
        "--out-template",
        default=DEFAULT_OUTPUT_TEMPLATE,
        help="Parquet output path template (must contain {face_extraction_model}, {image_name}, {current_datetime})",
    )
    parser.add_argument(
        "--compare-out",
        default=DEFAULT_COMPARISON_OUT,
        help="CSV output path template for comparison image (must contain {face_extraction_model})",
    )

    args = parser.parse_args()
    gpu = torch.cuda.is_available()
    logger.info(f"GPU enabled: {gpu}")
    start = datetime.datetime.now()
    logger.info(f"Start vectorizing at: {start}")

    logger.info(f"Vectorising with model {args.model} and directory {args.dir}")

    process_images_in_directory(
        model_name=args.model,
        gpu_enabled=gpu,
        directories=args.dir,
        out_template=args.out_template,
    )

    if args.compare:
        process_comparison_image(
            model_name=args.model,
            gpu_enabled=gpu,
            comp_path=args.compare,
            out_path=args.compare_out,
        )

    end = datetime.datetime.now()
    logger.info(f"Finished at: {end} (took {(end - start).total_seconds()}s)")


if __name__ == "__main__":
    main()
