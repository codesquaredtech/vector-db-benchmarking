import os
import numpy as np
import insightface
from deepface import DeepFace

from app.embedder.face_embedder import FaceEmbedder
from app.images import convert_bytes_to_image


class VGGFaceEmbedder(FaceEmbedder):
    def init_model(self, gpu_enabled=False):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0" if gpu_enabled else "-1"
        self.model_name = "VGG-Face"

        self.detection_model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider"] if gpu_enabled else None,
        )
        self.detection_model.prepare(ctx_id=0 if gpu_enabled else -1)

    def process_image(self, image_path):
        img, _ = convert_bytes_to_image(image_path)
        img = np.array(img)

        faces = self.detection_model.get(img)
        results = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            crop = img[y1:y2, x1:x2].astype(np.uint8)

            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                continue

            embedding_objs = DeepFace.represent(
                img_path=crop,
                model_name=self.model_name,
                enforce_detection=False,
            )

            if not isinstance(embedding_objs, list):
                embedding_objs = [embedding_objs]

            for obj in embedding_objs:
                results.append(
                    {
                        "embedding": obj["embedding"],
                        "image_path": image_path,
                    }
                )

        return results
