import insightface

import numpy as np

from app.embedder.face_embedder import FaceEmbedder
from app.face_detection import create_embedding, initialise_face_embedder
from app.images import convert_bytes_to_image


class MediapipeEmbedder(FaceEmbedder):
    def init_model(self, gpu_enabled=False):
        self.embedder = initialise_face_embedder()
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
            emb = create_embedding(crop, self.embedder)
            results.append({"embedding": emb, "image_path": image_path})
        return results
