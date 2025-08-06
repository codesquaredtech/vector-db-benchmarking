import insightface
import open_clip
import torch

import numpy as np

from app.images import convert_bytes_to_image
from PIL import Image

from app.embedder.face_embedder import FaceEmbedder


class CLIPEmbedder(FaceEmbedder):
    def init_model(self, gpu_enabled=False):
        self.detection_model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider"] if gpu_enabled else None,
        )
        self.detection_model.prepare(ctx_id=0 if gpu_enabled else -1)

        self.device = torch.device("cuda" if gpu_enabled else "cpu")
        self.model, self.transform, _ = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def process_image(self, image_path):
        img, _ = convert_bytes_to_image(image_path)
        img = np.array(img)
        faces = self.detection_model.get(img)

        results = []
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            crop = img[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            pil = Image.fromarray(crop).convert("RGB")
            tensor = self.transform(pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feats = self.model.encode_image(tensor)
                emb = feats.cpu().numpy().flatten()

            results.append({"embedding": emb, "image_path": image_path})
        return results
