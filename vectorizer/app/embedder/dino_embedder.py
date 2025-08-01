import insightface
import timm
import torch

import numpy as np
import torchvision.transforms as transforms

from app.images import convert_bytes_to_image
from PIL import Image

from app.embedder.face_embedder import FaceEmbedder


class DINOEmbedder(FaceEmbedder):
    def init_model(self, gpu_enabled=False):
        self.detection_model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider"] if gpu_enabled else None,
        )
        self.detection_model.prepare(ctx_id=0 if gpu_enabled else -1)

        self.device = torch.device("cuda" if gpu_enabled else "cpu")
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True).to(
            self.device
        )
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def process_image(self, image_path):
        img, _ = convert_bytes_to_image(image_path)
        img = np.array(img)
        faces = self.detection_model.get(img)

        results = []
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            crop = img[y1:y2, x1:x2]
            pil = Image.fromarray(crop).convert("RGB")
            tensor = self.transform(pil).unsqueeze(0).to(self.device)

            with torch.no_grad():
                feats = self.model.forward_features(tensor)
                emb = feats[:, 0, :].cpu().numpy().flatten()

            results.append({"embedding": emb, "image_path": image_path})
        return results
