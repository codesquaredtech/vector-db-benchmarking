import insightface

from app.embedder.face_embedder import FaceEmbedder
from app.images import convert_bytes_to_image


class InsightfaceEmbedder(FaceEmbedder):
    def init_model(self, gpu_enabled=False):
        self.embedder = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider"] if gpu_enabled else None,
        )
        self.embedder.prepare(ctx_id=0 if gpu_enabled else -1)

    def process_image(self, image_path):
        img, _ = convert_bytes_to_image(image_path)
        faces = self.embedder.get(img)
        return [
            {"embedding": face.normed_embedding, "image_path": image_path}
            for face in faces
        ]
