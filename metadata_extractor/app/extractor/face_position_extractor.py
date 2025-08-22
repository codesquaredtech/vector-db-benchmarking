from .metadata_extractor import MetadataExtractor

import insightface
import onnxruntime as ort


def build_detector():
    gpu_enabled = "CUDAExecutionProvider" in ort.get_available_providers()
    det = insightface.app.FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider"] if gpu_enabled else None,
    )
    det.prepare(ctx_id=0 if gpu_enabled else -1)
    return det


# TODO: With respect to the cropped face sequence, update every row in DB with the metadata
class FacePositionExtractor(MetadataExtractor):
    def __init__(self, detector=None, logger=None):
        self.detector = detector or build_detector()

    def extract(self, image):
        img = image
        faces = self.detector.get(img) or []

        h, w = img.shape[:2]
        results = []

        for f in faces:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            yaw = pitch = roll = None
            if hasattr(f, "pose") and f.pose is not None:
                pose = f.pose.tolist() if hasattr(f.pose, "tolist") else list(f.pose)
                if len(pose) >= 3:
                    yaw, pitch, roll = float(pose[0]), float(pose[1]), float(pose[2])

            quality = (
                float(getattr(f, "det_score", None))
                if hasattr(f, "det_score")
                else None
            )
            metadata = {
                "bbox": [x1, y1, x2, y2],
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "quality": quality,
            }
            results.append(metadata)

        return results
