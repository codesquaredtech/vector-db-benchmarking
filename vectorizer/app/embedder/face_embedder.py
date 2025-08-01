from abc import ABC, abstractmethod


class FaceEmbedder(ABC):
    """Abstract base for all face-embedding extractors."""

    def __init__(self):
        self.embedder = None
        self.detection_model = None

    @abstractmethod
    def init_model(self, gpu_enabled: bool = False):
        """Load/embed the model(s) needed for detection & embedding."""
        ...

    @abstractmethod
    def process_image(self, image_path: str):
        """
        Given a file path, return list of dicts:
            {"embedding": np.ndarray, "image_path": image_path}
        """
        ...
