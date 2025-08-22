from abc import ABC, abstractmethod


class MetadataExtractor(ABC):
    """
    Abstract base class for metadata extraction.
    Concrete subclasses must implement the extract() method.
    """

    @abstractmethod
    def extract(self, image):
        """
        Extract metadata from the given input.

        Parameters:
            image: Any
                The input from which metadata should be extracted.

        Returns:
            dict
                A dictionary of extracted metadata.
        """
        pass
