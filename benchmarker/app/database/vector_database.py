from abc import ABC, abstractmethod
from app.logger import get_logger

logger = get_logger()


def handle_exceptions(logger):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"An error occurred in {func.__name__}: {e}")
                raise

        return wrapper

    return decorator


class VectorDatabase(ABC):
    @abstractmethod
    @handle_exceptions(logger)
    def connect(self, host: str, port: int):
        pass

    @abstractmethod
    @handle_exceptions(logger)
    def drop_collection(self, collection_name: str):
        pass

    @abstractmethod
    @handle_exceptions(logger)
    def create_collection(self, collection_name: str):
        pass

    @abstractmethod
    @handle_exceptions(logger)
    def insert(self, collection_name: str, data: dict):
        pass

    @abstractmethod
    @handle_exceptions(logger)
    def search(self, collection_name: str, embedding: list, params: dict):
        pass

    @abstractmethod
    @handle_exceptions(logger)
    def delete(self, collection_name: str):
        pass

    @abstractmethod
    @handle_exceptions(logger)
    def parse_search_results(self, results: list):
        """
        Output is expected to be a list of unique image names on which the faces have been found.
        """
        pass
