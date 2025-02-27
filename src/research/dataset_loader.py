import jsonlines
import logging

logger = logging.getLogger(__name__)


class GoldDatasetLoader:
    """Class to load and iterate over the gold dataset."""

    @staticmethod
    def load_gold_dataset(file_path: str, max_examples: int = None):
        """Yield examples from the gold dataset."""
        try:
            with jsonlines.open(file_path) as reader:
                for index, obj in enumerate(reader):
                    if max_examples and index >= max_examples:
                        break
                    yield obj
        except Exception as e:
            logger.error(f"Error loading dataset from {file_path}: {e}")
            raise
