from abc import ABC, abstractmethod

class QVIMModel(ABC):

    @abstractmethod
    def compute_similarities(
            self, items: dict[str, str], queries: dict[str, str]
    ) -> dict[str, dict[str, float]]:
        """Compute similarity scores between items to be retrieved and a set of queries.

        Each <query, item> pairing should be assigned a single floating point score, where higher
        scores indicate higher similarity.

        Args:
            items (dict[str, str]): A dictionary mapping items to be retrieved ids to the corresponding file path
            queries (dict[str, str]): A dictionary mapping query ids to the corresponding file path

        Returns:
            scores (dict[str, dict[str, float]]): A dictionary mapping query ids to a dictionary of item
                ids and their corresponding similarity scores. E.g:
                {
                    "query_1": {
                        "item_1": 0.8,
                        "item_2": 0.6,
                        ...
                    },
                    "query_2": {
                        "item_1": 0.4,
                        "item_2": 0.9,
                        ...
                    },
                    ...
                }
        """
        pass