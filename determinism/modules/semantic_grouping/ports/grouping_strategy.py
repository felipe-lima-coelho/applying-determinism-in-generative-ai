from typing import Protocol

from ..domain import Partition, SimilarityMatrix


class GroupingStrategy(Protocol):
    def partition(self, matrix: SimilarityMatrix, threshold: float) -> Partition: ...
