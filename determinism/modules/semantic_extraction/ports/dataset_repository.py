from pathlib import Path
from typing import Protocol

from ..domain import Dataset


class DatasetRepository(Protocol):
    def load(self, path: Path) -> Dataset: ...

    def save(self, path: Path, dataset: Dataset) -> None: ...
