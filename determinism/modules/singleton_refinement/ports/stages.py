"""Etapas substituíveis; índices sempre pertencem ao dataset completo."""

from collections.abc import Callable
from typing import Protocol

from ..domain import Dataset, SegmentKey


ProgressCallback = Callable[[Dataset], None]


class SelectedExtractor(Protocol):
    def validate_dataset(self, dataset: Dataset) -> None: ...

    def extract_segments(
        self, dataset: Dataset, selected_segments: frozenset[SegmentKey], *, generation_round: int = 1,
    ) -> Dataset: ...


class SelectedGrouper(Protocol):
    def validate_dataset(self, dataset: Dataset) -> None: ...

    def group_segments(self, dataset: Dataset, selected_segments: frozenset[SegmentKey]) -> Dataset: ...
