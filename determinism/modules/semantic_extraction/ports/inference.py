from typing import Protocol

from ..domain import ExtractionError, InferenceResult, SegmentInput


class InferenceError(ExtractionError):
    """Falha de inferência; adaptadores devem fornecer diagnóstico sem segredos."""


class InferenceProvider(Protocol):
    """Implementações devem aceitar chamadas simultâneas e nunca alterar a entrada."""

    def extract(self, segment: SegmentInput) -> InferenceResult: ...
