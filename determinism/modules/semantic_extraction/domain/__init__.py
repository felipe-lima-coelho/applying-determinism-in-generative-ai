"""Contratos e validações puras da extração semântica."""

from .contracts import (
    Dataset,
    ExtractionError,
    InferenceResult,
    Information,
    MapValidationError,
    ResponseValidationError,
    SegmentInput,
    TokenReference,
    validate_information,
    validate_mapped_dataset,
)

__all__ = [
    "Dataset", "ExtractionError", "InferenceResult", "Information",
    "MapValidationError", "ResponseValidationError", "SegmentInput", "TokenReference",
    "validate_information", "validate_mapped_dataset",
]
