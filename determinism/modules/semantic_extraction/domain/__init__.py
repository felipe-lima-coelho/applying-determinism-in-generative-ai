"""Contratos e validações puras da extração semântica."""

from .contracts import (
    CoverageFeedback,
    Dataset,
    ExtractionError,
    InferenceResult,
    Information,
    MapValidationError,
    ResponseValidationError,
    SegmentInput,
    Syntagm,
    TokenReference,
    validate_information,
    validate_mapped_dataset,
)
from .coverage import (
    MIN_TOKEN_COVERAGE,
    IncompleteCoverageError,
    add_information_counts,
    count_information,
    validate_coverage_feasibility,
)

__all__ = [
    "CoverageFeedback", "Dataset", "ExtractionError", "InferenceResult", "Information",
    "MapValidationError", "ResponseValidationError", "SegmentInput", "Syntagm", "TokenReference",
    "validate_information", "validate_mapped_dataset",
    "MIN_TOKEN_COVERAGE", "IncompleteCoverageError", "add_information_counts",
    "count_information", "validate_coverage_feasibility",
]
