"""Definições explícitas dos experimentos científicos executáveis."""

from .word_mapping import WordMappingExperiment, map_dataset, mapping_pipeline, run_mapping
from .semantic_extraction import (
    SemanticExtractionExperiment, extract_dataset, extraction_pipeline, run_extraction,
)

__all__ = [
    "WordMappingExperiment", "map_dataset", "mapping_pipeline", "run_mapping",
    "SemanticExtractionExperiment", "extract_dataset", "extraction_pipeline", "run_extraction",
]
