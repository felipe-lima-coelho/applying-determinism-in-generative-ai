"""Definições explícitas dos experimentos científicos executáveis."""

from .word_mapping import WordMappingExperiment, map_dataset, mapping_pipeline, run_mapping
from .semantic_extraction import (
    SemanticExtractionExperiment, extract_dataset, extraction_pipeline, run_extraction,
)
from .semantic_grouping import SemanticGroupingExperiment, group_dataset, grouping_pipeline, run_grouping

__all__ = [
    "WordMappingExperiment", "map_dataset", "mapping_pipeline", "run_mapping",
    "SemanticExtractionExperiment", "extract_dataset", "extraction_pipeline", "run_extraction",
    "SemanticGroupingExperiment", "group_dataset", "grouping_pipeline", "run_grouping",
]
