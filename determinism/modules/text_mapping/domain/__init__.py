"""Modelos e regras puras do mapeamento textual."""

from .dataset import Dataset, DatasetFormatError, validate_dataset
from .word_map import TokenEntry, WordMap, build_word_map

__all__ = ["Dataset", "DatasetFormatError", "TokenEntry", "WordMap", "build_word_map", "validate_dataset"]
