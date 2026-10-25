"""Caso de uso que reconstrói os mapas de todos os segmentos."""

from copy import deepcopy
from dataclasses import dataclass

from ..domain import Dataset, build_word_map, validate_dataset
from ..ports import Tokenizer


@dataclass(frozen=True)
class MappingStage:
    tokenizer: Tokenizer

    def __call__(self, dataset: Dataset) -> Dataset:
        # Valide tudo antes de tokenizar ou construir qualquer resultado.
        validate_dataset(dataset)
        mapped = deepcopy(dataset)
        for source in mapped:
            for segment in source["segments"]:
                segment["map"] = build_word_map(self.tokenizer.tokenize(segment["content"]))
        return mapped
