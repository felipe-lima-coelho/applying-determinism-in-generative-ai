"""Etapa substituível: valida, reutiliza embeddings e agrupa cada segmento isolado."""

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..domain import (
    Dataset, EmbeddingProfile, GroupingError, Vector, cosine_matrix, member_mean,
    validate_information_dataset, validate_partition, validate_vector,
)
from ..ports import EmbeddingProvider, GroupingStrategy, VectorStore, VectorStoreError


@dataclass(frozen=True)
class GroupingStage:
    provider: EmbeddingProvider
    vector_store: VectorStore
    profile: EmbeddingProfile
    strategy: GroupingStrategy
    threshold: float = 0.85

    def __post_init__(self) -> None:
        if type(self.threshold) not in (int, float) or not math.isfinite(self.threshold) or not -1 <= self.threshold <= 1:
            raise GroupingError("threshold deve ser um número finito entre -1 e 1.")

    def __call__(self, dataset: Dataset) -> Dataset:
        return self._group(dataset)

    @staticmethod
    def validate_dataset(dataset: Dataset) -> None:
        """Preflight puro, reutilizável antes de outras inferências pagas."""
        validate_information_dataset(dataset)

    def group_segments(self, dataset: Dataset, selected_segments: frozenset[tuple[int, int]]) -> Dataset:
        """Reagrupa apenas os segmentos selecionados, sem renumerar fontes/segmentos."""
        return self._group(dataset, selected_segments)

    def _group(self, dataset: Dataset, selected_segments: frozenset[tuple[int, int]] | None = None) -> Dataset:
        segments = validate_information_dataset(dataset)
        if selected_segments is not None:
            known = {(segment.source_index, segment.segment_index) for segment in segments}
            if (not isinstance(selected_segments, frozenset)
                    or any(not isinstance(key, tuple) or len(key) != 2 or any(type(index) is not int for index in key)
                           for key in selected_segments)
                    or not selected_segments.issubset(known)):
                raise GroupingError("selected_segments deve conter somente pares válidos de índices originais.")
            segments = tuple(segment for segment in segments
                             if (segment.source_index, segment.segment_index) in selected_segments)
        texts = tuple(dict.fromkeys(item.info for segment in segments for item in segment.occurrences))
        # Nenhuma dependência externa é acionada se não houver informações.
        vectors: dict[str, Vector] = {}
        if texts:
            cached = dict(self.vector_store.fetch(self.profile, texts))
            if set(cached) - set(texts):
                raise VectorStoreError("Cache retornou textos não solicitados.")
            dimension = self.profile.dimensions
            for value in cached.values():
                checked = validate_vector(value, dimension)
                dimension = dimension or len(checked)
            missing = tuple(text for text in texts if text not in cached)
            if missing:
                generated = self.provider.embed(missing)
                if not isinstance(generated, Sequence) or isinstance(generated, (str, bytes)) or len(generated) != len(missing):
                    raise GroupingError("Provedor deve retornar exatamente um vetor por texto solicitado.")
                pending = {}
                for text, value in zip(missing, generated):
                    vector = validate_vector(value, dimension)
                    dimension = dimension or len(vector)
                    pending[text] = vector
                self.vector_store.put(self.profile, pending)
            # Inclusive a primeira execução usa o que foi realmente persistido em float32.
            persisted = dict(self.vector_store.fetch(self.profile, texts))
            if set(persisted) != set(texts):
                raise VectorStoreError("Cache não devolveu todos os embeddings persistidos.")
            for text in texts:
                vector = validate_vector(persisted[text], dimension)
                dimension = dimension or len(vector)
                vectors[text] = vector

        result = copy.deepcopy(dataset)
        for segment in segments:
            matrix = cosine_matrix(tuple(vectors[item.info] for item in segment.occurrences))
            partition = validate_partition(self.strategy.partition(matrix, self.threshold), matrix, self.threshold)
            groups = [
                [
                    {"info": segment.occurrences[index].info, "context": segment.occurrences[index].context,
                     "avg_similarity": member_mean(index, group, matrix)}
                    for index in sorted(group)
                ]
                for group in sorted(partition, key=min)
            ]
            result[segment.source_index]["segments"][segment.segment_index]["set_different_information"] = groups
        return result
