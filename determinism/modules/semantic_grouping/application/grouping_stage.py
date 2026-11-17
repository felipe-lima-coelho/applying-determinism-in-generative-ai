"""Valida, agrupa e escolhe uma variante canônica por grupo de cada segmento."""

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..domain import (
    Dataset, EmbeddingProfile, GroupingError, Vector, canonical_representative, cosine_matrix, member_mean,
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
        segments = validate_information_dataset(dataset)
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
            groups = []
            canonical_information = []
            for group in sorted(partition, key=min):
                indices = tuple(sorted(group))
                scores = {index: member_mean(index, indices, matrix) for index in indices}
                groups.append([
                    {"info": segment.occurrences[index].info, "context": segment.occurrences[index].context,
                     "avg_similarity": scores[index]}
                    for index in indices
                ])
                representative = canonical_representative(segment.occurrences, scores)
                occurrence = segment.occurrences[representative]
                canonical_information.append({
                    "info": occurrence.info, "context": occurrence.context,
                    "representative_index": representative, "member_indices": list(indices),
                    "avg_similarity": scores[representative],
                })
            target = result[segment.source_index]["segments"][segment.segment_index]
            target["set_different_information"] = groups
            target["canonical_information"] = canonical_information
        return result
