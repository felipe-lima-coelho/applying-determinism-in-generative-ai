"""Catálogo global: cada representante local contribui uma única ocorrência."""

import copy
import math
from dataclasses import dataclass
from typing import Any

from ..domain import (
    Dataset, EmbeddingProfile, GroupingError, InformationOccurrence, canonical_representative,
    cosine_matrix, member_mean, validate_information_dataset, validate_partition, validate_vector,
)
from ..ports import GroupingStrategy, VectorStore, VectorStoreError


@dataclass(frozen=True)
class GlobalCatalogBuilder:
    vector_store: VectorStore | None
    profile: EmbeddingProfile
    strategy: GroupingStrategy
    threshold: float = 0.90

    def __post_init__(self) -> None:
        if type(self.threshold) not in (int, float) or not math.isfinite(self.threshold) or not -1 <= self.threshold <= 1:
            raise GroupingError("threshold deve ser um número finito entre -1 e 1.")

    def __call__(self, dataset: Dataset) -> dict[str, Any]:
        segments = validate_information_dataset(dataset)
        information = []
        for segment in segments:
            path = f"$[{segment.source_index}].segments[{segment.segment_index}].canonical_information"
            canonical = dataset[segment.source_index]["segments"][segment.segment_index].get("canonical_information")
            if not isinstance(canonical, list):
                raise GroupingError(f"{path} deve ser uma lista; execute o agrupamento local primeiro.")
            covered = []
            for canonical_index, item in enumerate(canonical):
                item_path = f"{path}[{canonical_index}]"
                if not isinstance(item, dict):
                    raise GroupingError(f"{item_path} deve ser um objeto canônico local.")
                members, representative = item.get("member_indices"), item.get("representative_index")
                if (not isinstance(members, list) or not members
                        or any(type(index) is not int or not 0 <= index < len(segment.occurrences) for index in members)
                        or members != sorted(set(members))
                        or type(representative) is not int or representative not in members):
                    raise GroupingError(f"{item_path} contém índices de proveniência inválidos.")
                occurrence = segment.occurrences[representative]
                if (item.get("info"), item.get("context")) != (occurrence.info, occurrence.context):
                    raise GroupingError(f"{item_path} difere de sua ocorrência original representativa.")
                covered.extend(members)
                information.append({
                    "info": occurrence.info, "context": occurrence.context,
                    "provenance": {
                        "source_index": segment.source_index, "segment_index": segment.segment_index,
                        "canonical_index": canonical_index, "representative_index": representative,
                        "member_indices": list(members),
                    },
                })
            if sorted(covered) != list(range(len(segment.occurrences))):
                raise GroupingError(f"{path} deve particionar cada ocorrência original exatamente uma vez.")

        texts = tuple(dict.fromkeys(item["info"] for item in information))
        vectors = {}
        if texts:
            if self.vector_store is None:
                raise VectorStoreError("Catálogo global não vazio requer o cache dos embeddings locais.")
            persisted = dict(self.vector_store.fetch(self.profile, texts))
            if set(persisted) != set(texts):
                raise VectorStoreError("Catálogo global requer todos os embeddings locais persistidos.")
            dimension = self.profile.dimensions
            for text in texts:
                vector = validate_vector(persisted[text], dimension)
                dimension = dimension or len(vector)
                vectors[text] = vector
        matrix = cosine_matrix(tuple(vectors[item["info"]] for item in information))
        partition = validate_partition(self.strategy.partition(matrix, self.threshold), matrix, self.threshold)
        occurrences = tuple(InformationOccurrence(item["info"], item["context"]) for item in information)
        groups, canonical_information = [], []
        for group in sorted(partition, key=min):
            indices = tuple(sorted(group))
            scores = {index: member_mean(index, indices, matrix) for index in indices}
            groups.append([{**copy.deepcopy(information[index]), "avg_similarity": scores[index]} for index in indices])
            representative = canonical_representative(occurrences, scores)
            canonical_information.append({
                **copy.deepcopy(information[representative]),
                "representative_index": representative, "member_indices": list(indices),
                "avg_similarity": scores[representative],
            })
        excluded = {"content", "map", "information", "set_different_information", "canonical_information"}
        return {
            "metadata": {
                "schema_version": 1, "algorithm": "complete_linkage", "threshold": self.threshold,
                "embedding_profile": self.profile.metadata,
                "counts": {
                    "sources": len(dataset), "segments": len(segments),
                    "original_information": sum(len(segment.occurrences) for segment in segments),
                    "local_canonical_information": len(information),
                    "global_canonical_information": len(canonical_information),
                },
            },
            "sources": [
                {
                    "source_index": source_index,
                    "metadata": copy.deepcopy({key: value for key, value in source.items() if key != "segments"}),
                    "segments": [
                        {"segment_index": segment_index,
                         "metadata": copy.deepcopy({key: value for key, value in segment.items() if key not in excluded})}
                        for segment_index, segment in enumerate(source["segments"])
                    ],
                }
                for source_index, source in enumerate(dataset)
            ],
            "information": information,
            "set_different_information": groups,
            "canonical_information": canonical_information,
        }
