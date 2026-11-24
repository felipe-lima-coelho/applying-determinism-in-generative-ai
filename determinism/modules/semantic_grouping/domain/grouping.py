"""Agrupamento hierárquico aglomerativo por complete linkage (máximo do mínimo)."""

from .contracts import Partition, SimilarityMatrix, validate_partition


class CompleteLinkageGrouping:
    def partition(self, matrix: SimilarityMatrix, threshold: float) -> Partition:
        """Reutiliza a matriz de pares; cada fusão reduz o número de grupos em um."""
        groups = tuple((index,) for index in range(len(matrix)))
        while True:
            best = None
            for first in range(len(groups)):
                for second in range(first + 1, len(groups)):
                    score = min(matrix[a][b] for a in groups[first] for b in groups[second])
                    if score < threshold:
                        continue
                    # Tuplas canônicas de índices originais resolvem empates.
                    candidate = (-score, groups[first], groups[second], first, second)
                    if best is None or candidate < best:
                        best = candidate
            if best is None:
                return validate_partition(groups, matrix, threshold)
            _, _, _, first, second = best
            union = tuple(sorted((*groups[first], *groups[second])))
            groups = tuple(sorted(
                [group for position, group in enumerate(groups) if position not in (first, second)] + [union],
            ))
