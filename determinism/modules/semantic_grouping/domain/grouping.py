"""Aglomeração viável e realocação até estabilidade de melhor média viável."""

import math
from dataclasses import dataclass

from .contracts import GroupingError, Partition, SimilarityMatrix, member_mean, validate_partition


class GroupingConvergenceError(GroupingError):
    """A realocação repetiu um estado ou excedeu o limite explícito de movimentos."""


def canonical_partition(groups) -> Partition:
    return tuple(sorted((tuple(sorted(group)) for group in groups if group), key=lambda group: group))


def viable(group: tuple[int, ...], matrix: SimilarityMatrix, threshold: float) -> bool:
    return all(member_mean(index, group, matrix) >= threshold for index in group)


def feasible_destination(
    index: int, source: tuple[int, ...], destination: tuple[int, ...],
    matrix: SimilarityMatrix, threshold: float,
) -> bool:
    remainder = tuple(member for member in source if member != index)
    return viable(remainder, matrix, threshold) and viable(tuple(sorted((*destination, index))), matrix, threshold)


def agglomerate(matrix: SimilarityMatrix, threshold: float) -> Partition:
    groups = canonical_partition((index,) for index in range(len(matrix)))
    while True:
        best = None
        for first in range(len(groups)):
            for second in range(first + 1, len(groups)):
                union = tuple(sorted((*groups[first], *groups[second])))
                if not viable(union, matrix, threshold):
                    continue
                score = math.fsum(matrix[a][b] for a in groups[first] for b in groups[second]) / (len(groups[first]) * len(groups[second]))
                candidate = (-score, groups[first], groups[second], first, second, union)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            return groups
        *_, first, second, union = best
        groups = canonical_partition([group for position, group in enumerate(groups) if position not in (first, second)] + [union])


@dataclass(frozen=True)
class BestMeanGrouping:
    max_moves: int = 10000

    def __post_init__(self) -> None:
        if type(self.max_moves) is not int or self.max_moves < 1:
            raise GroupingError("max_moves deve ser um inteiro positivo.")

    def partition(self, matrix: SimilarityMatrix, threshold: float) -> Partition:
        groups = agglomerate(matrix, threshold)
        visited = {groups}
        moves = 0
        while True:
            changed = False
            for index in range(len(matrix)):
                source = next(group for group in groups if index in group)
                current = member_mean(index, source, matrix) if len(source) > 1 else 0.0
                candidates = []
                for destination in groups:
                    if destination == source or not feasible_destination(index, source, destination, matrix, threshold):
                        continue
                    score = math.fsum(matrix[index][other] for other in destination) / len(destination)
                    if score > current:
                        candidates.append((-score, destination))
                if not candidates:
                    continue
                if moves >= self.max_moves:
                    raise GroupingConvergenceError("Agrupamento excedeu max_moves; nenhum JSON será salvo.")
                _, destination = min(candidates)
                remainder = tuple(member for member in source if member != index)
                groups = canonical_partition(
                    [group for group in groups if group not in (source, destination)]
                    + [remainder, tuple((*destination, index))],
                )
                moves += 1
                if groups in visited:
                    raise GroupingConvergenceError("Agrupamento repetiu uma partição durante realocação; nenhum JSON será salvo.")
                visited.add(groups)
                changed = True
            if not changed:
                return validate_partition(groups, matrix, threshold)
