"""Partições existentes e critério de término; sem inferência ou persistência."""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TypeAlias


Dataset: TypeAlias = list[dict[str, Any]]
SegmentKey: TypeAlias = tuple[int, int]


class RefinementError(ValueError):
    """O refinamento não pôde ser concluído."""


class RefinementInputError(RefinementError):
    """Os grupos existentes não constituem uma partição válida do histórico."""


@dataclass(frozen=True)
class SingletonSegment:
    source_index: int
    segment_index: int
    singleton_groups: int

    @property
    def key(self) -> SegmentKey:
        return self.source_index, self.segment_index

    @property
    def path(self) -> str:
        return f"$[{self.source_index}].segments[{self.segment_index}]"


class IncompleteRefinementError(RefinementError):
    """Somente o último dataset integralmente reagrupado é retomável."""

    def __init__(
        self, message: str, *, partial_dataset: Dataset, pending: tuple[SingletonSegment, ...],
        completed_rounds: int, phase: str,
    ) -> None:
        super().__init__(message)
        self.partial_dataset = deepcopy(partial_dataset)
        self.pending = pending
        self.completed_rounds = completed_rounds
        self.phase = phase


def pending_description(pending: tuple[SingletonSegment, ...]) -> str:
    return "; ".join(f"{segment.path}: {segment.singleton_groups} grupo(s) singleton" for segment in pending)


def validate_grouped_dataset(value: object) -> tuple[SingletonSegment, ...]:
    """Exige a partição exata do multiconjunto (info, context) de CADA segmento.

    Repetições são ocorrências reais. O validador não consulta vetores nem
    reagrupa silenciosamente informações ausentes ou desatualizadas.
    """
    guidance = "Execute 'python -m determinism group <dataset>' antes de refine."

    def invalid(path: str, reason: str) -> None:
        raise RefinementInputError(f"{path}: {reason}. {guidance}")

    if not isinstance(value, list):
        invalid("$", "o dataset deve ser uma lista de fontes")
    pending = []
    for source_index, source in enumerate(value):
        source_path = f"$[{source_index}]"
        if not isinstance(source, dict) or not isinstance(source.get("segments"), list):
            invalid(source_path, "segments deve ser uma lista em um objeto de fonte")
        for segment_index, segment in enumerate(source["segments"]):
            path = f"{source_path}.segments[{segment_index}]"
            if not isinstance(segment, dict) or not isinstance(segment.get("information"), list):
                invalid(path, "information deve ser uma lista existente")
            expected = Counter()
            for index, item in enumerate(segment["information"]):
                if (not isinstance(item, dict) or not isinstance(item.get("info"), str)
                        or not isinstance(item.get("context"), str)):
                    invalid(f"{path}.information[{index}]", "info e context devem ser strings")
                expected[item["info"], item["context"]] += 1
            groups = segment.get("set_different_information")
            if not isinstance(groups, list):
                invalid(path, "set_different_information deve existir como lista para todos os segmentos")
            actual = Counter()
            singletons = 0
            for group_index, group in enumerate(groups):
                group_path = f"{path}.set_different_information[{group_index}]"
                if not isinstance(group, list) or not group:
                    invalid(group_path, "cada grupo deve ser uma lista não vazia")
                singletons += len(group) == 1
                for item in group:
                    if (not isinstance(item, dict) or not isinstance(item.get("info"), str)
                            or not isinstance(item.get("context"), str)):
                        invalid(group_path, "cada membro deve conter info e context como strings")
                    actual[item["info"], item["context"]] += 1
            if actual != expected:
                invalid(path, "grupos desatualizados: devem particionar exatamente as ocorrências de information, incluindo duplicatas e contextos")
            if singletons:
                pending.append(SingletonSegment(source_index, segment_index, singletons))
    return tuple(pending)
