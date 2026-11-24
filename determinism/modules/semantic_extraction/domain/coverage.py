"""Contagem determinística das referências validadas de cada ocorrência textual."""

from .contracts import Dataset, ExtractionError, Information, SegmentInput


MIN_TOKEN_COVERAGE = 2


class IncompleteCoverageError(ExtractionError):
    """Meta incompleta; um snapshot validado permite salvar o resultado parcial."""

    def __init__(self, message: str, *, partial_dataset: Dataset | None = None) -> None:
        super().__init__(message)
        self.partial_dataset = partial_dataset


def add_information_counts(counts: dict[int, int], information: list[Information]) -> None:
    """Cada ID distinto de cada informação validada contribui exatamente uma vez."""
    for item in information:
        for token in item["tokens"]:
            counts[token] += 1


def count_information(segment: SegmentInput, information: list[Information]) -> dict[int, int]:
    """Conta somente as informações fornecidas, independentemente de count antigo."""
    counts = {entry.token: 0 for entry in segment.tokens}
    add_information_counts(counts, information)
    return counts


def validate_coverage_feasibility(segment: SegmentInput) -> None:
    if len(segment.tokens) == 1:
        raise IncompleteCoverageError(
            f"{segment.path}: cobertura impossível para um mapa com somente um token; "
            "cada informação exige pelo menos dois IDs distintos presentes no mapa."
        )
