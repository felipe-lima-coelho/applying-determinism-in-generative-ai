"""Rodadas limitadas e checkpoints coerentes, sem conhecimento de arquivos."""

from copy import deepcopy
from dataclasses import dataclass

from ..domain import Dataset, IncompleteRefinementError, RefinementError, pending_description, validate_grouped_dataset
from ..ports import ProgressCallback, SelectedExtractor, SelectedGrouper


@dataclass(frozen=True)
class SingletonRefinementStage:
    extractor: SelectedExtractor
    grouper: SelectedGrouper
    max_rounds: int = 10
    on_progress: ProgressCallback | None = None

    def __post_init__(self) -> None:
        if type(self.max_rounds) is not int or self.max_rounds < 1:
            raise RefinementError("max_rounds deve ser um inteiro positivo.")

    def __call__(self, dataset: Dataset) -> Dataset:
        pending = validate_grouped_dataset(dataset)
        # Todas as dependências oferecem preflight puro; mesmo erros em segmentos
        # já resolvidos precisam ser detectados antes da primeira geração paga.
        self.extractor.validate_dataset(dataset)
        self.grouper.validate_dataset(dataset)
        current = deepcopy(dataset)
        for round_number in range(1, self.max_rounds + 1):
            if not pending:
                return current
            selected = frozenset(segment.key for segment in pending)
            phase = "extração"
            try:
                extracted = self.extractor.extract_segments(current, selected, generation_round=round_number)
                phase = "embeddings/agrupamento"
                grouped = self.grouper.group_segments(extracted, selected)
                next_pending = validate_grouped_dataset(grouped)
            except Exception as error:
                # Um snapshot parcial da extração ainda contém grupos antigos.
                # Ele nunca substitui o último estado integralmente reagrupado.
                raise IncompleteRefinementError(
                    f"Refinamento incompleto na rodada {round_number}, fase {phase}: {error}\n"
                    f"Último estado coerente após {round_number - 1} rodada(s). Pendências: {pending_description(pending)}.",
                    partial_dataset=current, pending=pending, completed_rounds=round_number - 1, phase=phase,
                ) from error
            if self.on_progress is not None:
                # O callback pode persistir e/ou observar sem modificar o estado
                # que será usado na rodada seguinte. Escrita atômica é do adaptador.
                self.on_progress(deepcopy(grouped))
            current, pending = grouped, next_pending
        if pending:
            raise IncompleteRefinementError(
                f"Refinamento incompleto após max_rounds={self.max_rounds}. "
                f"Pendências: {pending_description(pending)}. Retome a partir do último checkpoint coerente.",
                partial_dataset=current, pending=pending, completed_rounds=self.max_rounds, phase="limite de rodadas",
            )
        return current
