"""Gerações adaptativas limitadas; snapshots válidos quando a cobertura esgota."""

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass, replace

from ..domain import (
    MIN_TOKEN_COVERAGE, CoverageFeedback, Dataset, IncompleteCoverageError, InferenceResult, Information,
    SegmentInput, add_information_counts, count_information, validate_coverage_feasibility,
    validate_information, validate_mapped_dataset,
)
from ..ports import InferenceError, InferenceProvider


@dataclass
class _SegmentProgress:
    segment: SegmentInput
    information: list[Information]
    counts: dict[int, int]
    generations: int = 0

    @property
    def covered(self) -> bool:
        return all(count >= MIN_TOKEN_COVERAGE for count in self.counts.values())

    @property
    def pending(self) -> tuple[tuple[int, int], ...]:
        return tuple((token, count) for token, count in self.counts.items() if count < MIN_TOKEN_COVERAGE)


@dataclass(frozen=True)
class ExtractionStage:
    provider: InferenceProvider
    max_concurrency: int = 4
    max_generations_per_segment: int = 10

    def __post_init__(self) -> None:
        if type(self.max_concurrency) is not int or not 1 <= self.max_concurrency <= 256:
            raise ValueError("max_concurrency deve ser um inteiro entre 1 e 256.")
        if type(self.max_generations_per_segment) is not int or self.max_generations_per_segment < 1:
            raise ValueError("max_generations_per_segment deve ser um inteiro positivo.")

    def __call__(self, dataset: Dataset) -> Dataset:
        return self._extract(dataset)

    @staticmethod
    def validate_dataset(dataset: Dataset) -> None:
        """Preflight puro de todos os mapas e de todo o histórico, sem inferência."""
        ExtractionStage._prepare(dataset)

    def extract_segments(
        self, dataset: Dataset, selected_segments: frozenset[tuple[int, int]], *, generation_round: int = 1,
    ) -> Dataset:
        """Acrescenta ao menos uma geração nos índices originais selecionados.

        A cobertura continua obrigatória. Cada rodada reserva uma janela de
        ordinais do tamanho do limite de gerações, sem reiniciar ordinais entre
        rodadas que precisaram de várias inferências para alcançar cobertura.
        """
        if type(generation_round) is not int or generation_round < 1:
            raise ValueError("generation_round deve ser um inteiro positivo.")
        return self._extract(dataset, selected_segments, force_generation=True, generation_round=generation_round)

    @staticmethod
    def _prepare(dataset: Dataset) -> list[_SegmentProgress]:
        segments = validate_mapped_dataset(dataset)
        progress = []
        # Todo o histórico também precisa ser validado antes da primeira chamada.
        for segment in segments:
            stored = dataset[segment.source_index]["segments"][segment.segment_index]
            information = validate_information({"informacoes": stored.get("information", [])}, segment)
            progress.append(_SegmentProgress(segment, information, count_information(segment, information)))
        return progress

    def _extract(
        self, dataset: Dataset, selected_segments: frozenset[tuple[int, int]] | None = None, *,
        force_generation: bool = False, generation_round: int = 1,
    ) -> Dataset:
        progress = self._prepare(dataset)
        known = {(state.segment.source_index, state.segment.segment_index) for state in progress}
        if selected_segments is not None and (
            not isinstance(selected_segments, frozenset)
            or any(not isinstance(key, tuple) or len(key) != 2 or any(type(index) is not int for index in key)
                   for key in selected_segments)
            or not selected_segments.issubset(known)
        ):
            raise ValueError("selected_segments deve conter somente pares válidos de índices originais.")
        active = [state for state in progress if selected_segments is None
                  or (state.segment.source_index, state.segment.segment_index) in selected_segments]
        for state in active:
            validate_coverage_feasibility(state.segment)
        extracted = deepcopy(dataset)
        self._collect(active, force_generation, (generation_round - 1) * self.max_generations_per_segment)
        # A coleta valida também todas as chamadas em voo após observar o limite.
        # Outros erros saem antes deste ponto e não oferecem snapshot para salvar.
        for state in active:
            segment = state.segment
            target = extracted[segment.source_index]["segments"][segment.segment_index]
            target["information"] = state.information
            for token, count in state.counts.items():
                target["map"][str(token)]["count"] = count
        incomplete = [state for state in active if not state.covered]
        if incomplete:
            details = []
            for state in incomplete:
                missing = ", ".join(f"{token}={count}" for token, count in state.pending)
                details.append(
                    f"{state.segment.path}: cobertura incompleta após {state.generations} gerações "
                    f"(max_generations_per_segment={self.max_generations_per_segment}); "
                    f"tokens com count < {MIN_TOKEN_COVERAGE} (ID=count): {missing}."
                )
            raise IncompleteCoverageError("\n".join(details), partial_dataset=extracted)
        return extracted

    def _collect(
        self, progress: list[_SegmentProgress], force_generation: bool = False, generation_offset: int = 0,
    ) -> None:
        remaining = deque(index for index, state in enumerate(progress) if force_generation or not state.covered)
        if not remaining:
            return
        pending: dict[Future[InferenceResult], int] = {}
        exhausted = False
        with ThreadPoolExecutor(max_workers=self.max_concurrency, thread_name_prefix="semantic-extraction") as executor:
            def fill_available_slots() -> None:
                while not exhausted and remaining and len(pending) < self.max_concurrency:
                    index = remaining.popleft()
                    state = progress[index]
                    state.generations += 1
                    request = replace(state.segment, coverage=CoverageFeedback(generation_offset + state.generations, state.pending))
                    pending[executor.submit(self.provider.extract, request)] = index

            try:
                fill_available_slots()
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in sorted(done, key=pending.__getitem__):
                        index = pending.pop(future)
                        state = progress[index]
                        segment = state.segment
                        try:
                            response = future.result()
                        except InferenceError as error:
                            raise InferenceError(f"{segment.path}: {error}") from None
                        if not isinstance(response, InferenceResult):
                            raise InferenceError(f"{segment.path}: o provedor não retornou InferenceResult.")
                        information = validate_information(response.payload, segment)
                        state.information.extend(information)
                        add_information_counts(state.counts, information)
                        if not state.covered:
                            if state.generations >= self.max_generations_per_segment:
                                # Pare submissões, mas retenha respostas já pagas/em voo
                                # somente depois de validar todas elas.
                                exhausted = True
                            else:
                                remaining.append(index)
                    fill_available_slots()
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
