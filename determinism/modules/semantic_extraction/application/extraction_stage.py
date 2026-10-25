"""Gerações concorrentes limitadas até a cobertura, aplicadas após sucesso total."""

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass

from ..domain import (
    MIN_TOKEN_COVERAGE, Dataset, IncompleteCoverageError, InferenceResult, Information,
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
        segments = validate_mapped_dataset(dataset)
        progress = []
        # Todo o histórico também precisa ser validado antes da primeira chamada.
        for segment in segments:
            stored = dataset[segment.source_index]["segments"][segment.segment_index]
            information = validate_information({"informacoes": stored.get("information", [])}, segment)
            progress.append(_SegmentProgress(segment, information, count_information(segment, information)))
        for segment in segments:
            validate_coverage_feasibility(segment)
        extracted = deepcopy(dataset)
        self._collect(progress)
        # Nenhum resultado é aplicado antes de TODAS as inferências e validações.
        for state in progress:
            segment = state.segment
            target = extracted[segment.source_index]["segments"][segment.segment_index]
            target["information"] = state.information
            for token, count in state.counts.items():
                target["map"][str(token)]["count"] = count
        return extracted

    def _collect(self, progress: list[_SegmentProgress]) -> None:
        remaining = deque(index for index, state in enumerate(progress) if not state.covered)
        if not remaining:
            return
        pending: dict[Future[InferenceResult], int] = {}
        with ThreadPoolExecutor(max_workers=self.max_concurrency, thread_name_prefix="semantic-extraction") as executor:
            def fill_available_slots() -> None:
                while remaining and len(pending) < self.max_concurrency:
                    index = remaining.popleft()
                    state = progress[index]
                    state.generations += 1
                    pending[executor.submit(self.provider.extract, state.segment)] = index

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
                                missing = ", ".join(
                                    f"{token}={count}" for token, count in state.counts.items()
                                    if count < MIN_TOKEN_COVERAGE
                                )
                                raise IncompleteCoverageError(
                                    f"{segment.path}: cobertura incompleta após {state.generations} gerações "
                                    f"(max_generations_per_segment={self.max_generations_per_segment}); "
                                    f"tokens com count < {MIN_TOKEN_COVERAGE} (ID=count): {missing}. "
                                    "Nenhum resultado será salvo."
                                )
                            remaining.append(index)
                    fill_available_slots()
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
