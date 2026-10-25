"""Extração concorrente limitada, validada e aplicada somente após sucesso total."""

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from copy import deepcopy
from dataclasses import dataclass

from ..domain import Dataset, InferenceResult, Information, SegmentInput, validate_information, validate_mapped_dataset
from ..ports import InferenceError, InferenceProvider


@dataclass(frozen=True)
class ExtractionStage:
    provider: InferenceProvider
    max_concurrency: int = 4

    def __post_init__(self) -> None:
        if type(self.max_concurrency) is not int or not 1 <= self.max_concurrency <= 256:
            raise ValueError("max_concurrency deve ser um inteiro entre 1 e 256.")

    def __call__(self, dataset: Dataset) -> Dataset:
        segments = validate_mapped_dataset(dataset)
        extracted = deepcopy(dataset)
        results: list[list[Information] | None] = [None] * len(segments)
        if segments:
            self._collect(segments, results)
        # Nenhum resultado é aplicado antes de TODAS as inferências e validações.
        for segment, information in zip(segments, results):
            extracted[segment.source_index]["segments"][segment.segment_index]["information"] = information
        return extracted

    def _collect(self, segments: tuple[SegmentInput, ...], results: list[list[Information] | None]) -> None:
        remaining = iter(enumerate(segments))
        pending: dict[Future[InferenceResult], tuple[int, SegmentInput]] = {}
        with ThreadPoolExecutor(max_workers=self.max_concurrency, thread_name_prefix="semantic-extraction") as executor:
            def submit_next() -> None:
                next_item = next(remaining, None)
                if next_item is not None:
                    index, segment = next_item
                    pending[executor.submit(self.provider.extract, segment)] = (index, segment)

            for _ in range(min(self.max_concurrency, len(segments))):
                submit_next()
            try:
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        index, segment = pending.pop(future)
                        try:
                            response = future.result()
                        except InferenceError as error:
                            raise InferenceError(f"{segment.path}: {error}") from None
                        if not isinstance(response, InferenceResult):
                            raise InferenceError(f"{segment.path}: o provedor não retornou InferenceResult.")
                        results[index] = validate_information(response.payload, segment)
                    for _ in done:
                        submit_next()
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
