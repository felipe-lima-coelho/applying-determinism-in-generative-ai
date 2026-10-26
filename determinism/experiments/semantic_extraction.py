"""Composição explícita do experimento de extração; não inclui mapeamento."""

from dataclasses import dataclass, field
from pathlib import Path

from ..modules.semantic_extraction.adapters import (
    ExtractionConfig, OpenAICompatibleProvider, load_extraction_config, read_api_key,
)
from ..modules.semantic_extraction.application import ExtractionStage
from ..modules.semantic_extraction.domain import Dataset, IncompleteCoverageError
from ..modules.semantic_extraction.ports import DatasetRepository, InferenceProvider
from ..modules.text_mapping.adapters import JsonDatasetRepository
from ..pipelines import Pipeline


def extraction_pipeline(
    provider: InferenceProvider, *, max_concurrency: int = 4, max_generations_per_segment: int = 10,
) -> Pipeline[Dataset]:
    return Pipeline((ExtractionStage(provider, max_concurrency, max_generations_per_segment),))


@dataclass(frozen=True)
class SemanticExtractionExperiment:
    pipeline: Pipeline[Dataset]
    repository: DatasetRepository = field(default_factory=JsonDatasetRepository)

    def extract(self, dataset: Dataset) -> Dataset:
        return self.pipeline.run(dataset)

    def run(self, input_path: str | Path, output_path: str | Path | None = None) -> Dataset:
        source = Path(input_path)
        destination = Path(output_path) if output_path is not None else source
        try:
            extracted = self.extract(self.repository.load(source))
        except IncompleteCoverageError as error:
            if error.partial_dataset is None:
                raise
            self.repository.save(destination, error.partial_dataset)
            raise IncompleteCoverageError(
                f"{error}\nProgresso parcial salvo em {destination}. A cobertura continua incompleta. "
                "Para retomar, use esse arquivo salvo como entrada; as informações serão recontadas sem duplicar o histórico.",
                partial_dataset=error.partial_dataset,
            ) from None
        self.repository.save(destination, extracted)
        return extracted


def extract_dataset(
    dataset: Dataset, *, provider: InferenceProvider, max_concurrency: int = 4,
    max_generations_per_segment: int = 10,
) -> Dataset:
    return extraction_pipeline(
        provider, max_concurrency=max_concurrency, max_generations_per_segment=max_generations_per_segment,
    ).run(dataset)


def run_extraction(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    config: ExtractionConfig | None = None,
    provider: InferenceProvider | None = None,
    repository: DatasetRepository | None = None,
    env_file: str | Path | None = ".env",
) -> Dataset:
    selected_config = config if config is not None else load_extraction_config()
    selected_provider = provider if provider is not None else OpenAICompatibleProvider(
        selected_config, read_api_key(selected_config, env_file),
    )
    experiment = SemanticExtractionExperiment(
        extraction_pipeline(
            selected_provider, max_concurrency=selected_config.max_concurrency,
            max_generations_per_segment=selected_config.max_generations_per_segment,
        ),
        repository if repository is not None else JsonDatasetRepository(),
    )
    return experiment.run(input_path, output_path)
