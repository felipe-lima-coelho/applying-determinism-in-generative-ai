"""Composição explícita do experimento de extração; não inclui mapeamento."""

from dataclasses import dataclass, field
from pathlib import Path

from ..modules.semantic_extraction.adapters import (
    ExtractionConfig, OpenAICompatibleProvider, load_extraction_config, read_api_key,
)
from ..modules.semantic_extraction.application import ExtractionStage
from ..modules.semantic_extraction.domain import Dataset
from ..modules.semantic_extraction.ports import DatasetRepository, InferenceProvider
from ..modules.text_mapping.adapters import JsonDatasetRepository
from ..pipelines import Pipeline


def extraction_pipeline(provider: InferenceProvider, *, max_concurrency: int = 4) -> Pipeline[Dataset]:
    return Pipeline((ExtractionStage(provider, max_concurrency),))


@dataclass(frozen=True)
class SemanticExtractionExperiment:
    pipeline: Pipeline[Dataset]
    repository: DatasetRepository = field(default_factory=JsonDatasetRepository)

    def extract(self, dataset: Dataset) -> Dataset:
        return self.pipeline.run(dataset)

    def run(self, input_path: str | Path, output_path: str | Path | None = None) -> Dataset:
        source = Path(input_path)
        destination = Path(output_path) if output_path is not None else source
        extracted = self.extract(self.repository.load(source))
        self.repository.save(destination, extracted)
        return extracted


def extract_dataset(dataset: Dataset, *, provider: InferenceProvider, max_concurrency: int = 4) -> Dataset:
    return extraction_pipeline(provider, max_concurrency=max_concurrency).run(dataset)


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
        extraction_pipeline(selected_provider, max_concurrency=selected_config.max_concurrency),
        repository if repository is not None else JsonDatasetRepository(),
    )
    return experiment.run(input_path, output_path)
