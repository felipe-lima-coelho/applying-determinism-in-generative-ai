"""Composição do agrupamento e canonicalização, independente de mapear ou extrair."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..modules.semantic_grouping.adapters import (
    GroupingConfig, OpenAIEmbeddingProvider, PostgresVectorStore, load_grouping_config,
    global_catalog_output_path, read_embedding_api_key, read_postgres_settings,
    save_grouping_artifacts, validate_grouping_paths,
)
from ..modules.semantic_grouping.application import GlobalCatalogBuilder, GroupingStage
from ..modules.semantic_grouping.domain import CompleteLinkageGrouping, Dataset, EmbeddingProfile, validate_information_dataset
from ..modules.semantic_grouping.ports import DatasetRepository, EmbeddingProvider, GroupingStrategy, VectorStore
from ..modules.text_mapping.adapters import JsonDatasetRepository
from ..pipelines import Pipeline


def grouping_pipeline(
    provider: EmbeddingProvider, vector_store: VectorStore, profile: EmbeddingProfile, *,
    strategy: GroupingStrategy | None = None, threshold: float = 0.90,
) -> Pipeline[Dataset]:
    return Pipeline((GroupingStage(provider, vector_store, profile, strategy if strategy is not None else CompleteLinkageGrouping(), threshold),))


@dataclass(frozen=True)
class SemanticGroupingExperiment:
    pipeline: Pipeline[Dataset]
    repository: DatasetRepository = field(default_factory=JsonDatasetRepository)

    def group(self, dataset: Dataset) -> Dataset:
        return self.pipeline.run(dataset)

    def run(self, input_path: str | Path, output_path: str | Path | None = None) -> Dataset:
        source = Path(input_path)
        destination = Path(output_path) if output_path is not None else source
        grouped = self.group(self.repository.load(source))
        self.repository.save(destination, grouped)
        return grouped


def group_dataset(
    dataset: Dataset, *, provider: EmbeddingProvider, vector_store: VectorStore,
    profile: EmbeddingProfile, strategy: GroupingStrategy | None = None,
    threshold: float = 0.90,
) -> Dataset:
    return grouping_pipeline(provider, vector_store, profile, strategy=strategy, threshold=threshold).run(dataset)


@dataclass(frozen=True)
class GroupingResult:
    dataset: Dataset
    global_catalog: dict[str, Any]


def build_global_catalog(
    dataset: Dataset, *, vector_store: VectorStore | None, profile: EmbeddingProfile,
    strategy: GroupingStrategy | None = None, threshold: float = 0.90,
) -> dict[str, Any]:
    return GlobalCatalogBuilder(
        vector_store, profile, strategy if strategy is not None else CompleteLinkageGrouping(), threshold,
    )(dataset)


def group_dataset_with_catalog(
    dataset: Dataset, *, provider: EmbeddingProvider, vector_store: VectorStore,
    profile: EmbeddingProfile, strategy: GroupingStrategy | None = None, threshold: float = 0.90,
) -> GroupingResult:
    grouped = group_dataset(dataset, provider=provider, vector_store=vector_store, profile=profile,
                            strategy=strategy, threshold=threshold)
    catalog = build_global_catalog(grouped, vector_store=vector_store, profile=profile,
                                   strategy=strategy, threshold=threshold)
    return GroupingResult(grouped, catalog)


def run_grouping(
    input_path: str | Path, output_path: str | Path | None = None, *,
    global_output_path: str | Path | None = None,
    config: GroupingConfig | None = None, provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None, strategy: GroupingStrategy | None = None,
    repository: DatasetRepository | None = None, env_file: str | Path | None = ".env",
) -> Dataset:
    """Preserva o retorno list[source] e salva também o catálogo global irmão."""
    return run_grouping_with_catalog(
        input_path, output_path, global_output_path=global_output_path, config=config, provider=provider,
        vector_store=vector_store, strategy=strategy, repository=repository, env_file=env_file,
    ).dataset


def run_grouping_with_catalog(
    input_path: str | Path, output_path: str | Path | None = None, *,
    global_output_path: str | Path | None = None,
    config: GroupingConfig | None = None, provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None, strategy: GroupingStrategy | None = None,
    repository: DatasetRepository | None = None, env_file: str | Path | None = ".env",
) -> GroupingResult:
    selected_config = config if config is not None else load_grouping_config()
    selected_repository = repository if repository is not None else JsonDatasetRepository()
    source = Path(input_path)
    destination = Path(output_path) if output_path is not None else source
    global_destination = Path(global_output_path) if global_output_path is not None else global_catalog_output_path(destination)
    validate_grouping_paths(source, destination, global_destination)
    dataset = selected_repository.load(source)
    segments = validate_information_dataset(dataset)
    selected_store = vector_store
    if not any(segment.occurrences for segment in segments):
        # Ausência de informação não exige credenciais, PostgreSQL nem inferência.
        import copy
        grouped = copy.deepcopy(dataset)
        for segment in segments:
            target = grouped[segment.source_index]["segments"][segment.segment_index]
            target["set_different_information"] = []
            target["canonical_information"] = []
    else:
        selected_provider = provider if provider is not None else OpenAIEmbeddingProvider(
            selected_config, read_embedding_api_key(selected_config, env_file),
        )
        if vector_store is None:
            selected_store = PostgresVectorStore(read_postgres_settings(env_file))
            selected_store.initialize()
        else:
            selected_store = vector_store
        experiment = SemanticGroupingExperiment(
            grouping_pipeline(
                selected_provider, selected_store, selected_config.profile,
                strategy=strategy, threshold=selected_config.threshold,
            ), selected_repository,
        )
        grouped = experiment.group(dataset)
    catalog = build_global_catalog(
        grouped, vector_store=selected_store, profile=selected_config.profile,
        strategy=strategy, threshold=selected_config.threshold,
    )
    try:
        reference = os.path.relpath(destination.resolve(), global_destination.parent.resolve())
    except ValueError:  # Volumes Windows distintos não admitem um caminho relativo.
        reference = str(destination.resolve())
    catalog["metadata"]["dataset_file"] = Path(reference).as_posix()
    save_grouping_artifacts(selected_repository, destination, grouped, global_destination, catalog)
    return GroupingResult(grouped, catalog)
