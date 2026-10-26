"""Composição explícita: reaproveita extração e agrupamento, sem novos algoritmos."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..modules.semantic_extraction.adapters import (
    ExtractionConfig, OpenAICompatibleProvider, load_extraction_config, read_api_key,
)
from ..modules.semantic_extraction.application import ExtractionStage
from ..modules.semantic_extraction.ports import InferenceProvider
from ..modules.semantic_grouping.adapters import (
    GroupingConfig, OpenAIEmbeddingProvider, PostgresVectorStore, load_grouping_config,
    read_embedding_api_key, read_postgres_settings,
)
from ..modules.semantic_grouping.application import GroupingStage
from ..modules.semantic_grouping.domain import CompleteLinkageGrouping
from ..modules.semantic_grouping.ports import EmbeddingProvider, GroupingStrategy, VectorStore
from ..modules.singleton_refinement.adapters import RefinementConfig, load_refinement_config
from ..modules.singleton_refinement.application import SingletonRefinementStage
from ..modules.singleton_refinement.domain import Dataset, IncompleteRefinementError, validate_grouped_dataset
from ..modules.singleton_refinement.ports import (
    DatasetRepository, ProgressCallback, SelectedExtractor, SelectedGrouper,
)
from ..modules.text_mapping.adapters import JsonDatasetRepository
from ..pipelines import Pipeline


def refinement_pipeline(
    extractor: SelectedExtractor, grouper: SelectedGrouper, *, max_rounds: int = 10,
    on_progress: ProgressCallback | None = None,
) -> Pipeline[Dataset]:
    return Pipeline((SingletonRefinementStage(extractor, grouper, max_rounds, on_progress),))


@dataclass(frozen=True)
class SingletonRefinementExperiment:
    pipeline: Pipeline[Dataset]
    repository: DatasetRepository = field(default_factory=JsonDatasetRepository)

    def refine(self, dataset: Dataset) -> Dataset:
        return self.pipeline.run(dataset)

    def run(self, input_path: str | Path, output_path: str | Path | None = None) -> Dataset:
        source = Path(input_path)
        destination = Path(output_path) if output_path is not None else source
        return self.run_dataset(self.repository.load(source), destination)

    def run_dataset(self, dataset: Dataset, output_path: str | Path) -> Dataset:
        """Executa um snapshot já carregado, com destino explícito para checkpoints."""
        destination = Path(output_path)
        last_saved: Dataset | None = None

        def checkpoint(current: Dataset) -> None:
            nonlocal last_saved
            self.repository.save(destination, current)
            last_saved = deepcopy(current)

        stages = []
        for stage in self.pipeline.stages:
            if isinstance(stage, SingletonRefinementStage):
                observer = stage.on_progress

                def save_and_observe(current: Dataset, observer=observer) -> None:
                    checkpoint(current)
                    if observer is not None:
                        observer(current)

                stages.append(replace(stage, on_progress=save_and_observe))
            else:
                stages.append(stage)
        try:
            result = Pipeline(tuple(stages)).run(dataset)
        except IncompleteRefinementError as error:
            if last_saved is None:
                raise
            raise IncompleteRefinementError(
                f"{error}\nCheckpoint coerente salvo em {destination}. "
                "Para retomar, use esse arquivo como entrada; o histórico será recontado, sem copiá-lo novamente.",
                partial_dataset=error.partial_dataset, pending=error.pending,
                completed_rounds=error.completed_rounds, phase=error.phase,
            ) from error
        if last_saved != result:
            self.repository.save(destination, result)
        return result


def refine_dataset(
    dataset: Dataset, *, extractor: SelectedExtractor, grouper: SelectedGrouper,
    max_rounds: int = 10, on_progress: ProgressCallback | None = None,
) -> Dataset:
    return refinement_pipeline(extractor, grouper, max_rounds=max_rounds, on_progress=on_progress).run(dataset)


def run_refinement(
    input_path: str | Path, output_path: str | Path | None = None, *,
    config: RefinementConfig | None = None, extraction_config: ExtractionConfig | None = None,
    grouping_config: GroupingConfig | None = None, provider: InferenceProvider | None = None,
    embedding_provider: EmbeddingProvider | None = None, vector_store: VectorStore | None = None,
    strategy: GroupingStrategy | None = None, repository: DatasetRepository | None = None,
    env_file: str | Path | None = ".env",
) -> Dataset:
    selected_config = config if config is not None else load_refinement_config()
    extraction = extraction_config if extraction_config is not None else load_extraction_config()
    grouping = grouping_config if grouping_config is not None else load_grouping_config()
    selected_repository = repository if repository is not None else JsonDatasetRepository()
    source = Path(input_path)
    destination = Path(output_path) if output_path is not None else source
    dataset = selected_repository.load(source)
    # A partição de TODOS os segmentos e os contratos de ambos os módulos são
    # verificados antes de ler credenciais ou inicializar o banco.
    pending = validate_grouped_dataset(dataset)
    ExtractionStage.validate_dataset(dataset)
    GroupingStage.validate_dataset(dataset)
    if not pending:
        result = deepcopy(dataset)
        selected_repository.save(destination, result)
        return result
    selected_provider = provider if provider is not None else OpenAICompatibleProvider(
        extraction, read_api_key(extraction, env_file),
    )
    selected_embeddings = embedding_provider if embedding_provider is not None else OpenAIEmbeddingProvider(
        grouping, read_embedding_api_key(grouping, env_file),
    )
    if vector_store is None:
        selected_store = PostgresVectorStore(read_postgres_settings(env_file))
        selected_store.initialize()
    else:
        selected_store = vector_store
    experiment = SingletonRefinementExperiment(
        refinement_pipeline(
            ExtractionStage(selected_provider, extraction.max_concurrency, extraction.max_generations_per_segment),
            GroupingStage(selected_embeddings, selected_store, grouping.profile,
                          strategy if strategy is not None else CompleteLinkageGrouping(), grouping.threshold),
            max_rounds=selected_config.max_rounds,
        ), selected_repository,
    )
    return experiment.run_dataset(dataset, destination)
