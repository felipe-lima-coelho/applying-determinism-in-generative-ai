"""Definição do experimento inicial e de suas dependências substituíveis."""

from dataclasses import dataclass, field
from pathlib import Path

from ..modules.text_mapping.adapters import JsonDatasetRepository
from ..modules.text_mapping.application import MappingStage
from ..modules.text_mapping.domain import Dataset
from ..modules.text_mapping.domain.tokenization import UnicodeWordTokenizer
from ..modules.text_mapping.ports import DatasetRepository, Tokenizer
from ..pipelines import Pipeline


def mapping_pipeline(*, tokenizer: Tokenizer | None = None) -> Pipeline[Dataset]:
    selected = tokenizer if tokenizer is not None else UnicodeWordTokenizer()
    return Pipeline(stages=(MappingStage(selected),))


@dataclass(frozen=True)
class WordMappingExperiment:
    """Configuração executável: pipeline e repositório escolhidos pelo experimento."""

    pipeline: Pipeline[Dataset] = field(default_factory=mapping_pipeline)
    repository: DatasetRepository = field(default_factory=JsonDatasetRepository)

    def map(self, dataset: Dataset) -> Dataset:
        return self.pipeline.run(dataset)

    def run(self, input_path: str | Path, output_path: str | Path | None = None) -> Dataset:
        source = Path(input_path)
        destination = Path(output_path) if output_path is not None else source
        mapped = self.map(self.repository.load(source))
        self.repository.save(destination, mapped)
        return mapped


def map_dataset(dataset: Dataset, *, tokenizer: Tokenizer | None = None) -> Dataset:
    """Mapeia em memória; mantém a entrada intacta e retorna o resultado reutilizável."""
    return mapping_pipeline(tokenizer=tokenizer).run(dataset)


def run_mapping(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    repository: DatasetRepository | None = None,
    tokenizer: Tokenizer | None = None,
) -> Dataset:
    """Executa explicitamente o experimento e salva na entrada por padrão."""
    selected = repository if repository is not None else JsonDatasetRepository()
    experiment = WordMappingExperiment(mapping_pipeline(tokenizer=tokenizer), selected)
    return experiment.run(input_path, output_path)
