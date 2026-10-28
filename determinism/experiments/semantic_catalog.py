"""Composição do catálogo e da simulação, com artefatos independentes do dataset."""

from dataclasses import asdict
from pathlib import Path

from ..modules.semantic_catalog.adapters import (
    CatalogConfig, OpenAICompatibleCatalogProvider, contract_fingerprint, load_catalog_config, read_api_key,
)
from ..modules.semantic_catalog.adapters.json_artifacts import JsonResponseCache, read_json, write_json
from ..modules.semantic_catalog.application import CatalogStage, RequirementCatalogStage
from ..modules.semantic_catalog.domain import CatalogError, simulate_removal
from ..modules.text_mapping.adapters import JsonDatasetRepository


class _ConfiguredProvider:
    """Só exige credencial quando há uma requisição que não está no cache."""

    def __init__(self, config, env_file):
        self.config, self.env_file = config, env_file

    def _provider(self):
        return OpenAICompatibleCatalogProvider(self.config, read_api_key(self.config, self.env_file))

    def extract(self, segment):
        return self._provider().extract(segment)

    def compare(self, pairs):
        return self._provider().compare(pairs)

    def assess(self, segment, facts):
        return self._provider().assess(segment, facts)

    def requirements(self, candidate):
        return self._provider().requirements(candidate)

    def assess_requirements(self, segment, facts):
        return self._provider().assess_requirements(segment, facts)


def _destination(path, *protected):
    destination = Path(path).resolve()
    for source in protected:
        if source is None:
            continue
        original = Path(source).resolve()
        if original == destination or (original.exists() and destination.exists() and original.samefile(destination)):
            raise CatalogError("O destino deve ser diferente dos arquivos de entrada; catálogo e simulação são artefatos separados.")
    if not destination.parent.is_dir():
        raise CatalogError("O diretório de destino deve existir.")
    if destination.exists() and not destination.is_file():
        raise CatalogError("O destino deve ser um arquivo.")
    return destination


def build_catalog(dataset, *, provider, max_concurrency=4, comparison_batch_size=24,
                  assessment_batch_size=1, cache=None, provenance=None, method="requirements"):
    """API em memória. O provedor e cache são substituíveis; o dataset não é modificado."""
    if method not in ("requirements", "legacy"):
        raise CatalogError("method deve ser requirements ou legacy.")
    stage = RequirementCatalogStage if method == "requirements" else CatalogStage
    return stage(provider, max_concurrency, comparison_batch_size, assessment_batch_size,
                 cache, {} if provenance is None else provenance)(dataset)


def run_catalog(input_path, output_path, *, config: CatalogConfig | None = None, provider=None,
                env_file=".env", cache_dir=None, provider_identity=None):
    destination = _destination(output_path, input_path)
    dataset = JsonDatasetRepository().load(Path(input_path))
    selected = config if config is not None else load_catalog_config()
    provenance = {"configuration": asdict(selected), "contract_fingerprint": contract_fingerprint()}
    if provider is not None:
        if cache_dir is not None and not provider_identity:
            raise CatalogError("Cache de provedor injetado exige provider_identity versionado.")
        provenance["injected_provider"] = provider_identity or f"{type(provider).__module__}.{type(provider).__qualname__}"
    selected_provider = provider if provider is not None else _ConfiguredProvider(selected, env_file)
    catalog = build_catalog(
        dataset, provider=selected_provider, max_concurrency=selected.max_concurrency,
        comparison_batch_size=selected.comparison_batch_size, assessment_batch_size=selected.assessment_batch_size,
        cache=JsonResponseCache(Path(cache_dir)) if cache_dir is not None else None, provenance=provenance,
        method=selected.method,
    )
    write_json(destination, catalog)
    return catalog


def run_simulation(catalog_path, output_path, *, removed_segment_ids=(), removed_source_ids=(), dataset_path=None):
    destination = _destination(output_path, catalog_path, dataset_path)
    catalog = read_json(catalog_path)
    dataset = JsonDatasetRepository().load(Path(dataset_path)) if dataset_path is not None else None
    report = simulate_removal(catalog, removed_segment_ids, removed_source_ids, dataset=dataset)
    write_json(destination, report)
    return report
