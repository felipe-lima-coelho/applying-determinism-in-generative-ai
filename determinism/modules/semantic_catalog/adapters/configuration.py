"""Parâmetros explícitos do catálogo; credenciais não entram nos artefatos."""

import tomllib
import math
from dataclasses import dataclass, fields
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

from ...semantic_extraction.adapters.configuration import (
    ConfigurationError, ExtractionConfig, read_api_key,
)


@dataclass(frozen=True)
class CatalogConfig:
    provider: str = "openai"
    model: str = "gpt-6-luna"
    base_url: str = "https://api.openai.com/v1"
    max_concurrency: int = 4
    temperature: float = 0.0
    seed: int = 42
    reasoning_effort: str = "none"
    timeout_seconds: float = 120.0
    max_completion_tokens: int = 8192
    api_key_env: str = "OPENAI_API_KEY"
    comparison_batch_size: int = 24
    assessment_batch_size: int = 16
    cache_revision: str = "1"
    similarity_threshold: float = 0.85
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_base_url: str = "https://api.openai.com/v1"
    embedding_dimensions: int | None = None
    embedding_batch_size: int = 32
    embedding_timeout_seconds: float = 60.0
    embedding_api_key_env: str = "OPENAI_API_KEY"
    embedding_cache_revision: str = "1"

    def __post_init__(self) -> None:
        # Compartilha a política já usada pelos adaptadores de Chat Completions,
        # sem herdar parâmetros específicos da cobertura lexical da extração.
        common = {item.name for item in fields(ExtractionConfig)} - {"max_generations_per_segment"}
        ExtractionConfig(**{name: getattr(self, name) for name in common})
        if self.provider == "openai" and urlsplit(self.base_url).port not in (None, 443):
            raise ConfigurationError("provider=openai exige a porta HTTPS padrão; selecione openai-compatible para outro servidor.")
        for name in ("comparison_batch_size", "assessment_batch_size"):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 256:
                raise ConfigurationError(f"{name} deve ser um inteiro entre 1 e 256.")
        if not isinstance(self.cache_revision, str) or not self.cache_revision.strip():
            raise ConfigurationError("cache_revision deve ser uma string não vazia.")
        for value in (self.provider, self.model, self.base_url, self.cache_revision):
            try:
                value.encode("utf-8")
            except UnicodeError:
                raise ConfigurationError("Identificadores da configuração devem usar UTF-8 válido.") from None
            if any(ord(character) < 32 for character in value):
                raise ConfigurationError("Identificadores da configuração não podem conter caracteres de controle.")
        if (type(self.similarity_threshold) not in (int, float) or not math.isfinite(self.similarity_threshold)
                or not -1 <= self.similarity_threshold <= 1):
            raise ConfigurationError("similarity_threshold deve ser finito e estar entre -1 e 1.")
        self.embedding_config  # Valida o perfil completo antes de qualquer inferência.

    @property
    def embedding_config(self):
        from ...semantic_grouping.adapters.configuration import GroupingConfig, GroupingConfigurationError
        try:
            return GroupingConfig(
                provider=self.embedding_provider, model=self.embedding_model, base_url=self.embedding_base_url,
                threshold=self.similarity_threshold, dimensions=self.embedding_dimensions,
                batch_size=self.embedding_batch_size, timeout_seconds=self.embedding_timeout_seconds,
                api_key_env=self.embedding_api_key_env, cache_revision=self.embedding_cache_revision,
            )
        except GroupingConfigurationError as error:
            raise ConfigurationError(f"Configuração de embeddings do catálogo: {error}") from None


def load_catalog_config(path: str | Path | None = None) -> CatalogConfig:
    try:
        resource = files("determinism.experiments").joinpath("configs/semantic_catalog.toml")
        text = resource.read_text(encoding="utf-8") if path is None else Path(path).read_text(encoding="utf-8-sig")
        values = tomllib.loads(text)
    except (ValueError, UnicodeError):
        raise ConfigurationError("Não foi possível ler a configuração TOML do catálogo; verifique sintaxe e UTF-8.") from None
    if set(values) - {item.name for item in fields(CatalogConfig)}:
        raise ConfigurationError("A configuração do catálogo contém campos desconhecidos; consulte o arquivo padrão.")
    try:
        return CatalogConfig(**values)
    except TypeError:
        raise ConfigurationError("Tipos inválidos na configuração TOML do catálogo.") from None
