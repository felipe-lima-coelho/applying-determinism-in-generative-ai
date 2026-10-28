"""Parâmetros explícitos do catálogo; credenciais não entram nos artefatos."""

import tomllib
from dataclasses import dataclass, fields
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

from ...semantic_extraction.adapters.configuration import (
    ConfigurationError, ExtractionConfig, read_api_key,
)


@dataclass(frozen=True)
class CatalogConfig:
    method: str = "requirements"
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
    assessment_batch_size: int = 1
    cache_revision: str = "1"

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or self.method not in ("requirements", "legacy"):
            raise ConfigurationError("method deve ser requirements ou legacy.")
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
