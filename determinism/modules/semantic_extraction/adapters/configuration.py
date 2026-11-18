"""Configuração explícita TOML e credencial separada, sem dependências externas."""

import math
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit


class ConfigurationError(ValueError):
    """A configuração não é segura ou não atende ao contrato do adaptador."""


@dataclass(frozen=True)
class ExtractionConfig:
    provider: str = "openai"
    model: str = "gpt-6.1-terra"
    base_url: str = "https://api.openai.com/v1"
    max_concurrency: int = 4
    temperature: float = 0.0
    seed: int = 42
    reasoning_effort: str = "none"
    timeout_seconds: float = 60.0
    max_completion_tokens: int = 4096
    api_key_env: str = "OPENAI_API_KEY"
    max_generations_per_segment: int = 10

    def __post_init__(self) -> None:
        if self.provider not in ("openai", "openai-compatible"):
            raise ConfigurationError("provider deve ser 'openai' ou 'openai-compatible'; outros provedores usam a porta Python.")
        if not isinstance(self.model, str) or not self.model.strip() or any(char.isspace() for char in self.model):
            raise ConfigurationError("model deve ser um identificador não vazio, sem espaços.")
        if not isinstance(self.base_url, str):
            raise ConfigurationError("base_url deve ser uma URL HTTP(S) sem credenciais, query ou fragmento.")
        try:
            url = urlsplit(self.base_url)
            valid_url = url.scheme in ("http", "https") and url.hostname and url.port != 0
        except ValueError:
            valid_url = False
        if not valid_url or url.username or url.password or url.query or url.fragment:
            raise ConfigurationError("base_url deve ser uma URL HTTP(S) sem credenciais, query ou fragmento.")
        if self.provider == "openai" and (url.scheme != "https" or url.hostname != "api.openai.com"):
            raise ConfigurationError("provider='openai' exige HTTPS em api.openai.com; use openai-compatible para outro servidor.")
        if type(self.max_concurrency) is not int or not 1 <= self.max_concurrency <= 256:
            raise ConfigurationError("max_concurrency deve ser um inteiro entre 1 e 256.")
        if type(self.max_generations_per_segment) is not int or self.max_generations_per_segment < 1:
            raise ConfigurationError("max_generations_per_segment deve ser um inteiro positivo.")
        if type(self.temperature) not in (int, float) or not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ConfigurationError("temperature deve ser um número finito entre 0 e 2.")
        if type(self.seed) is not int or not -(2**63) <= self.seed < 2**63:
            raise ConfigurationError("seed deve ser um inteiro de 64 bits.")
        if self.reasoning_effort not in ("none", "low", "medium", "high", "xhigh", "max", "omit"):
            raise ConfigurationError("reasoning_effort inválido; 'omit' só é permitido para um servidor compatível.")
        if self.provider == "openai" and self.reasoning_effort != "none":
            raise ConfigurationError("Este experimento OpenAI exige reasoning_effort='none' para enviar temperature explicitamente.")
        if type(self.timeout_seconds) not in (int, float) or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ConfigurationError("timeout_seconds deve ser um número finito positivo.")
        if type(self.max_completion_tokens) is not int or self.max_completion_tokens <= 0:
            raise ConfigurationError("max_completion_tokens deve ser um inteiro positivo.")
        if not isinstance(self.api_key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", self.api_key_env):
            raise ConfigurationError("api_key_env deve nomear uma variável de ambiente, nunca conter uma credencial.")


def load_extraction_config(path: str | Path | None = None) -> ExtractionConfig:
    try:
        if path is None:
            text = files("determinism.experiments").joinpath("configs/semantic_extraction.toml").read_text(encoding="utf-8")
        else:
            text = Path(path).read_text(encoding="utf-8-sig")
        values = tomllib.loads(text)
    except (ValueError, UnicodeError):
        # TOMLDecodeError pode incluir conteúdo arbitrário do arquivo.
        raise ConfigurationError("Não foi possível ler a configuração TOML; verifique a sintaxe e UTF-8.") from None
    allowed = {field.name for field in fields(ExtractionConfig)}
    if set(values) - allowed:
        raise ConfigurationError("A configuração TOML contém campos desconhecidos; consulte o arquivo padrão.")
    try:
        return ExtractionConfig(**values)
    except TypeError:
        raise ConfigurationError("Tipos inválidos na configuração TOML.") from None


def read_api_key(
    config: ExtractionConfig,
    env_file: str | Path | None = ".env",
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Lê somente a credencial selecionada; ambiente tem precedência sobre .env."""
    environment = os.environ if environ is None else environ
    key = environment.get(config.api_key_env)
    if key is None and env_file is not None and Path(env_file).exists():
        try:
            lines = Path(env_file).read_text(encoding="utf-8-sig").splitlines()
        except UnicodeError:
            raise ConfigurationError("O arquivo de credenciais deve usar UTF-8.") from None
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            if name.strip() != config.api_key_env:
                continue
            value = value.strip()
            if value.startswith(("'", '"')):
                if len(value) < 2 or value[-1] != value[0]:
                    raise ConfigurationError("Aspas inválidas na credencial do arquivo .env.")
                value = value[1:-1]
            key = value
    if key is not None:
        key = key.strip()
        if any(character.isspace() for character in key) or any(ord(character) < 32 for character in key):
            raise ConfigurationError("A credencial não pode conter espaços ou caracteres de controle.")
    if not key and config.provider == "openai":
        raise ConfigurationError("Credencial OpenAI ausente; defina a variável indicada por api_key_env no ambiente ou .env.")
    return key or None
