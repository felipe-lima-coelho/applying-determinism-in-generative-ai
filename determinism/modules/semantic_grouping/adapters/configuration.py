"""Configuração versionada do experimento; credenciais ficam no ambiente/.env."""

import math
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit

from ..domain import EmbeddingProfile, GroupingError


class GroupingConfigurationError(GroupingError):
    """Configuração inválida; mensagens nunca incluem conteúdo de credenciais."""


@dataclass(frozen=True)
class GroupingConfig:
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    base_url: str = "https://api.openai.com/v1"
    threshold: float = 0.85
    dimensions: int | None = None
    batch_size: int = 32
    timeout_seconds: float = 60.0
    api_key_env: str = "OPENAI_API_KEY"
    cache_revision: str = "1"
    max_moves: int = 10000

    def __post_init__(self) -> None:
        if self.provider not in ("openai", "openai-compatible"):
            raise GroupingConfigurationError("provider deve ser openai ou openai-compatible; outros provedores usam a porta Python.")
        if not isinstance(self.model, str) or not self.model.strip() or any(char.isspace() for char in self.model):
            raise GroupingConfigurationError("model deve ser um identificador não vazio, sem espaços.")
        try:
            url = urlsplit(self.base_url)
            valid = url.scheme in ("http", "https") and url.hostname and url.port != 0
        except (ValueError, TypeError, AttributeError):
            valid = False
        if not valid or url.username or url.password or url.query or url.fragment:
            raise GroupingConfigurationError("base_url deve ser HTTP(S), sem credenciais, query ou fragmento.")
        if self.provider == "openai" and (url.scheme != "https" or url.hostname != "api.openai.com" or url.port not in (None, 443)):
            raise GroupingConfigurationError("provider=openai exige HTTPS em api.openai.com; selecione openai-compatible para outro servidor.")
        if type(self.threshold) not in (int, float) or not math.isfinite(self.threshold) or not -1 <= self.threshold <= 1:
            raise GroupingConfigurationError("threshold deve ser um número finito entre -1 e 1.")
        if self.dimensions is not None and (type(self.dimensions) is not int or not 1 <= self.dimensions <= 16000):
            raise GroupingConfigurationError("dimensions deve ser um inteiro de 1 a 16000 quando informado.")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 2048:
            raise GroupingConfigurationError("batch_size deve ser um inteiro de 1 a 2048.")
        if type(self.timeout_seconds) not in (int, float) or not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise GroupingConfigurationError("timeout_seconds deve ser finito e positivo.")
        if not isinstance(self.api_key_env, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", self.api_key_env):
            raise GroupingConfigurationError("api_key_env deve nomear uma variável de ambiente.")
        if not isinstance(self.cache_revision, str) or not self.cache_revision.strip():
            raise GroupingConfigurationError("cache_revision deve ser uma string não vazia.")
        if type(self.max_moves) is not int or self.max_moves < 1:
            raise GroupingConfigurationError("max_moves deve ser um inteiro positivo.")
        try:
            self.profile
        except GroupingError:
            raise GroupingConfigurationError("Identificadores da configuração devem conter UTF-8 válido, sem caracteres de controle.") from None

    @property
    def profile(self) -> EmbeddingProfile:
        return EmbeddingProfile(self.provider, self.model, self.base_url, self.dimensions, self.cache_revision)


def load_grouping_config(path: str | Path | None = None) -> GroupingConfig:
    try:
        text = files("determinism.experiments").joinpath("configs/semantic_grouping.toml").read_text(encoding="utf-8") if path is None else Path(path).read_text(encoding="utf-8-sig")
        values = tomllib.loads(text)
    except (ValueError, UnicodeError):
        raise GroupingConfigurationError("Não foi possível ler TOML; verifique sintaxe e UTF-8.") from None
    if set(values) - {item.name for item in fields(GroupingConfig)}:
        raise GroupingConfigurationError("Configuração TOML contém campos desconhecidos; consulte o arquivo padrão.")
    try:
        return GroupingConfig(**values)
    except TypeError:
        raise GroupingConfigurationError("Tipos inválidos na configuração TOML.") from None


def read_environment(
    names: tuple[str, ...], env_file: str | Path | None = ".env", *, environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Lê apenas nomes selecionados; nunca altera o ambiente do processo."""
    selected: dict[str, str] = {}
    if env_file is not None and Path(env_file).exists():
        try:
            lines = Path(env_file).read_text(encoding="utf-8-sig").splitlines()
        except UnicodeError:
            raise GroupingConfigurationError("O arquivo .env deve usar UTF-8.") from None
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            name, value = name.strip(), value.strip()
            if name not in names:
                continue
            if value.startswith(("'", '"')):
                if len(value) < 2 or value[-1] != value[0]:
                    raise GroupingConfigurationError("Aspas inválidas em uma variável selecionada do .env.")
                value = value[1:-1]
            selected[name] = value
    environment = os.environ if environ is None else environ
    selected.update((name, environment[name]) for name in names if name in environment)
    return selected


def read_embedding_api_key(config: GroupingConfig, env_file: str | Path | None = ".env") -> str | None:
    key = read_environment((config.api_key_env,), env_file).get(config.api_key_env, "").strip()
    if any(character.isspace() or ord(character) < 32 for character in key):
        raise GroupingConfigurationError("Credencial contém espaços ou caracteres de controle.")
    if not key and config.provider == "openai":
        raise GroupingConfigurationError("Credencial OpenAI ausente; configure api_key_env no ambiente ou .env.")
    return key or None


@dataclass(frozen=True)
class PostgresSettings:
    host: str = "127.0.0.1"
    port: int = 55432
    dbname: str = "semantic_grouping"
    user: str = "grouping"
    password: str = field(default="", repr=False)
    connect_timeout: int = 10

    def __post_init__(self) -> None:
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise GroupingConfigurationError("GROUPING_DB_PORT deve ser um inteiro entre 1 e 65535.")
        if any(not isinstance(value, str) or not value.strip() or any(ord(char) < 32 for char in value) for value in (self.host, self.dbname, self.user, self.password)):
            raise GroupingConfigurationError("Host, database, usuário e senha PostgreSQL devem ser strings não vazias sem caracteres de controle.")
        if type(self.connect_timeout) is not int or self.connect_timeout < 1:
            raise GroupingConfigurationError("connect_timeout PostgreSQL deve ser um inteiro positivo.")


def read_postgres_settings(env_file: str | Path | None = ".env") -> PostgresSettings:
    values = read_environment(("GROUPING_DB_HOST", "GROUPING_DB_PORT", "GROUPING_DB_NAME", "GROUPING_DB_USER", "GROUPING_DB_PASSWORD"), env_file)
    try:
        port = int(values.get("GROUPING_DB_PORT", "55432"))
    except ValueError:
        raise GroupingConfigurationError("GROUPING_DB_PORT deve ser um inteiro entre 1 e 65535.") from None
    return PostgresSettings(
        values.get("GROUPING_DB_HOST", "127.0.0.1"), port,
        values.get("GROUPING_DB_NAME", "semantic_grouping"), values.get("GROUPING_DB_USER", "grouping"),
        values.get("GROUPING_DB_PASSWORD", ""),
    )
