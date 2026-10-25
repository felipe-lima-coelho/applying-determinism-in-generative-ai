"""Embeddings HTTP OpenAI e servidores locais com o mesmo contrato."""

import json
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..domain import GroupingError, Vector, validate_vector
from ..ports import EmbeddingError
from .configuration import GroupingConfig, GroupingConfigurationError


MAX_RESPONSE_BYTES = 64 * 1024 * 1024


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("invalid JSON constant")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


@dataclass(frozen=True)
class OpenAIEmbeddingProvider:
    config: GroupingConfig
    api_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.config.provider == "openai" and not self.api_key:
            raise GroupingConfigurationError("Credencial OpenAI ausente.")
        if self.api_key is not None and (not isinstance(self.api_key, str) or any(char.isspace() or ord(char) < 32 for char in self.api_key)):
            raise GroupingConfigurationError("Formato de credencial inválido.")

    def embed(self, texts: Sequence[str]) -> tuple[Vector, ...]:
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)) or any(not isinstance(text, str) or not text.strip() or "\x00" in text for text in texts):
            raise EmbeddingError("Inputs de embedding devem ser strings não vazias, sem NUL.")
        try:
            for text in texts:
                text.encode("utf-8")
        except UnicodeError:
            raise EmbeddingError("Inputs de embedding devem conter UTF-8 válido.") from None
        if not texts:
            return ()
        result = []
        dimension = self.config.dimensions
        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[start:start + self.config.batch_size]
            vectors = self._embed_batch(batch, dimension)
            dimension = dimension or len(vectors[0])
            result.extend(vectors)
        return tuple(result)

    def _embed_batch(self, texts: Sequence[str], dimensions: int | None) -> tuple[Vector, ...]:
        body: dict[str, Any] = {"model": self.config.model, "input": list(texts), "encoding_format": "float"}
        if self.config.dimensions is not None:
            body["dimensions"] = self.config.dimensions
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/embeddings",
            data=json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8"), headers=headers, method="POST",
        )
        try:
            with urllib.request.build_opener(_NoRedirect()).open(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
        except urllib.error.HTTPError as error:
            error.close()
            raise EmbeddingError(f"Provedor de embeddings recusou a requisição: HTTP {error.code}; verifique modelo, dimensões e limites de tokens. Nenhum retry foi feito.") from None
        except (OSError, ValueError):
            raise EmbeddingError("Falha de conexão/timeout com embeddings; detalhes de transporte suprimidos para proteger credenciais.") from None
        if len(raw) > MAX_RESPONSE_BYTES:
            raise EmbeddingError("Resposta de embeddings excedeu 64 MiB.")
        try:
            envelope = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        except (ValueError, UnicodeError):
            raise EmbeddingError("Provedor de embeddings retornou JSON inválido.") from None
        return self._parse_response(envelope, len(texts), dimensions)

    @staticmethod
    def _parse_response(envelope: object, count: int, dimensions: int | None) -> tuple[Vector, ...]:
        if not isinstance(envelope, dict) or not isinstance(envelope.get("data"), list) or len(envelope["data"]) != count:
            raise EmbeddingError("Envelope de embeddings deve conter exatamente um vetor por input.")
        by_index: dict[int, Vector] = {}
        for entry in envelope["data"]:
            if not isinstance(entry, dict) or type(entry.get("index")) is not int or not 0 <= entry["index"] < count or entry["index"] in by_index:
                raise EmbeddingError("Índices de embeddings devem identificar cada input exatamente uma vez.")
            try:
                vector = validate_vector(entry.get("embedding"), dimensions)
            except GroupingError as error:
                raise EmbeddingError(str(error)) from None
            dimensions = dimensions or len(vector)
            by_index[entry["index"]] = vector
        return tuple(by_index[index] for index in range(count))
