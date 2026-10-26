"""Entradas imutáveis e respostas verificáveis, sem dependências de provedores."""

from dataclasses import dataclass
from typing import Any, TypeAlias, TypedDict


Dataset: TypeAlias = list[dict[str, Any]]


class ExtractionError(ValueError):
    """A extração não pôde ser concluída integralmente."""


class MapValidationError(ExtractionError):
    """O dataset não contém mapas completos e estruturalmente válidos."""


class ResponseValidationError(ExtractionError):
    """A resposta não atende ao contrato semântico estrutural."""


class Information(TypedDict):
    info: str
    tokens: list[int]
    context: str


@dataclass(frozen=True)
class TokenReference:
    token: int
    word: str


@dataclass(frozen=True)
class CoverageFeedback:
    """Estado numérico da tentativa; nunca contém texto do dataset ou do modelo."""

    generation: int
    pending: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if type(self.generation) is not int or self.generation < 1:
            raise ValueError("generation deve ser um inteiro positivo.")
        if not isinstance(self.pending, tuple) or not self.pending:
            raise ValueError("pending deve conter pares imutáveis de ID e contagem.")
        for pair in self.pending:
            if not isinstance(pair, tuple) or len(pair) != 2 or any(type(value) is not int or value < 0 for value in pair):
                raise ValueError("pending deve conter somente IDs e contagens inteiros não negativos.")
        if any(left[0] >= right[0] for left, right in zip(self.pending, self.pending[1:])):
            raise ValueError("Os IDs pendentes devem ser distintos e crescentes.")


@dataclass(frozen=True)
class SegmentInput:
    source_index: int
    segment_index: int
    content: str
    tokens: tuple[TokenReference, ...]
    coverage: CoverageFeedback | None = None

    @property
    def path(self) -> str:
        return f"$[{self.source_index}].segments[{self.segment_index}]"

    @property
    def token_ids(self) -> frozenset[int]:
        return frozenset(entry.token for entry in self.tokens)

    def projected_map(self) -> dict[str, dict[str, int | str]]:
        """Projeta apenas identificador e palavra; nunca expõe contagem/metadados."""
        return {
            str(entry.token): {"token": entry.token, "word": entry.word}
            for entry in self.tokens
        }


@dataclass(frozen=True)
class InferenceResult:
    payload: object
    model: str | None = None
    system_fingerprint: str | None = None


def validate_mapped_dataset(value: object) -> tuple[SegmentInput, ...]:
    """Valida TODOS os mapas antes da inferência, sem tokenizar novamente."""
    if not isinstance(value, list):
        raise MapValidationError("O dataset deve ser uma lista de fontes.")
    segments: list[SegmentInput] = []
    for source_index, source in enumerate(value):
        source_path = f"$[{source_index}]"
        if not isinstance(source, dict):
            raise MapValidationError(f"{source_path} deve ser um objeto de fonte.")
        if not isinstance(source.get("segments"), list):
            raise MapValidationError(f"{source_path}.segments deve ser uma lista.")
        for segment_index, segment in enumerate(source["segments"]):
            path = f"{source_path}.segments[{segment_index}]"
            if not isinstance(segment, dict):
                raise MapValidationError(f"{path} deve ser um objeto de segmento.")
            if not isinstance(segment.get("content"), str):
                raise MapValidationError(f"{path}.content deve ser uma string.")
            if "map" not in segment:
                raise MapValidationError(f"{path}.map é obrigatório; execute o mapeamento explicitamente.")
            mapping = segment["map"]
            if not isinstance(mapping, dict):
                raise MapValidationError(f"{path}.map deve ser um objeto.")
            if set(mapping) != {str(index) for index in range(len(mapping))}:
                raise MapValidationError(f"{path}.map deve conter IDs consecutivos a partir de zero.")
            references: list[TokenReference] = []
            for index in range(len(mapping)):
                entry_path = f"{path}.map[{index!r}]"
                entry = mapping[str(index)]
                if not isinstance(entry, dict):
                    raise MapValidationError(f"{entry_path} deve ser um objeto.")
                if type(entry.get("token")) is not int or entry["token"] != index:
                    raise MapValidationError(f"{entry_path}.token deve ser o inteiro {index}.")
                if not isinstance(entry.get("word"), str) or not entry["word"].strip():
                    raise MapValidationError(f"{entry_path}.word deve ser uma string não vazia.")
                if type(entry.get("count")) is not int or entry["count"] < 0:
                    raise MapValidationError(f"{entry_path}.count deve ser um inteiro não negativo.")
                references.append(TokenReference(index, entry["word"]))
            segments.append(SegmentInput(source_index, segment_index, segment["content"], tuple(references)))
    return tuple(segments)


def validate_information(payload: object, segment: SegmentInput) -> list[Information]:
    """Verifica o contrato além do schema do provedor; não inventa cobertura."""
    path = f"{segment.path}.information"
    if not isinstance(payload, dict) or set(payload) != {"informacoes"}:
        raise ResponseValidationError(f"{path}: a resposta deve conter somente 'informacoes'.")
    items = payload["informacoes"]
    if not isinstance(items, list):
        raise ResponseValidationError(f"{path}: 'informacoes' deve ser uma lista.")
    result: list[Information] = []
    known_ids = segment.token_ids
    for index, item in enumerate(items):
        item_path = f"{path}[{index}]"
        if not isinstance(item, dict) or set(item) != {"info", "tokens", "context"}:
            raise ResponseValidationError(f"{item_path} deve conter somente info, tokens e context.")
        for field in ("info", "context"):
            if not isinstance(item[field], str) or not any(character.isalnum() for character in item[field]):
                raise ResponseValidationError(f"{item_path}.{field} deve ser uma string com significado.")
        tokens = item["tokens"]
        if not isinstance(tokens, list) or len(tokens) < 2 or any(type(token) is not int for token in tokens):
            raise ResponseValidationError(f"{item_path}.tokens deve conter pelo menos dois IDs inteiros.")
        if any(left >= right for left, right in zip(tokens, tokens[1:])):
            raise ResponseValidationError(f"{item_path}.tokens deve ter IDs distintos em ordem crescente.")
        if not set(tokens).issubset(known_ids):
            raise ResponseValidationError(f"{item_path}.tokens contém um ID ausente no mapa original.")
        result.append({"info": item["info"], "tokens": list(tokens), "context": item["context"]})
    return result
