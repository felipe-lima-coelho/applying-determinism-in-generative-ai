"""Contratos puros: ocorrências não são deduplicadas nem misturam segmentos."""

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, TypeAlias, cast


Dataset: TypeAlias = list[dict[str, Any]]
Vector: TypeAlias = tuple[float, ...]
SimilarityMatrix: TypeAlias = tuple[tuple[float, ...], ...]
Partition: TypeAlias = tuple[tuple[int, ...], ...]


class GroupingError(ValueError):
    """O agrupamento não pôde ser concluído integralmente."""


@dataclass(frozen=True)
class EmbeddingProfile:
    provider: str
    model: str
    base_url: str
    dimensions: int | None = None
    cache_revision: str = "1"

    def __post_init__(self) -> None:
        for value in (self.provider, self.model, self.base_url, self.cache_revision):
            if not isinstance(value, str) or not value.strip() or any(ord(char) < 32 for char in value):
                raise GroupingError("Identificadores do perfil devem ser strings não vazias sem caracteres de controle.")
            try:
                value.encode("utf-8")
            except UnicodeError:
                raise GroupingError("Identificadores do perfil devem conter UTF-8 válido.") from None
        if self.dimensions is not None and (type(self.dimensions) is not int or not 1 <= self.dimensions <= 16000):
            raise GroupingError("Dimensão do perfil deve ser um inteiro entre 1 e 16000.")

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider, "model": self.model,
            "base_url": self.base_url.rstrip("/"), "dimensions": self.dimensions,
            "encoding_format": "float", "cache_revision": self.cache_revision,
        }

    @property
    def key(self) -> str:
        payload = json.dumps(self.metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class InformationOccurrence:
    info: str
    context: str


@dataclass(frozen=True)
class SegmentInformation:
    source_index: int
    segment_index: int
    occurrences: tuple[InformationOccurrence, ...]


def validate_information_dataset(value: object) -> tuple[SegmentInformation, ...]:
    """Valida o dataset inteiro antes de acesso ao provedor ou ao cache."""
    if not isinstance(value, list):
        raise GroupingError("O dataset deve ser uma lista de fontes.")
    result = []
    for source_index, source in enumerate(value):
        source_path = f"$[{source_index}]"
        if not isinstance(source, dict) or not isinstance(source.get("segments"), list):
            raise GroupingError(f"{source_path}.segments deve ser uma lista em um objeto de fonte.")
        for segment_index, segment in enumerate(source["segments"]):
            path = f"{source_path}.segments[{segment_index}]"
            if not isinstance(segment, dict) or not isinstance(segment.get("content"), str):
                raise GroupingError(f"{path}.content deve ser uma string em um objeto de segmento.")
            items = segment.get("information")
            if not isinstance(items, list):
                raise GroupingError(f"{path}.information deve ser uma lista; execute a extração explicitamente.")
            occurrences = []
            for index, item in enumerate(items):
                item_path = f"{path}.information[{index}]"
                if not isinstance(item, dict) or not isinstance(item.get("info"), str) or not item["info"].strip():
                    raise GroupingError(f"{item_path}.info deve ser uma string não vazia.")
                if "\x00" in item["info"]:
                    raise GroupingError(f"{item_path}.info não pode conter NUL (incompatível com PostgreSQL text).")
                if not isinstance(item.get("context"), str):
                    raise GroupingError(f"{item_path}.context deve ser uma string.")
                try:
                    item["info"].encode("utf-8")
                    item["context"].encode("utf-8")
                except UnicodeError:
                    raise GroupingError(f"{item_path}.info e context devem conter Unicode UTF-8 válido.") from None
                occurrences.append(InformationOccurrence(item["info"], item["context"]))
            result.append(SegmentInformation(source_index, segment_index, tuple(occurrences)))
    return tuple(result)


def validate_vector(value: object, dimensions: int | None = None) -> Vector:
    if not isinstance(value, (list, tuple)) or not value or len(value) > 16000:
        raise GroupingError("Embedding deve ser um vetor numérico não vazio com até 16000 dimensões.")
    if dimensions is not None and len(value) != dimensions:
        raise GroupingError("Dimensão do embedding difere do perfil ou dos demais vetores.")
    try:
        if any(type(component) not in (int, float) or not math.isfinite(component) for component in value):
            raise GroupingError("Embedding contém componente não numérico ou não finito.")
        vector = tuple(float(component) for component in value)
    except OverflowError:
        raise GroupingError("Embedding contém componente fora do domínio numérico finito.") from None
    if not any(vector):
        raise GroupingError("Embedding nulo não possui similaridade cosseno definida.")
    # hypot evita overflow/underflow ao verificar vetores produzidos por outros adaptadores.
    if not math.isfinite(math.hypot(*vector)):
        raise GroupingError("Norma do embedding deve ser finita.")
    return vector


def cosine_matrix(vectors: tuple[Vector, ...]) -> SimilarityMatrix:
    """Calcula cada par de ocorrências, inclusive textos idênticos, uma vez."""
    dimension = len(vectors[0]) if vectors else None
    checked = tuple(validate_vector(vector, dimension) for vector in vectors)
    norms = tuple(math.hypot(*vector) for vector in checked)
    normalized = tuple(tuple(component / norm for component in vector) for vector, norm in zip(checked, norms))
    rows = [[1.0 if i == j else 0.0 for j in range(len(checked))] for i in range(len(checked))]
    for first in range(len(checked)):
        for second in range(first + 1, len(checked)):
            similarity = math.fsum(a * b for a, b in zip(normalized[first], normalized[second]))
            rows[first][second] = rows[second][first] = min(1.0, max(-1.0, similarity))
    return tuple(tuple(row) for row in rows)


def member_mean(index: int, group: tuple[int, ...], matrix: SimilarityMatrix) -> float:
    """Média com os outros membros; singleton usa 1.0 por convenção."""
    return math.fsum(matrix[index][other] for other in group if other != index) / (len(group) - 1) if len(group) > 1 else 1.0


def validate_partition(partition: Partition, matrix: SimilarityMatrix, threshold: float) -> Partition:
    if not isinstance(partition, tuple) or any(not isinstance(group, tuple) or not group for group in partition):
        raise GroupingError("Estratégia deve retornar uma tupla de grupos não vazios.")
    indices = [index for group in partition for index in group]
    if any(type(index) is not int for index in indices) or sorted(indices) != list(range(len(matrix))):
        raise GroupingError("Estratégia deve particionar cada ocorrência exatamente uma vez.")
    for group in partition:
        for position, first in enumerate(group):
            if any(matrix[first][second] < threshold for second in group[position + 1:]):
                raise GroupingError("Estratégia retornou grupo com um par abaixo do limiar de complete linkage.")
    # Verificação final independente: nenhuma fusão complete linkage elegível pode restar.
    for position, first in enumerate(partition):
        for second in partition[position + 1:]:
            if all(matrix[a][b] >= threshold for a in first for b in second):
                raise GroupingError("Estratégia deixou uma fusão complete linkage elegível entre grupos.")
    return cast(Partition, partition)
