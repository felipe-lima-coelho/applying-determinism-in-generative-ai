from collections.abc import Mapping, Sequence
from typing import Protocol

from ..domain import EmbeddingProfile, GroupingError, Vector


class EmbeddingError(GroupingError):
    """O provedor de embeddings falhou ou violou o contrato."""


class VectorStoreError(GroupingError):
    """O armazenamento não pôde ler ou persistir embeddings válidos."""


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> Sequence[Vector]: ...


class VectorStore(Protocol):
    def fetch(self, profile: EmbeddingProfile, texts: Sequence[str]) -> Mapping[str, Vector]: ...

    def put(self, profile: EmbeddingProfile, vectors: Mapping[str, Vector]) -> None: ...
