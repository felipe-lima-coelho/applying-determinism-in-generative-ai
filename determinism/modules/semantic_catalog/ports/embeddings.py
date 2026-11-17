"""Embeddings substituíveis; mesma envoltória validável e cacheável das inferências."""

from collections.abc import Sequence
from typing import Protocol

from .provider import CatalogResponse


class CatalogEmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> CatalogResponse: ...
