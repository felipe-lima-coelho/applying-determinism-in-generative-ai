"""Reutiliza o transporte de embeddings existente, sem banco nem acoplamento de domínio."""

from dataclasses import dataclass

from ..domain import CatalogError
from ..ports.provider import CatalogResponse
from ...semantic_grouping.adapters.openai_embeddings import OpenAIEmbeddingProvider
from ...semantic_grouping.domain import GroupingError


@dataclass(frozen=True)
class CatalogEmbeddingAdapter:
    provider: OpenAIEmbeddingProvider

    def embed(self, texts):
        try:
            vectors = self.provider.embed(texts)
        except GroupingError as error:
            raise CatalogError(str(error)) from None
        return CatalogResponse({"embeddings": [list(vector) for vector in vectors]}, self.provider.config.model)
