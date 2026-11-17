"""Portas de inferência substituíveis para construir o catálogo semântico."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..domain import CatalogError


class CatalogInferenceError(CatalogError):
    """O provedor não entregou uma resposta completa e interpretável."""


@dataclass(frozen=True)
class CatalogResponse:
    payload: object
    model: str | None = None
    system_fingerprint: str | None = None


class CatalogProvider(Protocol):
    """As implementações devem permitir chamadas simultâneas sem estado mutável."""

    def extract(self, segment: dict) -> CatalogResponse: ...

    def compare(self, pairs: Sequence[dict]) -> CatalogResponse: ...

    def assess(self, segment: dict, facts: Sequence[dict]) -> CatalogResponse: ...
