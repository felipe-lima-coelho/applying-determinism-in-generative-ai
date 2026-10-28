from typing import Protocol

from .provider import CatalogResponse


class ResponseCache(Protocol):
    def get(self, key: str) -> CatalogResponse | None: ...

    def put(self, key: str, response: CatalogResponse) -> None: ...
