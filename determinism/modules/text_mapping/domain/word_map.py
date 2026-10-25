"""Mapa ordenado de ocorrências, independente da política de tokenização."""

from collections.abc import Iterable
from typing import TypeAlias, TypedDict


class TokenEntry(TypedDict):
    token: int
    word: str
    count: int


WordMap: TypeAlias = dict[str, TokenEntry]


def build_word_map(words: Iterable[str]) -> WordMap:
    """Enumera ocorrências na ordem recebida, reiniciando índices e contadores."""
    return {
        str(index): {"token": index, "word": word, "count": 0}
        for index, word in enumerate(words)
    }
