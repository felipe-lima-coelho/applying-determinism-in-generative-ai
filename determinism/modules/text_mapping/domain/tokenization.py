"""Política determinística de palavras, independente de arquivos ou serviços."""

from collections.abc import Iterator
from unicodedata import category


class UnicodeWordTokenizer:
    """Letras/números e marcas combinantes anexas; outros caracteres separam."""

    def tokenize(self, text: str) -> Iterator[str]:
        start: int | None = None
        for index, character in enumerate(text):
            kind = category(character)[0]
            if kind in ("L", "N") or (kind == "M" and start is not None):
                if start is None:
                    start = index
            elif start is not None:
                yield text[start:index]
                start = None
        if start is not None:
            yield text[start:]
