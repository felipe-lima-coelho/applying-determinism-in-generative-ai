from collections.abc import Iterable
from typing import Protocol


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> Iterable[str]: ...
