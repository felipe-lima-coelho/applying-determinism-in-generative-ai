"""Pipeline genérico: entrega a saída de cada etapa diretamente à próxima."""

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar


T = TypeVar("T")


class Stage(Protocol[T]):
    def __call__(self, value: T, /) -> T: ...


@dataclass(frozen=True)
class Pipeline(Generic[T]):
    stages: tuple[Stage[T], ...]

    def run(self, value: T) -> T:
        current = value
        for stage in self.stages:
            current = stage(current)
        return current
