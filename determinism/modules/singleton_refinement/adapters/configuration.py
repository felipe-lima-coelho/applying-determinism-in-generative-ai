"""Somente o limite do loop; extração e agrupamento mantêm seus próprios TOMLs."""

import tomllib
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from ..domain import RefinementError


@dataclass(frozen=True)
class RefinementConfig:
    max_rounds: int = 10

    def __post_init__(self) -> None:
        if type(self.max_rounds) is not int or self.max_rounds < 1:
            raise RefinementError("max_rounds deve ser um inteiro positivo.")


def load_refinement_config(path: str | Path | None = None) -> RefinementConfig:
    try:
        text = (files("determinism.experiments").joinpath("configs/singleton_refinement.toml").read_text(encoding="utf-8")
                if path is None else Path(path).read_text(encoding="utf-8-sig"))
        values = tomllib.loads(text)
    except (ValueError, UnicodeError):
        raise RefinementError("Não foi possível ler TOML de refinamento; verifique sintaxe e UTF-8.") from None
    if set(values) - {"max_rounds"}:
        raise RefinementError("TOML de refinamento contém campos desconhecidos; use somente max_rounds.")
    try:
        return RefinementConfig(**values)
    except TypeError:
        raise RefinementError("Tipos inválidos na configuração TOML de refinamento.") from None
