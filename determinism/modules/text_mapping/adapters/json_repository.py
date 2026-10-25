"""Entrada/saída JSON UTF-8 com substituição atômica do arquivo de destino."""

import json
import os
import stat
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from ..domain import Dataset, validate_dataset


class DatasetFileError(ValueError):
    """Um arquivo não pôde ser interpretado ou serializado como JSON válido."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DatasetFileError(f"Chave JSON duplicada: {key!r}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise DatasetFileError(f"Constante inválida em JSON: {value}.")


class JsonDatasetRepository:
    def load(self, path: Path) -> Dataset:
        try:
            # Aceita também arquivos UTF-8 com BOM; a saída sempre usa UTF-8 sem BOM.
            with Path(path).open(encoding="utf-8-sig") as handle:
                data = json.load(
                    handle,
                    object_pairs_hook=_unique_object,
                    parse_constant=_reject_constant,
                )
        except (ValueError, UnicodeError) as error:
            raise DatasetFileError(f"Não foi possível ler {path}: {error}") from error
        return validate_dataset(data)

    def save(self, path: Path, dataset: Dataset) -> None:
        validate_dataset(dataset)
        try:
            serialized = json.dumps(dataset, ensure_ascii=False, indent=4, allow_nan=False) + "\n"
        except (TypeError, ValueError) as error:
            raise DatasetFileError(f"Não foi possível serializar {path}: {error}") from error

        destination = Path(path).resolve()
        mode = stat.S_IMODE(destination.stat().st_mode) if destination.exists() else None
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            if mode is not None:
                os.chmod(temporary, mode)
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
