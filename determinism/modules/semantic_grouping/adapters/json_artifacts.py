"""Prepara o dataset e seu catálogo antes de substituir os dois arquivos."""

import json
import os
import shutil
import stat
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from ..domain import Dataset, GroupingError
from ..ports import DatasetRepository


def global_catalog_output_path(dataset_path: str | Path) -> Path:
    path = Path(dataset_path)
    return path.with_name(f"{path.stem}.global.json")


def validate_grouping_paths(source: Path, destination: Path, global_destination: Path) -> None:
    def same(first: Path, second: Path) -> bool:
        return (os.path.normcase(str(first.resolve())) == os.path.normcase(str(second.resolve()))
                or (first.exists() and second.exists() and first.samefile(second)))
    if same(global_destination, source) or same(global_destination, destination):
        raise GroupingError("O catálogo global deve ter destino diferente da entrada e do dataset agrupado.")
    for target in (destination, global_destination):
        if not target.parent.is_dir() or target.is_dir():
            raise GroupingError("Os destinos devem ser arquivos em diretórios existentes.")


def save_grouping_artifacts(
    repository: DatasetRepository, destination: Path, dataset: Dataset,
    global_destination: Path, catalog: dict[str, Any],
) -> None:
    """Rollback de erros de escrita; duas substituições não são uma transação de FS."""
    destination, global_destination = destination.resolve(), global_destination.resolve()
    try:
        serialized = json.dumps(catalog, ensure_ascii=False, indent=4, allow_nan=False) + "\n"
    except (TypeError, ValueError):
        raise GroupingError("Não foi possível serializar o catálogo global como JSON.") from None
    pending: list[tuple[Path, Path, Path | None]] = []
    temporary: list[Path] = []
    replaced: list[tuple[Path, Path | None]] = []

    def new_temporary(target: Path) -> Path:
        with NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
            path = Path(handle.name)
        temporary.append(path)
        return path

    try:
        for target in (destination, global_destination):
            staged = new_temporary(target)
            backup = new_temporary(target) if target.exists() else None
            if backup is not None:
                shutil.copy2(target, backup)
            pending.append((target, staged, backup))
        repository.save(pending[0][1], dataset)
        with pending[1][1].open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        for target, staged, backup in pending:
            if backup is not None:
                os.chmod(staged, stat.S_IMODE(backup.stat().st_mode))
            os.replace(staged, target)
            replaced.append((target, backup))
    except BaseException:
        recovery = []
        for target, backup in reversed(replaced):
            try:
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    os.replace(backup, target)
            except OSError:
                if backup is not None:
                    temporary.remove(backup)
                    recovery.append(str(backup))
                else:
                    recovery.append(str(target))
        if recovery:
            raise GroupingError("Falha ao restaurar a gravação pareada; arquivos para recuperação: " + ", ".join(recovery)) from None
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
