"""Artefatos e cache JSON atômicos; nenhum acesso ao dataset original para escrita."""

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from ..domain import CatalogError
from ..ports.provider import CatalogResponse


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise CatalogError("JSON contém chave duplicada.")
        result[key] = value
    return result


def _constant(value):
    raise CatalogError("JSON contém número não finito.")


def read_json(path: str | Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_object, parse_constant=_constant)
    except (UnicodeError, ValueError) as error:
        raise CatalogError(f"Artefato JSON inválido: {path}.") from error


def write_json(path: str | Path, value) -> None:
    try:
        serialized = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        # JSON pode representar surrogates isolados; o artefato exige UTF-8 válido.
        serialized.encode("utf-8")
    except (ValueError, TypeError, UnicodeError):
        raise CatalogError("Artefato contém dados não serializáveis em JSON.") from None
    destination = Path(path).resolve()
    mode = stat.S_IMODE(destination.stat().st_mode) if destination.exists() else None
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=destination.parent,
                                prefix=f".{destination.name}.", suffix=".tmp", delete=False) as handle:
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


def response_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class JsonResponseCache:
    directory: Path

    def _path(self, key):
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
            raise CatalogError("Chave inválida no cache do catálogo.")
        return Path(self.directory) / f"{key}.json"

    def get(self, key: str) -> CatalogResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        data = read_json(path)
        if not isinstance(data, dict) or set(data) != {"key", "response", "digest"} or data["key"] != key:
            raise CatalogError("Registro inválido no cache do catálogo.")
        value = data["response"]
        if response_digest(value) != data["digest"]:
            raise CatalogError("Checksum inválido no cache do catálogo.")
        if not isinstance(value, dict) or set(value) != {"payload", "model", "system_fingerprint"}:
            raise CatalogError("Resposta inválida no cache do catálogo.")
        for field in ("model", "system_fingerprint"):
            if value[field] is not None and not isinstance(value[field], str):
                raise CatalogError("Metadados inválidos no cache do catálogo.")
        return CatalogResponse(**value)

    def put(self, key: str, response: CatalogResponse) -> None:
        path = self._path(key)
        value = {"payload": response.payload, "model": response.model, "system_fingerprint": response.system_fingerprint}
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, {"key": key, "response": value, "digest": response_digest(value)})
