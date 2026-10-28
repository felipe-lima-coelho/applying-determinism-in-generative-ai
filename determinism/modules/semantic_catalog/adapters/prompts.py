"""Prompts fixos e schemas versionados; conteúdo do corpus é sempre dado."""

import hashlib
import json
from importlib.resources import files


OPERATIONS = ("extract", "compare", "assess")
ALL_OPERATIONS = (*OPERATIONS, "requirements", "assess_requirements")
DYNAMIC_CONTRACT_REVISION = "complete-id-maps-v1"


def _asset(operation: str, suffix: str) -> str:
    if operation not in ALL_OPERATIONS:
        raise ValueError("Operação de catálogo desconhecida.")
    return files("determinism.modules.semantic_catalog").joinpath(
        f"assets/{operation}.{suffix}.v1.{ 'txt' if suffix == 'system' else 'json'}",
    ).read_text(encoding="utf-8")


def system_prompt(operation: str) -> str:
    return _asset(operation, "system")


def structured_output(operation: str) -> dict:
    return json.loads(_asset(operation, "schema"))


def contract_fingerprint() -> str:
    """Muda quando qualquer instrução ou schema do avaliador muda."""
    assets = [[operation, system_prompt(operation), _asset(operation, "schema")] for operation in ALL_OPERATIONS]
    raw = json.dumps([DYNAMIC_CONTRACT_REVISION, assets], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
