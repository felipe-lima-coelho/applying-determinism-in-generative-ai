"""Requisitos ancorados na afirmação e redução pura das avaliações semânticas."""

import re
import unicodedata
from collections.abc import Iterable

from .contracts import CatalogError, _digest


REQUIREMENT_KINDS = frozenset({
    "predicate", "participant", "attribution", "polarity", "modality",
    "time", "location", "quantity", "qualification",
})
REQUIREMENT_STATUSES = frozenset({"entailed", "absent", "contradicted", "ambiguous"})
REFERENCE_STATUSES = frozenset({"compatible", "ambiguous", "different"})


def normalize_statement(statement: object) -> str:
    """Normaliza apenas Unicode/espaços, sem remover palavras ou qualificadores."""
    if not isinstance(statement, str):
        raise CatalogError("A afirmação deve ser textual.")
    try:
        statement.encode("utf-8")
    except UnicodeError:
        raise CatalogError("A afirmação deve conter UTF-8 válido.") from None
    normalized = unicodedata.normalize("NFC", " ".join(statement.split()))
    if not any(character.isalnum() for character in normalized):
        raise CatalogError("A afirmação deve conter palavras ou números.")
    return normalized


def statement_tokens(statement: str) -> list[dict]:
    """Palavras/números com offsets Unicode; pontuação permanece no texto original."""
    if not isinstance(statement, str):
        raise CatalogError("A afirmação deve ser textual.")
    normalize_statement(statement)  # Valida UTF-8 e existência de conteúdo lexical.
    return [
        {"id": index, "text": match.group(), "start": match.start(), "end": match.end()}
        for index, match in enumerate(re.finditer(r"[^\W_]+", statement, flags=re.UNICODE))
    ]


def candidate_identifier(segment_id: str, content: str, statement: str) -> str:
    """Identifica a ocorrência normalizada; não pretende provar equivalência."""
    return "c_" + _digest([segment_id, content, normalize_statement(statement)])


def _requirement_text(statement: str, tokens: list[dict], ids: list[int]) -> str:
    runs: list[list[int]] = []
    for identifier in ids:
        if runs and identifier == runs[-1][-1] + 1:
            runs[-1].append(identifier)
        else:
            runs.append([identifier])
    return " … ".join(statement[tokens[run[0]]["start"]:tokens[run[-1]]["end"]] for run in runs)


def _groups_from_token_assignments(assignments: object, token_count: int) -> list[dict]:
    """Converte o contrato HTTP com uma chave obrigatória para cada token.

    O par (kind, group) identifica o agrupamento local. Os números de grupo não
    entram na identidade persistida, que depende do tipo e dos tokens reais.
    """
    expected = {f"t{index}" for index in range(token_count)}
    if not isinstance(assignments, dict) or set(assignments) != expected:
        raise CatalogError("requirements_by_token deve conter exatamente uma chave tN para cada token.")
    groups: dict[tuple[str, int], dict] = {}
    for token_id in range(token_count):
        values = assignments[f"t{token_id}"]
        if not isinstance(values, list) or not values:
            raise CatalogError("Cada token deve possuir ao menos uma atribuição de requisito.")
        assigned_groups = set()
        for assignment in values:
            if not isinstance(assignment, dict) or set(assignment) != {"kind", "group"}:
                raise CatalogError("Cada atribuição de token deve conter somente kind e group.")
            kind, group = assignment["kind"], assignment["group"]
            if not isinstance(kind, str) or kind not in REQUIREMENT_KINDS:
                raise CatalogError("Tipo de requisito desconhecido.")
            if type(group) is not int or not 0 <= group < token_count:
                raise CatalogError("group deve ser um inteiro entre zero e o último índice de token.")
            key = kind, group
            if key in assigned_groups:
                # Atribuição idêntica não comunica outra obrigação semântica.
                continue
            assigned_groups.add(key)
            if key not in groups:
                groups[key] = {"kind": kind, "token_ids": []}
            groups[key]["token_ids"].append(token_id)
    # Números distintos também podem nomear o mesmo conjunto. A normalização
    # vale para o wireformat; o contrato persistido/programático permanece estrito.
    unique = {(group["kind"], tuple(group["token_ids"])): group for group in groups.values()}
    return list(unique.values())


def compile_requirements(payload: object, statement: str) -> list[dict]:
    """Compila agrupamentos de tokens sem permitir redação livre ou omissões.

    Cobertura lexical é uma condição estrutural necessária, não demonstra que
    o modelo identificou corretamente a estrutura ou a granularidade semântica.
    """
    if not isinstance(payload, dict) or set(payload) not in ({"requirements"}, {"requirements_by_token"}):
        raise CatalogError("A resposta deve conter somente requirements ou requirements_by_token.")
    tokens = statement_tokens(statement)
    raw = (_groups_from_token_assignments(payload["requirements_by_token"], len(tokens))
           if "requirements_by_token" in payload else payload["requirements"])
    if not isinstance(raw, list) or not raw:
        raise CatalogError("requirements deve ser uma lista não vazia.")
    covered: set[int] = set()
    seen: set[tuple[str, tuple[int, ...]]] = set()
    compiled = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"kind", "token_ids"}:
            raise CatalogError("Cada requisito deve conter somente kind e token_ids.")
        kind, values = item["kind"], item["token_ids"]
        if not isinstance(kind, str) or kind not in REQUIREMENT_KINDS:
            raise CatalogError("Tipo de requisito desconhecido.")
        if (not isinstance(values, list) or not values
                or any(type(value) is not int or not 0 <= value < len(tokens) for value in values)
                or len(set(values)) != len(values)):
            raise CatalogError("token_ids deve conter identificadores válidos, distintos e não vazios.")
        ids = sorted(values)
        key = kind, tuple(ids)
        if key in seen:
            raise CatalogError("A resposta repete um requisito.")
        seen.add(key)
        covered.update(ids)
        compiled.append({
            "id": "r_" + _digest({"kind": kind, "token_ids": ids}),
            "kind": kind, "token_ids": ids,
            "text": _requirement_text(statement, tokens, ids),
        })
    if covered != set(range(len(tokens))):
        raise CatalogError("Os requisitos devem cobrir todos os tokens da afirmação; há palavras omitidas.")
    return sorted(compiled, key=lambda item: (item["token_ids"][0], item["kind"], item["token_ids"]))


def reduce_requirement_status(reference_status: str, statuses: Iterable[str]) -> str:
    """Ausência/contradição prevalecem sobre dúvida de referência ou escopo."""
    if not isinstance(reference_status, str) or reference_status not in REFERENCE_STATUSES:
        raise CatalogError("reference_status inválido.")
    if isinstance(statuses, (str, bytes)):
        raise CatalogError("Estados dos requisitos devem ser uma coleção não vazia.")
    try:
        values = list(statuses)
    except TypeError:
        raise CatalogError("Estados dos requisitos devem ser uma coleção não vazia.") from None
    if not values or any(not isinstance(value, str) or value not in REQUIREMENT_STATUSES for value in values):
        raise CatalogError("Estado de requisito inválido ou lista vazia.")
    if reference_status == "different" or any(value in ("absent", "contradicted") for value in values):
        return "not_supported"
    if reference_status == "ambiguous" or "ambiguous" in values:
        return "uncertain"
    return "supported"
