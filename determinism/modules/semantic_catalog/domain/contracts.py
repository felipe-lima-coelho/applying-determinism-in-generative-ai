"""Catálogo versionado e verificável: decisões semânticas preservam suas evidências."""

import hashlib
import json
import math
import re
from itertools import combinations
from typing import Any

from ...text_mapping.domain import DatasetFormatError, validate_dataset


class CatalogError(ValueError):
    """O catálogo, uma decisão semântica ou um cenário não atende ao contrato."""


SCHEMA_VERSION = 1
REQUIREMENTS_SCHEMA_VERSION = 2
_CATALOG_FIELDS = {
    "schema_version", "catalog_id", "dataset_fingerprint", "policy", "provenance",
    "segments", "candidates", "comparisons", "facts", "rejected_facts",
}
_STATUSES = {"supported", "uncertain", "not_supported"}


def _json_value(value: object) -> None:
    """Não aceita coerções silenciosas de tuplas, chaves numéricas ou NaN."""
    if value is None or type(value) in (bool, int, str):
        return
    if type(value) is float and math.isfinite(value):
        return
    if isinstance(value, list):
        for item in value:
            _json_value(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json_value(item)
        return
    raise CatalogError("O catálogo e o dataset devem conter somente valores JSON finitos.")


def _digest(value: object) -> str:
    try:
        _json_value(value)
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
    except (TypeError, UnicodeError, RecursionError):
        raise CatalogError("Não foi possível calcular o hash: JSON ou UTF-8 inválido.") from None


def dataset_digest(dataset: object) -> str:
    """Hash do snapshot inteiro, incluindo conteúdo, ordem e metadados."""
    try:
        validate_dataset(dataset)
    except DatasetFormatError as error:
        raise CatalogError(str(error)) from error
    return _digest(dataset)


def catalog_digest(catalog: object) -> str:
    """Identifica todos os dados persistidos, exceto o próprio catalog_id."""
    if not isinstance(catalog, dict):
        raise CatalogError("O catálogo deve ser um objeto.")
    return _digest({key: value for key, value in catalog.items() if key != "catalog_id"})


def decision_digest(catalog: object) -> str:
    """Hash das decisões v2, independente de razões, citações e metadados de API.

    A ordem física das respostas não muda a identidade das decisões. A escolha
    do candidato canônico, os requisitos, os agrupamentos e todos os estados de
    suporte continuam fazendo parte dela. Não certifica correção semântica.
    """
    if not isinstance(catalog, dict) or catalog.get("schema_version") != REQUIREMENTS_SCHEMA_VERSION:
        raise CatalogError("decision_digest exige um catálogo com schema_version 2.")
    try:
        candidates = [
            {"id": item["id"], "segment_id": item["segment_id"], "statement": item["statement"],
             "information_indices": sorted(item["information_indices"]),
             "requirements": sorted(item["requirements"], key=lambda requirement: requirement["id"])}
            for item in catalog["candidates"]
        ]
        pairs = [
            {"candidate_ids": sorted((item["left_id"], item["right_id"])), "relation": item["relation"]}
            for item in catalog["comparisons"]
        ]
        facts = []
        for category in ("facts", "rejected_facts"):
            for item in catalog[category]:
                assessments = [
                    {"segment_ids": sorted(assessment["segment_ids"]), "status": assessment["status"],
                     "reference_status": assessment["reference_status"],
                     "requirements": sorted(
                         [{"requirement_id": result["requirement_id"], "status": result["status"]}
                          for result in assessment["requirement_results"]],
                         key=lambda result: result["requirement_id"])}
                    for assessment in item["assessments"]
                ]
                facts.append({"category": category, "candidate_ids": sorted(item["candidate_ids"]),
                              "canonical_candidate_id": item["candidate_ids"][0], "statement": item["statement"],
                              "assessments": sorted(assessments, key=lambda assessment: assessment["segment_ids"])})
        projection = {
            "schema_version": REQUIREMENTS_SCHEMA_VERSION,
            "dataset_fingerprint": catalog["dataset_fingerprint"], "policy": catalog["policy"],
            "candidates": sorted(candidates, key=lambda item: item["id"]),
            "comparisons": sorted(pairs, key=lambda item: item["candidate_ids"]),
            "facts": sorted(facts, key=lambda item: item["candidate_ids"]),
        }
    except (KeyError, TypeError, IndexError):
        raise CatalogError("Estrutura inválida para calcular decision_digest.") from None
    return _digest(projection)


def _object(value: object, fields: set[str], path: str, *, extras: bool = False) -> dict:
    if not isinstance(value, dict) or not fields.issubset(value) or (not extras and set(value) != fields):
        raise CatalogError(f"{path} não atende aos campos obrigatórios do contrato.")
    return value


def _text(value: object, path: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise CatalogError(f"{path} deve ser uma string{' não vazia' if not empty else ''}.")
    return value


def _list(value: object, path: str) -> list:
    if not isinstance(value, list):
        raise CatalogError(f"{path} deve ser uma lista.")
    return value


def _identifier(value: object, path: str) -> str:
    result = _text(value, path)
    if any(character.isspace() or ord(character) < 32 for character in result):
        raise CatalogError(f"{path} deve ser um identificador sem espaços ou controles.")
    return result


def _ids(value: object, path: str, known: set[str], *, nonempty: bool = False, ordered: bool = False) -> list[str]:
    items = _list(value, path)
    for item in items:
        _identifier(item, path)
    if (nonempty and not items) or len(set(items)) != len(items) or not set(items).issubset(known):
        raise CatalogError(f"{path} contém referências vazias, duplicadas ou desconhecidas.")
    if ordered and items != sorted(items):
        raise CatalogError(f"{path} deve estar em ordem canônica.")
    return items


def _evidence(value: object, segments: dict[str, dict], allowed: set[str], path: str) -> set[str]:
    seen = set()
    evidenced = set()
    for position, raw in enumerate(_list(value, path)):
        item_path = f"{path}[{position}]"
        item = _object(raw, {"segment_id", "quote", "start", "end"}, item_path)
        segment_id = _identifier(item["segment_id"], f"{item_path}.segment_id")
        if segment_id not in allowed or segment_id not in segments:
            raise CatalogError(f"{item_path} referencia segmento alheio ao suporte.")
        quote = _text(item["quote"], f"{item_path}.quote")
        start, end = item["start"], item["end"]
        content = segments[segment_id]["content"]
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(content)
                or content[start:end] != quote):
            raise CatalogError(f"{item_path} não corresponde literalmente ao conteúdo e aos offsets.")
        key = segment_id, start, end
        if key in seen:
            raise CatalogError(f"{item_path} repete uma evidência.")
        seen.add(key)
        evidenced.add(segment_id)
    return evidenced


def validate_catalog(catalog: object, dataset: object = None) -> dict[str, Any]:
    """Valida snapshot, partição, comparações e suportes sem mutar a entrada.

    A validação confirma estrutura e proveniência literal, não a correção dos
    julgamentos semânticos. Um catalog_id consistente tampouco é uma assinatura.
    """
    if not isinstance(catalog, dict) or type(catalog.get("schema_version")) is not int:
        raise CatalogError("schema_version do catálogo não é suportada.")
    version = catalog["schema_version"]
    if version not in (SCHEMA_VERSION, REQUIREMENTS_SCHEMA_VERSION):
        raise CatalogError("schema_version do catálogo não é suportada.")
    structured = version == REQUIREMENTS_SCHEMA_VERSION
    value = _object(catalog, _CATALOG_FIELDS | ({"decision_id"} if structured else set()), "catalog")
    digest = catalog_digest(value)
    if value["catalog_id"] != digest:
        raise CatalogError("catalog_id não corresponde ao conteúdo do catálogo.")
    fingerprint = value["dataset_fingerprint"]
    if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise CatalogError("dataset_fingerprint deve ser um SHA-256 hexadecimal.")
    if not isinstance(value["policy"], dict) or not isinstance(value["provenance"], dict):
        raise CatalogError("policy e provenance devem ser objetos.")

    segments: dict[str, dict] = {}
    source_names: dict[str, str] = {}
    locations: set[tuple[int, int]] = set()
    for index, raw in enumerate(_list(value["segments"], "segments")):
        path = f"segments[{index}]"
        item = _object(raw, {"id", "source_id", "source_index", "segment_index", "source_name", "content", "information"}, path, extras=True)
        source_index, segment_index = item["source_index"], item["segment_index"]
        if any(type(number) is not int or number < 0 for number in (source_index, segment_index)):
            raise CatalogError(f"{path} deve ter índices inteiros não negativos.")
        if item["id"] != f"s{source_index}.seg{segment_index}" or item["source_id"] != f"s{source_index}":
            raise CatalogError(f"{path} contém IDs incompatíveis com os índices do snapshot.")
        if item["id"] in segments or (source_index, segment_index) in locations:
            raise CatalogError(f"{path} repete um segmento.")
        name = _text(item["source_name"], f"{path}.source_name", empty=True)
        if item["source_id"] in source_names and source_names[item["source_id"]] != name:
            raise CatalogError(f"{path} usa nomes diferentes para a mesma fonte.")
        source_names[item["source_id"]] = name
        _text(item["content"], f"{path}.content", empty=True)
        _list(item["information"], f"{path}.information")
        segments[item["id"]] = item
        locations.add((source_index, segment_index))
    known_segments = set(segments)
    for source_index in {location[0] for location in locations}:
        indices = {index for source, index in locations if source == source_index}
        if indices != set(range(len(indices))):
            raise CatalogError("O snapshot de uma fonte deve conter todos os seus segmentos consecutivos.")

    if dataset is not None:
        if dataset_digest(dataset) != fingerprint:
            raise CatalogError("O dataset difere do snapshot usado para construir o catálogo; reconstrua o catálogo.")
        expected = {
            (source_index, segment_index): segment
            for source_index, source in enumerate(dataset)
            for segment_index, segment in enumerate(source["segments"])
        }
        if set(expected) != locations:
            raise CatalogError("Os segmentos do catálogo não correspondem ao dataset informado.")
        for segment in segments.values():
            original = expected[segment["source_index"], segment["segment_index"]]
            if segment["content"] != original["content"] or segment["information"] != original.get("information", []):
                raise CatalogError("O conteúdo ou o histórico do snapshot difere do dataset informado.")

    candidates: dict[str, dict] = {}
    represented_indices: dict[str, set[int]] = {identifier: set() for identifier in segments}
    for index, raw in enumerate(_list(value["candidates"], "candidates")):
        path = f"candidates[{index}]"
        item = _object(raw, {"id", "segment_id", "statement", "evidence", "information_indices"}
                       | ({"requirements"} if structured else set()), path)
        candidate_id = _identifier(item["id"], f"{path}.id")
        segment_id = _identifier(item["segment_id"], f"{path}.segment_id")
        if candidate_id in candidates or segment_id not in segments:
            raise CatalogError(f"{path} repete um candidato ou referencia segmento desconhecido.")
        _text(item["statement"], f"{path}.statement")
        if not _evidence(item["evidence"], segments, {segment_id}, f"{path}.evidence"):
            raise CatalogError(f"{path} deve conter evidência literal da extração.")
        indices = _list(item["information_indices"], f"{path}.information_indices")
        if (any(type(number) is not int or not 0 <= number < len(segments[segment_id]["information"]) for number in indices)
                or indices != sorted(set(indices))):
            raise CatalogError(f"{path}.information_indices contém índices inválidos, repetidos ou fora de ordem.")
        if structured:
            from .requirements import candidate_identifier, compile_requirements, normalize_statement
            if not indices:
                raise CatalogError(f"{path} deve referenciar ao menos uma ocorrência de information.")
            if normalize_statement(item["statement"]) != item["statement"]:
                raise CatalogError(f"{path}.statement deve usar normalização canônica de Unicode e espaços.")
            for occurrence_index in indices:
                occurrence = segments[segment_id]["information"][occurrence_index]
                if (not isinstance(occurrence, dict) or "info" not in occurrence
                        or normalize_statement(occurrence["info"]) != item["statement"]):
                    raise CatalogError(f"{path}.statement deve preservar a afirmação de information, sem reescrita.")
            if represented_indices[segment_id].intersection(indices):
                raise CatalogError(f"{path} repete uma ocorrência de information em candidatos diferentes.")
            represented_indices[segment_id].update(indices)
            if candidate_id != candidate_identifier(segment_id, segments[segment_id]["content"], item["statement"]):
                raise CatalogError(f"{path}.id não corresponde à ocorrência normalizada.")
            requirements = _list(item["requirements"], f"{path}.requirements")
            for requirement in requirements:
                _object(requirement, {"id", "kind", "token_ids", "text"}, f"{path}.requirements")
            recompiled = compile_requirements(
                {"requirements": [{"kind": requirement["kind"], "token_ids": requirement["token_ids"]}
                                  for requirement in requirements]}, item["statement"])
            if requirements != recompiled:
                raise CatalogError(f"{path}.requirements não corresponde à compilação canônica dos tokens.")
        candidates[candidate_id] = item

    if structured:
        for segment_id, segment in segments.items():
            if represented_indices[segment_id] != set(range(len(segment["information"]))):
                raise CatalogError("Os candidatos v2 devem cobrir exatamente todas as ocorrências de information.")

    comparisons: dict[frozenset[str], dict] = {}
    pair_ids: set[str] = set()
    for index, raw in enumerate(_list(value["comparisons"], "comparisons")):
        path = f"comparisons[{index}]"
        item = _object(raw, {"pair_id", "left_id", "right_id", "relation", "reason"}, path)
        pair_id = _identifier(item["pair_id"], f"{path}.pair_id")
        left = _identifier(item["left_id"], f"{path}.left_id")
        right = _identifier(item["right_id"], f"{path}.right_id")
        if left == right or left not in candidates or right not in candidates:
            raise CatalogError(f"{path} deve comparar dois candidatos conhecidos e distintos.")
        pair = frozenset((left, right))
        if pair_id in pair_ids or pair in comparisons:
            raise CatalogError(f"{path} repete uma comparação.")
        if item["relation"] not in ("equivalent", "different", "uncertain"):
            raise CatalogError(f"{path}.relation é inválida.")
        _text(item["reason"], f"{path}.reason")
        comparisons[pair] = item
        pair_ids.add(pair_id)
    if len(comparisons) != len(candidates) * (len(candidates) - 1) // 2:
        raise CatalogError("comparisons deve cobrir exatamente todos os pares de candidatos.")

    fact_ids: set[str] = set()
    assigned: set[str] = set()
    for field in ("facts", "rejected_facts"):
        for index, raw in enumerate(_list(value[field], field)):
            path = f"{field}[{index}]"
            fact = _object(raw, {"id", "statement", "candidate_ids", "assessments"}, path)
            fact_id = _identifier(fact["id"], f"{path}.id")
            if fact_id in fact_ids:
                raise CatalogError(f"{path} repete um identificador de informação.")
            fact_ids.add(fact_id)
            statement = _text(fact["statement"], f"{path}.statement")
            members = _ids(fact["candidate_ids"], f"{path}.candidate_ids", set(candidates), nonempty=True)
            if assigned.intersection(members):
                raise CatalogError("Um candidato não pode pertencer a mais de uma informação.")
            assigned.update(members)
            if statement not in {candidates[member]["statement"] for member in members}:
                raise CatalogError(f"{path}.statement deve ser a formulação de um de seus candidatos.")
            if structured and statement != candidates[members[0]]["statement"]:
                raise CatalogError(f"{path}.statement deve preservar o primeiro candidato canônico.")
            if any(comparisons[frozenset(pair)]["relation"] != "equivalent" for pair in combinations(members, 2)):
                raise CatalogError(f"{path} agrupa candidatos sem equivalência confirmada em todos os pares.")
            supports: set[tuple[str, ...]] = set()
            singletons: set[str] = set()
            has_possible_support = False
            for position, raw_assessment in enumerate(_list(fact["assessments"], f"{path}.assessments")):
                assessment_path = f"{path}.assessments[{position}]"
                assessment = _object(raw_assessment, {"segment_ids", "status", "evidence", "reason"}
                                     | ({"reference_status", "requirement_results"} if structured else set()), assessment_path)
                ids = _ids(assessment["segment_ids"], f"{assessment_path}.segment_ids", known_segments, nonempty=True, ordered=True)
                support_key = tuple(ids)
                if support_key in supports:
                    raise CatalogError(f"{assessment_path} repete o mesmo conjunto de suporte.")
                supports.add(support_key)
                if len(ids) == 1:
                    singletons.add(ids[0])
                status = assessment["status"]
                if not isinstance(status, str) or status not in _STATUSES:
                    raise CatalogError(f"{assessment_path}.status é inválido.")
                _text(assessment["reason"], f"{assessment_path}.reason")
                evidenced = _evidence(assessment["evidence"], segments, set(ids), f"{assessment_path}.evidence")
                if status == "supported" and evidenced != set(ids):
                    raise CatalogError(f"{assessment_path} exige evidência para todos os segmentos que sustentam a informação.")
                if structured:
                    from .requirements import REQUIREMENT_STATUSES, reduce_requirement_status
                    if len(ids) != 1:
                        raise CatalogError(f"{assessment_path}: o contrato v2 avalia segmentos independentes.")
                    requirements = candidates[members[0]]["requirements"]
                    known_requirements = {requirement["id"] for requirement in requirements}
                    results = _list(assessment["requirement_results"], f"{assessment_path}.requirement_results")
                    seen_requirements = set()
                    merged_evidence = {}
                    states = []
                    for result in results:
                        _object(result, {"requirement_id", "status", "evidence"}, f"{assessment_path}.requirement_results")
                        requirement_id = _identifier(result["requirement_id"], f"{assessment_path}.requirement_id")
                        if requirement_id not in known_requirements or requirement_id in seen_requirements:
                            raise CatalogError(f"{assessment_path} contém requisito desconhecido ou repetido.")
                        seen_requirements.add(requirement_id)
                        requirement_status = result["status"]
                        if not isinstance(requirement_status, str) or requirement_status not in REQUIREMENT_STATUSES:
                            raise CatalogError(f"{assessment_path} contém estado de requisito inválido.")
                        states.append(requirement_status)
                        proved = _evidence(result["evidence"], segments, set(ids), f"{assessment_path}.requirement_evidence")
                        if (requirement_status == "absent" and result["evidence"]
                                or requirement_status != "absent" and proved != set(ids)):
                            raise CatalogError(f"{assessment_path}: requisito ausente exige []; demais estados exigem evidência literal.")
                        for quote in result["evidence"]:
                            merged_evidence[quote["segment_id"], quote["start"], quote["end"]] = quote
                    if seen_requirements != known_requirements:
                        raise CatalogError(f"{assessment_path} deve avaliar todos os requisitos do candidato canônico.")
                    if status != reduce_requirement_status(assessment["reference_status"], states):
                        raise CatalogError(f"{assessment_path}.status difere da redução determinística dos requisitos.")
                    expected_evidence = [merged_evidence[key] for key in sorted(merged_evidence)]
                    if assessment["evidence"] != expected_evidence:
                        raise CatalogError(f"{assessment_path}.evidence deve ser a união canônica das evidências dos requisitos.")
                has_possible_support |= status in ("supported", "uncertain")
            if singletons != known_segments:
                raise CatalogError(f"{path} deve avaliar explicitamente cada segmento do snapshot.")
            if (field == "facts") != has_possible_support:
                raise CatalogError(f"{path} está na lista incorreta de informações aceitas ou rejeitadas.")
    if assigned != set(candidates):
        raise CatalogError("facts e rejected_facts devem particionar exatamente todos os candidatos.")
    if structured and value["decision_id"] != decision_digest(value):
        raise CatalogError("decision_id não corresponde às decisões do catálogo.")
    return value
