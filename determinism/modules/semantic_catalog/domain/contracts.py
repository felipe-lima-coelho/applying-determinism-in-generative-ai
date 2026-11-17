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


SCHEMA_VERSION = 2
_CATALOG_FIELDS = {
    "schema_version", "catalog_id", "dataset_fingerprint", "policy", "provenance",
    "segments", "candidates", "comparisons", "facts", "rejected_facts",
}
_STATUSES = {"supported", "uncertain", "not_supported"}
_V2_POLICY = {"equivalence": "threshold_gated_factual_complete_link", "support": "independent_segments",
              "uncertainty": "explicit", "canonical_statement": "cosine_medoid",
              "merge_verification": "context_and_statement_only_agreement",
              "scope": "transcript_claims_not_external_fact_checking"}


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
    version = catalog.get("schema_version") if isinstance(catalog, dict) else None
    if type(version) is not int or version not in (1, SCHEMA_VERSION):
        raise CatalogError("schema_version do catálogo não é suportada; v1 permanece legível e novas construções usam v2.")
    value = _object(catalog, _CATALOG_FIELDS | ({"clustering"} if version == 2 else set()), "catalog")
    digest = catalog_digest(value)
    if value["catalog_id"] != digest:
        raise CatalogError("catalog_id não corresponde ao conteúdo do catálogo.")
    fingerprint = value["dataset_fingerprint"]
    if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise CatalogError("dataset_fingerprint deve ser um SHA-256 hexadecimal.")
    if not isinstance(value["policy"], dict) or not isinstance(value["provenance"], dict):
        raise CatalogError("policy e provenance devem ser objetos.")
    threshold = None
    if version == 2:
        from .clustering import APPLICATION_REVISION, validate_threshold
        clustering = _object(value["clustering"], {"algorithm", "metric", "similarity_threshold", "tie_break", "representative",
                                                      "embedding_dimension", "embedding_inputs", "merges"}, "clustering")
        if (clustering["algorithm"] != "global_hac_complete_link" or clustering["metric"] != "cosine"
                or clustering["tie_break"] != "candidate_snapshot_index" or clustering["representative"] != "cosine_medoid"
                or value["policy"] != _V2_POLICY or value["provenance"].get("application_revision") != APPLICATION_REVISION):
            raise CatalogError("Política ou revisão de canonicalização v2 incompatível.")
        threshold = validate_threshold(clustering["similarity_threshold"])

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
    for index, raw in enumerate(_list(value["candidates"], "candidates")):
        path = f"candidates[{index}]"
        item = _object(raw, {"id", "segment_id", "statement", "evidence", "information_indices"}, path)
        candidate_id = _identifier(item["id"], f"{path}.id")
        if version == 2 and candidate_id != f"c{index}":
            raise CatalogError("Candidatos v2 devem seguir os índices do snapshot.")
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
        candidates[candidate_id] = item

    if version == 2:
        dimension = clustering["embedding_dimension"]
        if ((candidates and (type(dimension) is not int or not 1 <= dimension <= 16000))
                or (not candidates and dimension is not None)):
            raise CatalogError("clustering.embedding_dimension incompatível com os candidatos.")
        configuration = value["provenance"].get("configuration")
        if configuration is not None:
            if (not isinstance(configuration, dict) or configuration.get("similarity_threshold") != threshold
                    or (configuration.get("embedding_dimensions") is not None and candidates
                        and configuration["embedding_dimensions"] != dimension)):
                raise CatalogError("Configuração na proveniência difere dos parâmetros de canonicalização.")
        texts = dict.fromkeys(item["statement"] for item in candidates.values())
        expected_inputs = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
        if clustering["embedding_inputs"] != expected_inputs:
            raise CatalogError("clustering.embedding_inputs difere dos textos únicos em ordem de ocorrência.")

    comparisons: dict[frozenset[str], dict] = {}
    pair_ids: set[str] = set()
    for index, raw in enumerate(_list(value["comparisons"], "comparisons")):
        path = f"comparisons[{index}]"
        comparison_fields = {"pair_id", "left_id", "right_id", "relation", "reason"}
        if version == 2:
            comparison_fields |= {"similarity", "context_decision", "statement_decision"}
        item = _object(raw, comparison_fields, path)
        pair_id = _identifier(item["pair_id"], f"{path}.pair_id")
        left = _identifier(item["left_id"], f"{path}.left_id")
        right = _identifier(item["right_id"], f"{path}.right_id")
        if left == right or left not in candidates or right not in candidates:
            raise CatalogError(f"{path} deve comparar dois candidatos conhecidos e distintos.")
        pair = frozenset((left, right))
        if pair_id in pair_ids or pair in comparisons:
            raise CatalogError(f"{path} repete uma comparação.")
        relations = {"equivalent", "different", "uncertain"} | ({"below_threshold"} if version == 2 else set())
        if not isinstance(item["relation"], str) or item["relation"] not in relations:
            raise CatalogError(f"{path}.relation é inválida.")
        _text(item["reason"], f"{path}.reason")
        if version == 2:
            similarity = item["similarity"]
            if type(similarity) not in (int, float) or not math.isfinite(similarity) or not -1 <= similarity <= 1:
                raise CatalogError(f"{path}.similarity deve ser um cosseno finito entre -1 e 1.")
            context, statement = item["context_decision"], item["statement_decision"]
            if similarity < threshold:
                if item["relation"] != "below_threshold" or context is not None or statement is not None:
                    raise CatalogError("Par abaixo do limiar deve ser explicitamente não avaliado.")
            else:
                context = _object(context, {"relation", "reason"}, f"{path}.context_decision")
                if context["relation"] not in ("equivalent", "different", "uncertain"):
                    raise CatalogError("Decisão contextual inválida.")
                _text(context["reason"], f"{path}.context_decision.reason")
                expected_relation = context["relation"]
                if expected_relation == "equivalent":
                    statement = _object(statement, {"relation", "reason"}, f"{path}.statement_decision")
                    if statement["relation"] not in ("equivalent", "different", "uncertain"):
                        raise CatalogError("Decisão das afirmações isoladas inválida.")
                    _text(statement["reason"], f"{path}.statement_decision.reason")
                    expected_relation = statement["relation"]
                elif statement is not None:
                    raise CatalogError("Somente equivalências contextuais podem receber confirmação isolada.")
                if item["relation"] != expected_relation:
                    raise CatalogError("Relação final difere das decisões factuais preservadas.")
        comparisons[pair] = item
        pair_ids.add(pair_id)
    if len(comparisons) != len(candidates) * (len(candidates) - 1) // 2:
        raise CatalogError("comparisons deve cobrir exatamente todos os pares de candidatos.")
    if version == 2:
        expected_pairs = [(f"p{index}", left, right) for index, (left, right) in enumerate(combinations(candidates, 2))]
        if [(pair["pair_id"], pair["left_id"], pair["right_id"]) for pair in value["comparisons"]] != expected_pairs:
            raise CatalogError("Comparações v2 devem seguir a ordem canônica dos pares do snapshot.")

    fact_ids: set[str] = set()
    assigned: set[str] = set()
    for field in ("facts", "rejected_facts"):
        for index, raw in enumerate(_list(value[field], field)):
            path = f"{field}[{index}]"
            fact = _object(raw, {"id", "statement", "candidate_ids", "assessments"} |
                           ({"canonical_candidate_id"} if version == 2 else set()), path)
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
            if any(comparisons[frozenset(pair)]["relation"] != "equivalent" for pair in combinations(members, 2)):
                raise CatalogError(f"{path} agrupa candidatos sem equivalência confirmada em todos os pares.")
            supports: set[tuple[str, ...]] = set()
            singletons: set[str] = set()
            has_possible_support = False
            for position, raw_assessment in enumerate(_list(fact["assessments"], f"{path}.assessments")):
                assessment_path = f"{path}.assessments[{position}]"
                assessment = _object(raw_assessment, {"segment_ids", "status", "evidence", "reason"}, assessment_path)
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
                has_possible_support |= status in ("supported", "uncertain")
            if singletons != known_segments:
                raise CatalogError(f"{path} deve avaliar explicitamente cada segmento do snapshot.")
            if (field == "facts") != has_possible_support:
                raise CatalogError(f"{path} está na lista incorreta de informações aceitas ou rejeitadas.")
    if assigned != set(candidates):
        raise CatalogError("facts e rejected_facts devem particionar exatamente todos os candidatos.")
    if version == 2:
        from .clustering import complete_linkage
        for raw in _list(clustering["merges"], "clustering.merges"):
            merge = _object(raw, {"left_candidate_ids", "right_candidate_ids", "candidate_ids", "similarity"}, "clustering.merges[]")
            similarity = merge["similarity"]
            if type(similarity) not in (int, float) or not math.isfinite(similarity) or not -1 <= similarity <= 1:
                raise CatalogError("Escores do histórico devem ser cossenos numéricos finitos.")
        groups, merges, representatives = complete_linkage(list(candidates), value["comparisons"], threshold)
        if clustering["merges"] != merges:
            raise CatalogError("Histórico de fusões difere da hierarquia complete linkage numérica determinística.")
        actual_facts = {fact["id"]: fact for fact in value["facts"] + value["rejected_facts"]}
        if set(actual_facts) != {f"f{index}" for index in range(len(groups))}:
            raise CatalogError("IDs dos fatos v2 diferem dos grupos hierárquicos.")
        for index, (group, representative) in enumerate(zip(groups, representatives)):
            fact = actual_facts[f"f{index}"]
            if (fact["candidate_ids"] != group or fact["canonical_candidate_id"] != representative
                    or fact["statement"] != candidates[representative]["statement"]):
                raise CatalogError("Partição ou representante medoid difere da canonicalização v2 determinística.")
    return value
